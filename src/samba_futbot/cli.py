from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import deep_get, load_config
from .drive import (
    download_manifest_files,
    download_drive_file,
    find_manifest_item,
    index_public_folder,
    load_manifest,
    write_manifest,
)
from .events import detect_events
from .io_utils import read_detections, write_detections, write_events, write_json
from .metrics import summarize_tracks
from .sam3_adapter import run_sam3_video
from .tracking import track_detections
from .video import sample_frames, video_info
from .visualize import render_demo_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="samba-futbot")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index-drive", help="Indexar carpeta publica de Google Drive.")
    index.add_argument("--config", default=None)
    index.add_argument("--folder-id", default=None)
    index.add_argument("--root-name", default="Meta_Glasses")
    index.add_argument("--out", default="data/manifests/drive_index.json")
    index.set_defaults(func=cmd_index_drive)

    download = sub.add_parser("download", help="Descargar un archivo del manifest.")
    download.add_argument("--manifest", default="data/manifests/drive_index.json")
    download.add_argument("--id", default=None)
    download.add_argument("--name", default=None)
    download.add_argument("--out-dir", default="data/raw")
    download.add_argument("--max-bytes", type=int, default=None)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    download_all = sub.add_parser("download-all", help="Descargar todos los videos del manifest.")
    download_all.add_argument("--manifest", default="data/manifests/drive_index.json")
    download_all.add_argument("--out-dir", default="data/raw")
    download_all.add_argument("--extensions", default=".mov,.mp4,.avi,.mkv")
    download_all.add_argument("--strip-root", action="store_true")
    download_all.add_argument("--limit", type=int, default=None)
    download_all.add_argument("--force", action="store_true")
    download_all.set_defaults(func=cmd_download_all)

    sample = sub.add_parser("sample-frames", help="Extraer frames para calibracion.")
    sample.add_argument("--video", required=True)
    sample.add_argument("--out-dir", required=True)
    sample.add_argument("--every", type=float, default=None)
    sample.add_argument("--stride", type=int, default=None)
    sample.add_argument("--max-frames", type=int, default=None)
    sample.set_defaults(func=cmd_sample_frames)

    run = sub.add_parser("run-sam3", help="Ejecutar SAM 3/SAM 3.1 sobre un video.")
    run.add_argument("--config", default="config/default.yml")
    run.add_argument("--video", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--backend", choices=["official", "transformers"], default=None)
    run.add_argument("--model-id", default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--stride", type=int, default=None)
    run.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.set_defaults(func=cmd_run_sam3)

    track = sub.add_parser("track", help="Reparar/asignar IDs con tracker IoU.")
    track.add_argument("--detections", required=True)
    track.add_argument("--out", required=True)
    track.add_argument("--iou-threshold", type=float, default=0.25)
    track.add_argument("--max-age", type=int, default=12)
    track.set_defaults(func=cmd_track)

    events = sub.add_parser("events", help="Detectar eventos de juego.")
    events.add_argument("--tracks", required=True)
    events.add_argument("--out", required=True)
    events.add_argument("--possession-radius-px", type=float, default=90)
    events.add_argument("--collision-radius-px", type=float, default=55)
    events.add_argument("--frame-width", type=int, default=None)
    events.set_defaults(func=cmd_events)

    metrics = sub.add_parser("metrics", help="Calcular metricas operativas.")
    metrics.add_argument("--tracks", required=True)
    metrics.add_argument("--out", required=True)
    metrics.set_defaults(func=cmd_metrics)

    render = sub.add_parser("render-demo", help="Renderizar video lado a lado.")
    render.add_argument("--video", required=True)
    render.add_argument("--tracks", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--max-seconds", type=float, default=120)
    render.add_argument("--trail-length", type=int, default=45)
    render.set_defaults(func=cmd_render_demo)

    info = sub.add_parser("video-info", help="Mostrar metadata de video.")
    info.add_argument("--video", required=True)
    info.set_defaults(func=cmd_video_info)
    return parser


def cmd_index_drive(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else {}
    folder_id = args.folder_id or deep_get(config, "project.drive_root_id")
    if not folder_id:
        raise SystemExit("Provide --folder-id or project.drive_root_id in config.")
    items = index_public_folder(folder_id, root_name=args.root_name)
    write_manifest(items, args.out)
    files = [item for item in items if not item.is_folder]
    print(json.dumps({"out": args.out, "items": len(items), "files": len(files)}, indent=2))


def cmd_download(args: argparse.Namespace) -> None:
    if not args.id and not args.name:
        raise SystemExit("Provide --id or --name.")
    item = find_manifest_item(load_manifest(args.manifest), file_id=args.id, name=args.name)
    out_path = Path(args.out_dir) / item.name
    if out_path.exists() and not args.force:
        print(json.dumps({"path": str(out_path), "status": "exists"}, indent=2))
        return
    download_drive_file(item.id, out_path, max_bytes=args.max_bytes)
    print(json.dumps({"path": str(out_path), "source": item.path}, ensure_ascii=False, indent=2))


def cmd_download_all(args: argparse.Namespace) -> None:
    extensions = {ext.strip() for ext in args.extensions.split(",") if ext.strip()}

    def progress(result: dict[str, str | int]) -> None:
        print(
            f"[{result['index']}/{result['total']}] {result['status']}: "
            f"{result['path']} ({result['bytes']} bytes)",
            flush=True,
        )

    results = download_manifest_files(
        load_manifest(args.manifest),
        args.out_dir,
        extensions=extensions,
        strip_root=args.strip_root,
        limit=args.limit,
        force=args.force,
        progress=progress,
    )
    downloaded = sum(1 for result in results if result["status"] == "downloaded")
    existing = sum(1 for result in results if result["status"] == "exists")
    total_bytes = sum(int(result.get("bytes", 0)) for result in results)
    print(
        json.dumps(
            {
                "files": len(results),
                "downloaded": downloaded,
                "existing": existing,
                "bytes": total_bytes,
                "out_dir": args.out_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_sample_frames(args: argparse.Namespace) -> None:
    frames = sample_frames(
        args.video,
        args.out_dir,
        every_seconds=args.every,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(json.dumps({"frames": [str(path) for path in frames]}, indent=2))


def cmd_run_sam3(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sam_config = config.get("sam3", {})
    detections = run_sam3_video(
        args.video,
        args.out,
        prompts=sam_config.get("prompts", {}),
        backend=args.backend or sam_config.get("backend", "official"),
        model_id=args.model_id or sam_config.get("model_id", "facebook/sam3.1"),
        max_frames=args.max_frames or sam_config.get("max_frames"),
        stride=args.stride or sam_config.get("stride", 1),
        threshold=float(sam_config.get("threshold", 0.45)),
        mask_threshold=float(sam_config.get("mask_threshold", 0.5)),
        use_fa3=(
            args.use_fa3
            if args.use_fa3 is not None
            else bool(sam_config.get("use_fa3", False))
        ),
        offload_video_to_cpu=(
            args.offload_video_to_cpu
            if args.offload_video_to_cpu is not None
            else bool(sam_config.get("offload_video_to_cpu", True))
        ),
        offload_state_to_cpu=(
            args.offload_state_to_cpu
            if args.offload_state_to_cpu is not None
            else bool(sam_config.get("offload_state_to_cpu", True))
        ),
    )
    print(json.dumps({"detections": len(detections), "out": str(Path(args.out) / "detections.jsonl")}, indent=2))


def cmd_track(args: argparse.Namespace) -> None:
    tracked = track_detections(
        read_detections(args.detections),
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
    )
    write_detections(args.out, tracked)
    print(json.dumps({"tracks_out": args.out, "detections": len(tracked)}, indent=2))


def cmd_events(args: argparse.Namespace) -> None:
    events = detect_events(
        read_detections(args.tracks),
        possession_radius_px=args.possession_radius_px,
        collision_radius_px=args.collision_radius_px,
        frame_width=args.frame_width,
    )
    write_events(args.out, events)
    print(json.dumps({"events_out": args.out, "events": len(events)}, indent=2))


def cmd_metrics(args: argparse.Namespace) -> None:
    summary = summarize_tracks(read_detections(args.tracks))
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_render_demo(args: argparse.Namespace) -> None:
    out = render_demo_video(
        args.video,
        args.tracks,
        args.out,
        max_seconds=args.max_seconds,
        trail_length=args.trail_length,
    )
    print(json.dumps({"video": str(out)}, indent=2))


def cmd_video_info(args: argparse.Namespace) -> None:
    print(json.dumps(video_info(args.video), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
