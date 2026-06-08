from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .ball_refinement import refine_ball_trajectory
from .calibration import calibration_quality_report, render_calibration_frame, write_calibration_quality
from .config import deep_get, load_config
from .color_ball import detect_orange_ball
from .color_goals import detect_colored_goals, enforce_goal_frame_constraints
from .drive import (
    download_manifest_files,
    download_drive_file,
    find_manifest_item,
    index_public_folder,
    load_manifest,
    write_manifest,
)
from .events import detect_events, summarize_events
from .field_analysis import (
    analyze_field_tracks,
    load_field_calibration,
    write_field_robot_csv,
    write_field_trajectory_csv,
    write_field_zone_control_csv,
)
from .field_viz import render_field_map
from .io_utils import read_detections, read_json, write_detections, write_events, write_json
from .metrics import summarize_tracks
from .pseudolabels import export_pseudolabel_candidates
from .qa import (
    collect_quality_reports,
    evaluate_run_quality,
    write_quality_index_json,
    write_quality_index_markdown,
    write_quality_json,
    write_quality_markdown,
)
from .reporting import write_run_report
from .sam3_adapter import run_sam3_video
from .team import assign_robot_teams_from_video
from .tracking import track_detections
from .video import extract_video_clip, sample_frames, video_info
from .visualize import render_demo_video
from .windowing import (
    filter_edge_ball_detections,
    merge_detection_files,
    offset_detections,
    parse_int_list,
    write_window_manifest,
)


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
    run.add_argument("--prompt-frame-index", type=int, default=None)
    run.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    run.set_defaults(func=cmd_run_sam3)

    sweep = sub.add_parser("run-sam3-sweep", help="Ejecutar SAM 3 por ventanas.")
    sweep.add_argument("--config", default="config/default.yml")
    sweep.add_argument("--video", required=True)
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--prompt-frames", required=True, help="Lista CSV de frames ancla.")
    sweep.add_argument("--window-size", type=int, default=None)
    sweep.add_argument("--end-frame", type=int, default=None)
    sweep.add_argument("--classes", default=None, help="Clases CSV a conservar del config.")
    sweep.add_argument("--threshold", type=float, default=None)
    sweep.add_argument("--dedupe-iou", type=float, default=0.9)
    sweep.add_argument("--backend", choices=["official", "transformers"], default=None)
    sweep.add_argument("--model-id", default=None)
    sweep.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=None)
    sweep.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    sweep.set_defaults(func=cmd_run_sam3_sweep)

    merge = sub.add_parser("merge-detections", help="Fusionar JSONL de detecciones.")
    merge.add_argument("--inputs", required=True, help="Archivos JSONL separados por coma.")
    merge.add_argument("--out", required=True)
    merge.add_argument("--dedupe-iou", type=float, default=0.9)
    merge.set_defaults(func=cmd_merge_detections)

    filter_dets = sub.add_parser("filter-detections", help="Filtrar falsos positivos geometricos.")
    filter_dets.add_argument("--detections", required=True)
    filter_dets.add_argument("--out", required=True)
    filter_dets.add_argument("--frame-width", type=int, required=True)
    filter_dets.add_argument("--frame-height", type=int, required=True)
    filter_dets.add_argument("--ball-border-margin-px", type=float, default=4.0)
    filter_dets.set_defaults(func=cmd_filter_detections)

    pseudolabels = sub.add_parser(
        "export-pseudolabels",
        help="Exportar candidatos de pseudo-etiquetas desde detecciones SAM3.",
    )
    pseudolabels.add_argument("--detections", required=True)
    pseudolabels.add_argument("--out", required=True)
    pseudolabels.add_argument("--classes", default="robots,ball,goal_blue,goal_yellow")
    pseudolabels.add_argument("--min-score", type=float, default=0.60)
    pseudolabels.add_argument("--min-area", type=float, default=1.0)
    pseudolabels.add_argument("--require-mask", action=argparse.BooleanOptionalAction, default=True)
    pseudolabels.add_argument(
        "--root",
        default=None,
        help="Base para escribir mask_path relativo; por defecto usa la carpeta del JSONL.",
    )
    pseudolabels.set_defaults(func=cmd_export_pseudolabels)

    color_ball = sub.add_parser(
        "detect-orange-ball",
        help="Detectar pelota por color/forma configurable.",
    )
    color_ball.add_argument("--video", required=True)
    color_ball.add_argument("--out", required=True)
    color_ball.add_argument("--max-frames", type=int, default=None)
    color_ball.add_argument("--min-area", type=float, default=80.0)
    color_ball.add_argument("--max-area", type=float, default=2200.0)
    color_ball.add_argument("--min-circularity", type=float, default=0.45)
    color_ball.add_argument("--color-profile", default="orange")
    color_ball.add_argument("--hsv-lower", default=None, help="HSV lower bound as H,S,V.")
    color_ball.add_argument("--hsv-upper", default=None, help="HSV upper bound as H,S,V.")
    color_ball.add_argument("--context-detections", default=None)
    color_ball.add_argument("--robot-margin-px", type=float, default=8.0)
    color_ball.add_argument("--border-margin-px", type=float, default=4.0)
    color_ball.add_argument("--max-per-frame", type=int, default=1)
    color_ball.set_defaults(func=cmd_detect_orange_ball)

    refine_ball = sub.add_parser("refine-ball", help="Refinar trayectoria temporal de pelota.")
    refine_ball.add_argument("--detections", required=True)
    refine_ball.add_argument("--out", required=True)
    refine_ball.add_argument("--max-jump-px", type=float, default=45.0)
    refine_ball.add_argument("--preferred-area", type=float, default=650.0)
    refine_ball.add_argument("--score-weight", type=float, default=2.0)
    refine_ball.add_argument("--area-weight", type=float, default=1.0)
    refine_ball.add_argument("--max-candidates-per-frame", type=int, default=6)
    refine_ball.set_defaults(func=cmd_refine_ball)

    process = sub.add_parser("process-video", help="Pipeline completo: SAM3, merge, tracking, metricas y demo.")
    process.add_argument("--config", default="config/default.yml")
    process.add_argument("--video", required=True)
    process.add_argument("--results-dir", default="outputs")
    process.add_argument("--suffix", default="full-windowed-orange-v2-clipped")
    process.add_argument("--field-window-size", type=int, default=300)
    process.add_argument("--ball-window-size", type=int, default=220)
    process.add_argument("--field-step", type=int, default=300)
    process.add_argument("--ball-step", type=int, default=150)
    process.add_argument("--field-start", type=int, default=0)
    process.add_argument("--ball-start", type=int, default=150)
    process.add_argument("--field-threshold", type=float, default=0.45)
    process.add_argument("--ball-threshold", type=float, default=0.05)
    process.add_argument("--field-dedupe-iou", type=float, default=0.90)
    process.add_argument("--ball-dedupe-iou", type=float, default=0.70)
    process.add_argument("--goals", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--color-goals", action=argparse.BooleanOptionalAction, default=None)
    process.add_argument("--merge-dedupe-iou", type=float, default=0.85)
    process.add_argument("--track-iou-threshold", type=float, default=0.05)
    process.add_argument("--track-max-age", type=int, default=20)
    process.add_argument("--possession-radius-px", type=float, default=None)
    process.add_argument("--collision-radius-px", type=float, default=None)
    process.add_argument("--goal-x-margin-ratio", type=float, default=None)
    process.add_argument("--in-play-field-margin-px", type=float, default=None)
    process.add_argument("--ball-border-margin-px", type=float, default=None)
    process.add_argument("--field-calibration", default=None)
    process.add_argument("--field-analysis-out", default=None)
    process.add_argument("--field-trajectory-csv", default=None)
    process.add_argument("--field-robot-csv", default=None)
    process.add_argument("--field-zone-control-csv", default=None)
    process.add_argument("--field-map-out", default=None)
    process.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--qa-out", default=None)
    process.add_argument("--qa-report-out", default=None)
    process.add_argument("--run-report-out", default=None)
    process.add_argument("--run-manifest-out", default=None)
    process.add_argument("--field-grid-cols", type=int, default=6)
    process.add_argument("--field-grid-rows", type=int, default=4)
    process.add_argument(
        "--field-robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    process.add_argument("--trail-length", type=int, default=45)
    process.add_argument("--max-seconds", type=float, default=None)
    process.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render-narrative", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--render-analysis", action=argparse.BooleanOptionalAction, default=True)
    process.set_defaults(func=cmd_process_video)

    top_camera = sub.add_parser(
        "process-top-camera",
        help="Pipeline para camara superior: SAM3 pelota + color configurable + refinamiento.",
    )
    top_camera.add_argument("--config", default="config/default.yml")
    top_camera.add_argument("--video", required=True)
    top_camera.add_argument("--results-dir", default="outputs")
    top_camera.add_argument("--suffix", default="top-hybrid-ball-v1")
    top_camera.add_argument("--field-window-size", type=int, default=300)
    top_camera.add_argument("--field-step", type=int, default=300)
    top_camera.add_argument("--field-start", type=int, default=0)
    top_camera.add_argument("--field-threshold", type=float, default=0.45)
    top_camera.add_argument("--field-dedupe-iou", type=float, default=0.90)
    top_camera.add_argument("--goals", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--color-goals", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--sam3-ball", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--color-ball", action=argparse.BooleanOptionalAction, default=None)
    top_camera.add_argument("--ball-window-size", type=int, default=220)
    top_camera.add_argument("--ball-step", type=int, default=150)
    top_camera.add_argument("--ball-start", type=int, default=0)
    top_camera.add_argument("--ball-threshold", type=float, default=0.05)
    top_camera.add_argument("--ball-dedupe-iou", type=float, default=0.70)
    top_camera.add_argument("--merge-dedupe-iou", type=float, default=0.85)
    top_camera.add_argument("--track-iou-threshold", type=float, default=0.05)
    top_camera.add_argument("--track-max-age", type=int, default=20)
    top_camera.add_argument("--possession-radius-px", type=float, default=None)
    top_camera.add_argument("--collision-radius-px", type=float, default=None)
    top_camera.add_argument("--goal-x-margin-ratio", type=float, default=None)
    top_camera.add_argument("--in-play-field-margin-px", type=float, default=None)
    top_camera.add_argument("--ball-border-margin-px", type=float, default=None)
    top_camera.add_argument("--orange-min-area", type=float, default=300.0)
    top_camera.add_argument("--orange-max-area", type=float, default=2200.0)
    top_camera.add_argument("--orange-min-circularity", type=float, default=0.45)
    top_camera.add_argument("--orange-robot-margin-px", type=float, default=8.0)
    top_camera.add_argument("--orange-max-per-frame", type=int, default=6)
    top_camera.add_argument("--ball-color-profile", default=None)
    top_camera.add_argument("--ball-hsv-lower", default=None, help="HSV lower bound as H,S,V.")
    top_camera.add_argument("--ball-hsv-upper", default=None, help="HSV upper bound as H,S,V.")
    top_camera.add_argument("--refine-max-jump-px", type=float, default=35.0)
    top_camera.add_argument("--refine-preferred-area", type=float, default=680.0)
    top_camera.add_argument("--refine-score-weight", type=float, default=2.0)
    top_camera.add_argument("--refine-area-weight", type=float, default=1.0)
    top_camera.add_argument("--refine-max-candidates-per-frame", type=int, default=6)
    top_camera.add_argument("--field-calibration", default=None)
    top_camera.add_argument("--field-analysis-out", default=None)
    top_camera.add_argument("--field-trajectory-csv", default=None)
    top_camera.add_argument("--field-robot-csv", default=None)
    top_camera.add_argument("--field-zone-control-csv", default=None)
    top_camera.add_argument("--field-map-out", default=None)
    top_camera.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--qa-out", default=None)
    top_camera.add_argument("--qa-report-out", default=None)
    top_camera.add_argument("--run-report-out", default=None)
    top_camera.add_argument("--run-manifest-out", default=None)
    top_camera.add_argument("--field-grid-cols", type=int, default=6)
    top_camera.add_argument("--field-grid-rows", type=int, default=4)
    top_camera.add_argument(
        "--field-robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    top_camera.add_argument("--trail-length", type=int, default=45)
    top_camera.add_argument("--max-seconds", type=float, default=None)
    top_camera.add_argument("--clip-windows", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render-narrative", action=argparse.BooleanOptionalAction, default=True)
    top_camera.add_argument("--render-analysis", action=argparse.BooleanOptionalAction, default=True)
    top_camera.set_defaults(func=cmd_process_top_camera)

    field_analysis = sub.add_parser(
        "field-analysis",
        help="Convertir trayectoria de pelota a coordenadas de cancha con homografia.",
    )
    field_analysis.add_argument("--tracks", required=True)
    field_analysis.add_argument("--calibration", required=True)
    field_analysis.add_argument("--video", default=None)
    field_analysis.add_argument("--config", default="config/default.yml")
    field_analysis.add_argument("--out", required=True)
    field_analysis.add_argument("--csv-out", default=None)
    field_analysis.add_argument("--robot-csv-out", default=None)
    field_analysis.add_argument("--zone-control-csv-out", default=None)
    field_analysis.add_argument("--map-out", default=None)
    field_analysis.add_argument("--fps", type=float, default=None)
    field_analysis.add_argument("--possession-radius-px", type=float, default=90.0)
    field_analysis.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    field_analysis.add_argument("--grid-cols", type=int, default=6)
    field_analysis.add_argument("--grid-rows", type=int, default=4)
    field_analysis.add_argument(
        "--robot-anchor",
        choices=["centroid", "bottom_center"],
        default="bottom_center",
    )
    field_analysis.set_defaults(func=cmd_field_analysis)

    field_map = sub.add_parser(
        "render-field-map",
        help="Renderizar PNG tactico desde un JSON de field-analysis.",
    )
    field_map.add_argument("--analysis", required=True)
    field_map.add_argument("--out", required=True)
    field_map.add_argument("--width", type=int, default=1200)
    field_map.set_defaults(func=cmd_render_field_map)

    calibration_frame = sub.add_parser(
        "render-calibration-frame",
        help="Extraer un frame con guias para calibrar esquinas del campo.",
    )
    calibration_frame.add_argument("--video", required=True)
    calibration_frame.add_argument("--out", required=True)
    calibration_frame.add_argument("--frame-index", type=int, default=0)
    calibration_frame.add_argument("--calibration", default=None)
    calibration_frame.set_defaults(func=cmd_render_calibration_frame)

    calibration_check = sub.add_parser(
        "calibration-check",
        help="Validar una homografia/calibracion de cancha.",
    )
    calibration_check.add_argument("--calibration", required=True)
    calibration_check.add_argument("--out", default=None)
    calibration_check.add_argument("--video", default=None)
    calibration_check.add_argument("--frame-width", type=int, default=None)
    calibration_check.add_argument("--frame-height", type=int, default=None)
    calibration_check.set_defaults(func=cmd_calibration_check)

    report = sub.add_parser("summarize-run", help="Crear resumen Markdown de una corrida.")
    report.add_argument("--out", required=True)
    report.add_argument("--title", default="SAMBA FutBot Run")
    report.add_argument("--metrics", default=None)
    report.add_argument("--events", default=None)
    report.add_argument("--field-analysis", default=None)
    report.add_argument("--qa", default=None)
    report.add_argument("--demo", default=None)
    report.add_argument("--field-map", default=None)
    report.set_defaults(func=cmd_summarize_run)

    qa = sub.add_parser("qa-run", help="Evaluar calidad automatica de una corrida.")
    qa.add_argument("--out", required=True)
    qa.add_argument("--report-out", default=None)
    qa.add_argument("--metrics", default=None)
    qa.add_argument("--events", default=None)
    qa.add_argument("--field-analysis", default=None)
    qa.add_argument("--min-ball-coverage", type=float, default=None)
    qa.add_argument("--fail-ball-coverage", type=float, default=None)
    qa.add_argument("--max-ball-jump-px-frame", type=float, default=None)
    qa.add_argument("--fail-ball-jump-px-frame", type=float, default=None)
    qa.add_argument("--max-out-of-bounds-ratio", type=float, default=None)
    qa.add_argument("--fail-out-of-bounds-ratio", type=float, default=None)
    qa.add_argument("--max-unknown-team-ratio", type=float, default=None)
    qa.add_argument("--fail-unknown-team-ratio", type=float, default=None)
    qa.set_defaults(func=cmd_qa_run)

    qa_index = sub.add_parser(
        "qa-index",
        help="Indexar y ordenar reportes QA de una carpeta de resultados.",
    )
    qa_index.add_argument("--root", default="outputs")
    qa_index.add_argument("--pattern", default="*.json")
    qa_index.add_argument("--out", required=True)
    qa_index.add_argument("--report-out", default=None)
    qa_index.set_defaults(func=cmd_qa_index)

    track = sub.add_parser("track", help="Reparar/asignar IDs con tracker IoU.")
    track.add_argument("--detections", required=True)
    track.add_argument("--out", required=True)
    track.add_argument("--iou-threshold", type=float, default=0.25)
    track.add_argument("--max-age", type=int, default=12)
    track.set_defaults(func=cmd_track)

    events = sub.add_parser("events", help="Detectar eventos de juego.")
    events.add_argument("--tracks", required=True)
    events.add_argument("--out", required=True)
    events.add_argument("--summary-out", default=None)
    events.add_argument("--possession-radius-px", type=float, default=90)
    events.add_argument("--collision-radius-px", type=float, default=55)
    events.add_argument("--frame-width", type=int, default=None)
    events.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    events.set_defaults(func=cmd_events)

    event_summary = sub.add_parser(
        "event-summary",
        help="Resumir eventos en marcador candidato y conteos deportivos.",
    )
    event_summary.add_argument("--events", required=True)
    event_summary.add_argument("--out", required=True)
    event_summary.set_defaults(func=cmd_event_summary)

    metrics = sub.add_parser("metrics", help="Calcular metricas operativas.")
    metrics.add_argument("--tracks", required=True)
    metrics.add_argument("--out", required=True)
    metrics.add_argument("--fps", type=float, default=None)
    metrics.add_argument("--possession-radius-px", type=float, default=90.0)
    metrics.add_argument("--in-play-field-margin-px", type=float, default=8.0)
    metrics.set_defaults(func=cmd_metrics)

    render = sub.add_parser("render-demo", help="Renderizar video lado a lado.")
    render.add_argument("--video", required=True)
    render.add_argument("--tracks", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--events", default=None)
    render.add_argument("--max-seconds", type=float, default=120)
    render.add_argument("--trail-length", type=int, default=45)
    render.add_argument("--style", choices=["narrative", "analysis"], default="narrative")
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
        prompt_frame_index=(
            args.prompt_frame_index
            if args.prompt_frame_index is not None
            else int(sam_config.get("prompt_frame_index", 0))
        ),
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


def cmd_run_sam3_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sam_config = config.get("sam3", {})
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_frames = parse_int_list(args.prompt_frames)
    end_frame = args.end_frame or int(video_info(args.video)["frames"])
    window_size = int(args.window_size or sam_config.get("max_frames", 300))
    prompts = _filtered_prompts(sam_config.get("prompts", {}), args.classes)
    threshold = args.threshold if args.threshold is not None else float(sam_config.get("threshold", 0.45))

    detection_files: list[Path] = []
    windows: list[dict] = []
    for prompt_frame in prompt_frames:
        if prompt_frame >= end_frame:
            continue
        window_end = min(prompt_frame + window_size, end_frame)
        window_dir = output_dir / f"window_{prompt_frame:06d}_{window_end:06d}"
        source_video: str | Path = args.video
        source_max_frames = window_end
        source_prompt_frame = prompt_frame
        frame_offset = 0
        clip_path: Path | None = None
        if args.clip_windows:
            clip_path = output_dir / "clips" / f"window_{prompt_frame:06d}_{window_end:06d}.mp4"
            extract_video_clip(
                args.video,
                clip_path,
                start_frame=prompt_frame,
                end_frame=window_end,
            )
            source_video = clip_path
            source_max_frames = window_end - prompt_frame
            source_prompt_frame = 0
            frame_offset = prompt_frame
        detections = run_sam3_video(
            source_video,
            window_dir,
            prompts=prompts,
            backend=args.backend or sam_config.get("backend", "official"),
            model_id=args.model_id or sam_config.get("model_id", "facebook/sam3"),
            max_frames=source_max_frames,
            stride=sam_config.get("stride", 1),
            threshold=threshold,
            mask_threshold=float(sam_config.get("mask_threshold", 0.5)),
            prompt_frame_index=source_prompt_frame,
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
        detections_path = window_dir / "detections.jsonl"
        detections = offset_detections(detections, frame_offset)
        write_detections(detections_path, detections)
        detection_files.append(detections_path)
        windows.append(
            {
                "prompt_frame": prompt_frame,
                "end_frame": window_end,
                "clip_path": str(clip_path) if clip_path else None,
                "detections": len(detections),
                "detections_path": str(detections_path),
            }
        )

    merged = merge_detection_files(
        detection_files,
        output_dir / "detections.jsonl",
        iou_threshold=args.dedupe_iou,
    )
    write_window_manifest(
        output_dir / "manifest.json",
        video=args.video,
        windows=windows,
        detections=len(merged),
    )
    print(
        json.dumps(
            {
                "out": str(output_dir / "detections.jsonl"),
                "detections": len(merged),
                "windows": len(windows),
            },
            indent=2,
        )
    )


def cmd_merge_detections(args: argparse.Namespace) -> None:
    inputs = [Path(part.strip()) for part in args.inputs.split(",") if part.strip()]
    merged = merge_detection_files(inputs, args.out, iou_threshold=args.dedupe_iou)
    print(json.dumps({"out": args.out, "inputs": len(inputs), "detections": len(merged)}, indent=2))


def cmd_filter_detections(args: argparse.Namespace) -> None:
    detections = read_detections(args.detections)
    filtered = filter_edge_ball_detections(
        detections,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        border_margin_px=args.ball_border_margin_px,
    )
    write_detections(args.out, filtered)
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": len(detections),
                "detections": len(filtered),
                "removed": len(detections) - len(filtered),
            },
            indent=2,
        )
    )


def cmd_export_pseudolabels(args: argparse.Namespace) -> None:
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]
    manifest = export_pseudolabel_candidates(
        args.detections,
        args.out,
        classes=classes,
        min_score=args.min_score,
        min_area=args.min_area,
        require_mask=args.require_mask,
        root=args.root,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": manifest["summary"]["input_detections"],
                "candidates": manifest["summary"]["candidates"],
                "candidates_by_class": manifest["summary"]["candidates_by_class"],
                "rejected": manifest["summary"]["rejected"],
            },
            indent=2,
        )
    )


def cmd_detect_orange_ball(args: argparse.Namespace) -> None:
    _, hsv_lower, hsv_upper = _resolve_ball_color_profile(
        {"default_profile": args.color_profile},
        profile=args.color_profile,
        hsv_lower=args.hsv_lower,
        hsv_upper=args.hsv_upper,
    )
    detections = detect_orange_ball(
        args.video,
        args.out,
        max_frames=args.max_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        color_profile=args.color_profile,
        context_detections_path=args.context_detections,
        robot_margin_px=args.robot_margin_px,
        border_margin_px=args.border_margin_px,
        max_per_frame=args.max_per_frame,
    )
    print(json.dumps({"out": args.out, "detections": len(detections)}, indent=2))


def cmd_refine_ball(args: argparse.Namespace) -> None:
    detections = read_detections(args.detections)
    refined = refine_ball_trajectory(
        detections,
        max_jump_px=args.max_jump_px,
        preferred_area=args.preferred_area,
        score_weight=args.score_weight,
        area_weight=args.area_weight,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    write_detections(args.out, refined)
    ball_in = sum(1 for det in detections if det.class_name in {"ball", "balon", "soccer_ball"})
    ball_out = sum(1 for det in refined if det.class_name in {"ball", "balon", "soccer_ball"})
    print(
        json.dumps(
            {
                "out": args.out,
                "input_detections": len(detections),
                "detections": len(refined),
                "input_ball_detections": ball_in,
                "ball_detections": ball_out,
            },
            indent=2,
        )
    )


def cmd_process_video(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    analysis_config = config.get("analysis", {})
    info = video_info(args.video)
    end_frame = int(info["frames"])
    fps = float(info.get("fps") or analysis_config.get("fps", 30))
    duration_seconds = info.get("duration_seconds")
    stem = Path(args.video).stem
    results_dir = Path(args.results_dir)

    field_prompt_frames = _frame_anchors(
        end_frame,
        start=args.field_start,
        step=args.field_step,
    )
    ball_prompt_frames = _frame_anchors(
        end_frame,
        start=args.ball_start,
        step=args.ball_step,
    )

    field_out = results_dir / "detections" / f"{stem}-field-robots-sweep-clipped"
    ball_out = results_dir / "detections" / f"{stem}-ball-sweep-orange-v2-clipped"
    color_goals_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-goals.jsonl"
    merged_out = results_dir / "detections" / f"{stem}-{args.suffix}" / "detections.jsonl"
    tracks_out = results_dir / "tracks" / f"{stem}-{args.suffix}-tracks.jsonl"
    metrics_out = results_dir / "metrics" / f"{stem}-{args.suffix}-metrics.json"
    events_out = results_dir / "events" / f"{stem}-{args.suffix}-events.json"
    event_summary_out = results_dir / "events" / f"{stem}-{args.suffix}-event-summary.json"

    field_detections = _run_sweep_for_process(
        args,
        classes=_context_classes(args),
        prompt_frames=field_prompt_frames,
        window_size=args.field_window_size,
        end_frame=end_frame,
        out=field_out,
        threshold=args.field_threshold,
        dedupe_iou=args.field_dedupe_iou,
    )
    _run_sweep_for_process(
        args,
        classes="ball",
        prompt_frames=ball_prompt_frames,
        window_size=args.ball_window_size,
        end_frame=end_frame,
        out=ball_out,
        threshold=args.ball_threshold,
        dedupe_iou=args.ball_dedupe_iou,
    )
    color_goal_detections = _detect_color_goals_for_process(
        args,
        config,
        end_frame=end_frame,
        out=color_goals_out,
        seed_detections=field_detections,
    )

    merge_inputs = [field_out / "detections.jsonl", ball_out / "detections.jsonl"]
    if color_goal_detections is not None:
        merge_inputs.append(color_goals_out)
    merged = merge_detection_files(merge_inputs, merged_out, iou_threshold=args.merge_dedupe_iou)
    ball_border_margin_px = (
        args.ball_border_margin_px
        if args.ball_border_margin_px is not None
        else float(analysis_config.get("ball_border_margin_px", 4))
    )
    filtered = filter_edge_ball_detections(
        merged,
        frame_width=int(info.get("width") or 0) or None,
        frame_height=int(info.get("height") or 0) or None,
        border_margin_px=ball_border_margin_px,
    )
    edge_ball_filter_removed = len(merged) - len(filtered)
    if len(filtered) != len(merged):
        write_detections(merged_out, filtered)
    merged = filtered
    constrained = _enforce_goal_constraints_for_process(merged, config)
    goal_constraints_removed = len(merged) - len(constrained)
    if goal_constraints_removed:
        write_detections(merged_out, constrained)
    merged = constrained
    tracked = track_detections(
        merged,
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
    )
    tracked = _assign_teams_for_process(args.video, tracked, config)
    write_detections(tracks_out, tracked)

    possession_radius_px = (
        args.possession_radius_px
        if args.possession_radius_px is not None
        else float(analysis_config.get("possession_radius_px", 90))
    )
    field_margin_px = (
        args.in_play_field_margin_px
        if args.in_play_field_margin_px is not None
        else float(analysis_config.get("in_play_field_margin_px", 8))
    )
    summary = summarize_tracks(
        tracked,
        fps=fps,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    write_json(metrics_out, summary)
    events = detect_events(
        tracked,
        possession_radius_px=possession_radius_px,
        collision_radius_px=(
            args.collision_radius_px
            if args.collision_radius_px is not None
            else float(analysis_config.get("collision_radius_px", 55))
        ),
        frame_width=int(info.get("width") or 0) or None,
        goal_x_margin_ratio=(
            args.goal_x_margin_ratio
            if args.goal_x_margin_ratio is not None
            else float(analysis_config.get("goal_x_margin_ratio", 0.08))
        ),
        field_margin_px=field_margin_px,
    )
    write_events(events_out, events)
    event_summary = summarize_events(events)
    write_json(event_summary_out, event_summary)
    field_analysis_result = None
    field_analysis_out = None
    field_trajectory_csv = None
    field_robot_csv = None
    field_zone_control_csv = None
    field_map_out = None
    if args.field_calibration:
        field_analysis_out = Path(
            args.field_analysis_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-analysis.json"
        )
        field_trajectory_csv = Path(
            args.field_trajectory_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-trajectory.csv"
        )
        field_robot_csv = Path(
            args.field_robot_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-robots.csv"
        )
        field_zone_control_csv = Path(
            args.field_zone_control_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-zone-control.csv"
        )
        field_map_out = Path(
            args.field_map_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-map.png"
        )
        field_analysis_result = analyze_field_tracks(
            tracked,
            load_field_calibration(args.field_calibration),
            fps=fps,
            possession_radius_px=possession_radius_px,
            field_margin_px=field_margin_px,
            grid_cols=args.field_grid_cols,
            grid_rows=args.field_grid_rows,
            robot_anchor=args.field_robot_anchor,
        )
        write_json(field_analysis_out, field_analysis_result)
        write_field_trajectory_csv(field_trajectory_csv, field_analysis_result)
        write_field_robot_csv(field_robot_csv, field_analysis_result)
        write_field_zone_control_csv(field_zone_control_csv, field_analysis_result)
        render_field_map(field_analysis_result, field_map_out)

    rendered_videos = _render_pipeline_videos(
        args,
        video_path=args.video,
        tracks_out=tracks_out,
        events_out=events_out,
        results_dir=results_dir,
        stem=stem,
        duration_seconds=duration_seconds,
    )
    rendered = rendered_videos.get("narrative") or rendered_videos.get("analysis")

    qa_out, qa_report_out, qa_report = _write_pipeline_qa(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
    )
    run_report_out = _write_pipeline_run_report(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
        qa_out=qa_out,
        rendered=rendered,
        field_map_out=field_map_out,
    )
    run_manifest_out = _write_pipeline_manifest(
        args,
        results_dir=results_dir,
        stem=stem,
        artifacts={
            "detections": merged_out,
            "color_goals": color_goals_out if color_goal_detections is not None else None,
            "tracks": tracks_out,
            "metrics": metrics_out,
            "events": events_out,
            "event_summary": event_summary_out,
            "field_analysis": field_analysis_out,
            "field_trajectory_csv": field_trajectory_csv,
            "field_robot_csv": field_robot_csv,
            "field_zone_control_csv": field_zone_control_csv,
            "field_map": field_map_out,
            "qa": qa_out,
            "qa_report": qa_report_out,
            "run_report": run_report_out,
            "demo": rendered,
            "narrative_demo": rendered_videos.get("narrative"),
            "analysis_demo": rendered_videos.get("analysis"),
        },
        metrics_summary=summary,
        event_summary=event_summary,
        field_analysis_summary=(
            field_analysis_result.get("summary") if field_analysis_result else None
        ),
        qa_status=qa_report.get("status") if qa_report else None,
    )

    print(
        json.dumps(
            {
                "video": args.video,
                "frames": end_frame,
                "field_prompt_frames": field_prompt_frames,
                "ball_prompt_frames": ball_prompt_frames,
                "detections": len(merged),
                "edge_ball_filter_removed": edge_ball_filter_removed,
                "goal_constraints_removed": goal_constraints_removed,
                "tracks": len({det.track_id for det in tracked if det.track_id is not None}),
                "paths": {
                    "detections": str(merged_out),
                    "color_goals": str(color_goals_out) if color_goal_detections is not None else None,
                    "tracks": str(tracks_out),
                    "metrics": str(metrics_out),
                    "events": str(events_out),
                    "event_summary": str(event_summary_out),
                    "field_analysis": str(field_analysis_out) if field_analysis_out else None,
                    "field_trajectory_csv": (
                        str(field_trajectory_csv) if field_trajectory_csv else None
                    ),
                    "field_robot_csv": str(field_robot_csv) if field_robot_csv else None,
                    "field_zone_control_csv": (
                        str(field_zone_control_csv) if field_zone_control_csv else None
                    ),
                    "field_map": str(field_map_out) if field_map_out else None,
                    "qa": str(qa_out) if qa_out else None,
                    "qa_report": str(qa_report_out) if qa_report_out else None,
                    "run_report": str(run_report_out) if run_report_out else None,
                    "run_manifest": str(run_manifest_out) if run_manifest_out else None,
                    "demo": str(rendered) if rendered else None,
                    "narrative_demo": (
                        str(rendered_videos["narrative"])
                        if rendered_videos.get("narrative")
                        else None
                    ),
                    "analysis_demo": (
                        str(rendered_videos["analysis"])
                        if rendered_videos.get("analysis")
                        else None
                    ),
                },
                "metrics": summary,
                "field_analysis_summary": (
                    field_analysis_result.get("summary") if field_analysis_result else None
                ),
                "qa_status": qa_report.get("status") if qa_report else None,
                "events": len(events),
                "event_summary": event_summary,
                "color_goal_detections": (
                    len(color_goal_detections) if color_goal_detections is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_process_top_camera(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    analysis_config = config.get("analysis", {})
    ball_detection_config = config.get("ball_detection", {})
    sam3_ball_enabled = (
        bool(ball_detection_config.get("sam3_enabled", True))
        if args.sam3_ball is None
        else bool(args.sam3_ball)
    )
    color_ball_enabled = (
        bool(ball_detection_config.get("color_enabled", True))
        if args.color_ball is None
        else bool(args.color_ball)
    )
    if not sam3_ball_enabled and not color_ball_enabled:
        raise SystemExit("Enable at least one ball source: --sam3-ball or --color-ball.")

    info = video_info(args.video)
    end_frame = int(info["frames"])
    fps = float(info.get("fps") or analysis_config.get("fps", 30))
    duration_seconds = info.get("duration_seconds")
    stem = Path(args.video).stem
    results_dir = Path(args.results_dir)

    field_prompt_frames = _frame_anchors(
        end_frame,
        start=args.field_start,
        step=args.field_step,
    )
    ball_prompt_frames = _frame_anchors(
        end_frame,
        start=args.ball_start,
        step=args.ball_step,
    )

    field_out = results_dir / "detections" / f"{stem}-field-robots-sweep-clipped"
    sam3_ball_out = results_dir / "detections" / f"{stem}-{args.suffix}-sam3-ball-sweep-clipped"
    color_ball_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-ball.jsonl"
    color_goals_out = results_dir / "detections" / f"{stem}-{args.suffix}-color-goals.jsonl"
    merged_dir = results_dir / "detections" / f"{stem}-{args.suffix}"
    merged_out = merged_dir / "detections.jsonl"
    refined_out = merged_dir / "detections-refined.jsonl"
    tracks_out = results_dir / "tracks" / f"{stem}-{args.suffix}-tracks.jsonl"
    metrics_out = results_dir / "metrics" / f"{stem}-{args.suffix}-metrics.json"
    events_out = results_dir / "events" / f"{stem}-{args.suffix}-events.json"
    event_summary_out = results_dir / "events" / f"{stem}-{args.suffix}-event-summary.json"

    field_detections = _run_sweep_for_process(
        args,
        classes=_context_classes(args),
        prompt_frames=field_prompt_frames,
        window_size=args.field_window_size,
        end_frame=end_frame,
        out=field_out,
        threshold=args.field_threshold,
        dedupe_iou=args.field_dedupe_iou,
    )

    sam3_ball_detections = []
    if sam3_ball_enabled:
        sam3_ball_detections = _run_sweep_for_process(
            args,
            classes="ball",
            prompt_frames=ball_prompt_frames,
            window_size=args.ball_window_size,
            end_frame=end_frame,
            out=sam3_ball_out,
            threshold=args.ball_threshold,
            dedupe_iou=args.ball_dedupe_iou,
        )

    ball_border_margin_px = (
        args.ball_border_margin_px
        if args.ball_border_margin_px is not None
        else float(analysis_config.get("ball_border_margin_px", 4))
    )
    color_profile, hsv_lower, hsv_upper = _resolve_ball_color_profile(
        ball_detection_config,
        profile=args.ball_color_profile,
        hsv_lower=args.ball_hsv_lower,
        hsv_upper=args.ball_hsv_upper,
    )
    color_ball_detections = []
    if color_ball_enabled:
        color_ball_detections = detect_orange_ball(
            args.video,
            color_ball_out,
            max_frames=end_frame,
            min_area=args.orange_min_area,
            max_area=args.orange_max_area,
            min_circularity=args.orange_min_circularity,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            color_profile=color_profile,
            context_detections_path=field_out / "detections.jsonl",
            robot_margin_px=args.orange_robot_margin_px,
            border_margin_px=ball_border_margin_px,
            max_per_frame=args.orange_max_per_frame,
        )
    color_goal_detections = _detect_color_goals_for_process(
        args,
        config,
        end_frame=end_frame,
        out=color_goals_out,
        seed_detections=field_detections,
    )

    merge_inputs = [field_out / "detections.jsonl"]
    if sam3_ball_enabled:
        merge_inputs.append(sam3_ball_out / "detections.jsonl")
    if color_ball_enabled:
        merge_inputs.append(color_ball_out)
    if color_goal_detections is not None:
        merge_inputs.append(color_goals_out)
    merged = merge_detection_files(
        merge_inputs,
        merged_out,
        iou_threshold=args.merge_dedupe_iou,
    )
    refined = refine_ball_trajectory(
        merged,
        max_jump_px=args.refine_max_jump_px,
        preferred_area=args.refine_preferred_area,
        score_weight=args.refine_score_weight,
        area_weight=args.refine_area_weight,
        max_candidates_per_frame=args.refine_max_candidates_per_frame,
    )
    constrained_refined = _enforce_goal_constraints_for_process(refined, config)
    goal_constraints_removed = len(refined) - len(constrained_refined)
    refined = constrained_refined
    write_detections(refined_out, refined)

    tracked = track_detections(
        refined,
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
    )
    tracked = _assign_teams_for_process(args.video, tracked, config)
    write_detections(tracks_out, tracked)

    possession_radius_px = (
        args.possession_radius_px
        if args.possession_radius_px is not None
        else float(analysis_config.get("possession_radius_px", 90))
    )
    field_margin_px = (
        args.in_play_field_margin_px
        if args.in_play_field_margin_px is not None
        else float(analysis_config.get("in_play_field_margin_px", 8))
    )
    summary = summarize_tracks(
        tracked,
        fps=fps,
        possession_radius_px=possession_radius_px,
        field_margin_px=field_margin_px,
    )
    write_json(metrics_out, summary)
    events = detect_events(
        tracked,
        possession_radius_px=possession_radius_px,
        collision_radius_px=(
            args.collision_radius_px
            if args.collision_radius_px is not None
            else float(analysis_config.get("collision_radius_px", 55))
        ),
        frame_width=int(info.get("width") or 0) or None,
        goal_x_margin_ratio=(
            args.goal_x_margin_ratio
            if args.goal_x_margin_ratio is not None
            else float(analysis_config.get("goal_x_margin_ratio", 0.08))
        ),
        field_margin_px=field_margin_px,
    )
    write_events(events_out, events)
    event_summary = summarize_events(events)
    write_json(event_summary_out, event_summary)

    field_analysis_result = None
    field_analysis_out = None
    field_trajectory_csv = None
    field_robot_csv = None
    field_zone_control_csv = None
    field_map_out = None
    if args.field_calibration:
        field_analysis_out = Path(
            args.field_analysis_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-analysis.json"
        )
        field_trajectory_csv = Path(
            args.field_trajectory_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-trajectory.csv"
        )
        field_robot_csv = Path(
            args.field_robot_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-robots.csv"
        )
        field_zone_control_csv = Path(
            args.field_zone_control_csv
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-zone-control.csv"
        )
        field_map_out = Path(
            args.field_map_out
            or results_dir / "field_analysis" / f"{stem}-{args.suffix}-field-map.png"
        )
        field_analysis_result = analyze_field_tracks(
            tracked,
            load_field_calibration(args.field_calibration),
            fps=fps,
            possession_radius_px=possession_radius_px,
            field_margin_px=field_margin_px,
            grid_cols=args.field_grid_cols,
            grid_rows=args.field_grid_rows,
            robot_anchor=args.field_robot_anchor,
        )
        write_json(field_analysis_out, field_analysis_result)
        write_field_trajectory_csv(field_trajectory_csv, field_analysis_result)
        write_field_robot_csv(field_robot_csv, field_analysis_result)
        write_field_zone_control_csv(field_zone_control_csv, field_analysis_result)
        render_field_map(field_analysis_result, field_map_out)

    rendered_videos = _render_pipeline_videos(
        args,
        video_path=args.video,
        tracks_out=tracks_out,
        events_out=events_out,
        results_dir=results_dir,
        stem=stem,
        duration_seconds=duration_seconds,
    )
    rendered = rendered_videos.get("narrative") or rendered_videos.get("analysis")

    qa_out, qa_report_out, qa_report = _write_pipeline_qa(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
    )
    run_report_out = _write_pipeline_run_report(
        args,
        results_dir=results_dir,
        stem=stem,
        metrics_out=metrics_out,
        events_out=events_out,
        field_analysis_out=field_analysis_out,
        qa_out=qa_out,
        rendered=rendered,
        field_map_out=field_map_out,
    )
    run_manifest_out = _write_pipeline_manifest(
        args,
        results_dir=results_dir,
        stem=stem,
        artifacts={
            "field_detections": field_out / "detections.jsonl",
            "sam3_ball_detections": (
                sam3_ball_out / "detections.jsonl" if sam3_ball_enabled else None
            ),
            "color_ball_detections": color_ball_out if color_ball_enabled else None,
            "color_goal_detections": (
                color_goals_out if color_goal_detections is not None else None
            ),
            "detections": merged_out,
            "refined_detections": refined_out,
            "tracks": tracks_out,
            "metrics": metrics_out,
            "events": events_out,
            "event_summary": event_summary_out,
            "field_analysis": field_analysis_out,
            "field_trajectory_csv": field_trajectory_csv,
            "field_robot_csv": field_robot_csv,
            "field_zone_control_csv": field_zone_control_csv,
            "field_map": field_map_out,
            "qa": qa_out,
            "qa_report": qa_report_out,
            "run_report": run_report_out,
            "demo": rendered,
            "narrative_demo": rendered_videos.get("narrative"),
            "analysis_demo": rendered_videos.get("analysis"),
        },
        metrics_summary=summary,
        event_summary=event_summary,
        field_analysis_summary=(
            field_analysis_result.get("summary") if field_analysis_result else None
        ),
        qa_status=qa_report.get("status") if qa_report else None,
    )

    ball_in = sum(1 for det in merged if det.class_name in {"ball", "balon", "soccer_ball"})
    ball_out = sum(1 for det in refined if det.class_name in {"ball", "balon", "soccer_ball"})
    print(
        json.dumps(
            {
                "video": args.video,
                "frames": end_frame,
                "field_prompt_frames": field_prompt_frames,
                "ball_sources": {
                    "sam3": sam3_ball_enabled,
                    "color": color_ball_enabled,
                },
                "ball_prompt_frames": ball_prompt_frames if sam3_ball_enabled else [],
                "sam3_ball_candidates": len(sam3_ball_detections),
                "color_ball_candidates": len(color_ball_detections),
                "color_goal_candidates": (
                    len(color_goal_detections) if color_goal_detections is not None else None
                ),
                "ball_color_profile": color_profile if color_ball_enabled else None,
                "ball_candidates_before_refine": ball_in,
                "ball_detections_after_refine": ball_out,
                "goal_constraints_removed": goal_constraints_removed,
                "detections": len(refined),
                "tracks": len({det.track_id for det in tracked if det.track_id is not None}),
                "paths": {
                    "field_detections": str(field_out / "detections.jsonl"),
                    "sam3_ball_detections": (
                        str(sam3_ball_out / "detections.jsonl") if sam3_ball_enabled else None
                    ),
                    "color_ball_detections": str(color_ball_out) if color_ball_enabled else None,
                    "color_goal_detections": (
                        str(color_goals_out) if color_goal_detections is not None else None
                    ),
                    "detections": str(merged_out),
                    "refined_detections": str(refined_out),
                    "tracks": str(tracks_out),
                    "metrics": str(metrics_out),
                    "events": str(events_out),
                    "event_summary": str(event_summary_out),
                    "field_analysis": str(field_analysis_out) if field_analysis_out else None,
                    "field_trajectory_csv": (
                        str(field_trajectory_csv) if field_trajectory_csv else None
                    ),
                    "field_robot_csv": str(field_robot_csv) if field_robot_csv else None,
                    "field_zone_control_csv": (
                        str(field_zone_control_csv) if field_zone_control_csv else None
                    ),
                    "field_map": str(field_map_out) if field_map_out else None,
                    "qa": str(qa_out) if qa_out else None,
                    "qa_report": str(qa_report_out) if qa_report_out else None,
                    "run_report": str(run_report_out) if run_report_out else None,
                    "run_manifest": str(run_manifest_out) if run_manifest_out else None,
                    "demo": str(rendered) if rendered else None,
                    "narrative_demo": (
                        str(rendered_videos["narrative"])
                        if rendered_videos.get("narrative")
                        else None
                    ),
                    "analysis_demo": (
                        str(rendered_videos["analysis"])
                        if rendered_videos.get("analysis")
                        else None
                    ),
                },
                "metrics": summary,
                "field_analysis_summary": (
                    field_analysis_result.get("summary") if field_analysis_result else None
                ),
                "qa_status": qa_report.get("status") if qa_report else None,
                "events": len(events),
                "event_summary": event_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_pipeline_qa(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    metrics_out: Path,
    events_out: Path,
    field_analysis_out: Path | None,
) -> tuple[Path | None, Path | None, dict | None]:
    if not args.qa:
        return None, None, None
    qa_out = Path(args.qa_out or results_dir / "qa" / f"{stem}-{args.suffix}-qa.json")
    qa_report_out = Path(
        args.qa_report_out or results_dir / "qa" / f"{stem}-{args.suffix}-qa.md"
    )
    report = evaluate_run_quality(
        metrics_path=metrics_out,
        events_path=events_out,
        field_analysis_path=field_analysis_out,
    )
    write_quality_json(qa_out, report)
    write_quality_markdown(qa_report_out, report)
    return qa_out, qa_report_out, report


def _write_pipeline_run_report(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    metrics_out: Path,
    events_out: Path,
    field_analysis_out: Path | None,
    qa_out: Path | None,
    rendered: Path | None,
    field_map_out: Path | None,
) -> Path:
    report_out = Path(
        args.run_report_out or results_dir / "reports" / f"{stem}-{args.suffix}-report.md"
    )
    return write_run_report(
        report_out,
        title=f"{stem} {args.suffix}",
        metrics_path=metrics_out,
        events_path=events_out,
        field_analysis_path=field_analysis_out,
        qa_path=qa_out,
        demo_path=rendered,
        field_map_path=field_map_out,
    )


def _render_pipeline_videos(
    args: argparse.Namespace,
    *,
    video_path: str,
    tracks_out: Path,
    events_out: Path,
    results_dir: Path,
    stem: str,
    duration_seconds: float | None,
) -> dict[str, Path]:
    if not args.render:
        return {}
    render_seconds = args.max_seconds
    if render_seconds is None:
        render_seconds = duration_seconds

    rendered: dict[str, Path] = {}
    styles = []
    if getattr(args, "render_narrative", True):
        styles.append("narrative")
    if getattr(args, "render_analysis", True):
        styles.append("analysis")
    for style in styles:
        out_path = results_dir / "videos" / f"{stem}-{args.suffix}-{style}-demo.mp4"
        rendered[style] = render_demo_video(
            video_path,
            tracks_out,
            out_path,
            events_path=events_out,
            max_seconds=render_seconds,
            trail_length=args.trail_length,
            style=style,
        )
    return rendered


def _write_pipeline_manifest(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    stem: str,
    artifacts: dict,
    metrics_summary: dict,
    event_summary: dict,
    field_analysis_summary: dict | None,
    qa_status: str | None,
) -> Path:
    manifest_out = Path(
        args.run_manifest_out
        or results_dir / "reports" / f"{stem}-{args.suffix}-manifest.json"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "command_argv": list(sys.argv),
        "command_name": getattr(args, "command", None),
        "video": getattr(args, "video", None),
        "config": getattr(args, "config", None),
        "results_dir": str(results_dir),
        "suffix": getattr(args, "suffix", None),
        "git": _git_snapshot(Path(__file__).resolve().parents[2]),
        "source_fingerprint": _source_fingerprint(Path(__file__).resolve().parents[2]),
        "runtime": _runtime_snapshot(),
        "args": _jsonable(vars(args)),
        "artifacts": {
            key: str(value) if value is not None else None for key, value in sorted(artifacts.items())
        },
        "metrics": {
            "frames_observed": metrics_summary.get("frames_observed"),
            "detections": metrics_summary.get("detections"),
            "tracks": metrics_summary.get("tracks"),
            "possession": metrics_summary.get("possession", {}),
        },
        "event_summary": event_summary,
        "field_analysis_summary": field_analysis_summary,
        "qa_status": qa_status,
    }
    write_json(manifest_out, manifest)
    return manifest_out


def _runtime_snapshot() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _source_fingerprint(repo_root: Path) -> dict:
    patterns = [
        "src/samba_futbot/*.py",
        "config/*.yml",
        "config/*.yaml",
        "pyproject.toml",
        "requirements*.txt",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in repo_root.glob(pattern) if path.is_file())

    digest = hashlib.sha256()
    hashed_files = []
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        hashed_files.append(relative)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "files_hashed": len(hashed_files),
        "paths": hashed_files,
    }


def _git_snapshot(repo_root: Path) -> dict:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()

    try:
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        commit = run_git("rev-parse", "HEAD")
        status = run_git("status", "--short")
    except Exception as exc:
        return {
            "available": False,
            "repo_root": str(repo_root),
            "error": f"{type(exc).__name__}: {exc}",
        }

    changed_files = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "dirty": bool(changed_files),
        "changed_files": len(changed_files),
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


def _assign_teams_for_process(video_path: str, detections: list, config: dict) -> list:
    team_config = config.get("team_detection", {})
    if not bool(team_config.get("enabled", True)):
        return detections
    palette = _team_palette(team_config.get("palette", {}))
    max_distance = float(team_config.get("max_color_distance", 170.0))
    return assign_robot_teams_from_video(
        video_path,
        detections,
        palette=palette,
        max_color_distance=max_distance,
        min_saturation=int(team_config.get("min_saturation", 45)),
        min_value=int(team_config.get("min_value", 40)),
        min_pixels=int(team_config.get("min_pixels", 8)),
    )


def _detect_color_goals_for_process(
    args: argparse.Namespace,
    config: dict,
    *,
    end_frame: int,
    out: Path,
    seed_detections: list | None = None,
) -> list | None:
    if not getattr(args, "goals", False):
        return None
    goal_config = config.get("goal_detection", {})
    enabled = (
        bool(goal_config.get("color_enabled", True))
        if args.color_goals is None
        else bool(args.color_goals)
    )
    if not enabled:
        return None
    return detect_colored_goals(
        args.video,
        out,
        max_frames=end_frame,
        profiles=_goal_color_profiles(goal_config.get("profiles", {})) or None,
        seed_detections=seed_detections,
        adaptive_color=bool(goal_config.get("adaptive_color", True)),
        broad_profiles=_goal_color_profiles(goal_config.get("broad_profiles", {})) or None,
        adaptive_hsv_margin=_parse_hsv_bound(
            None,
            fallback=goal_config.get("adaptive_hsv_margin", [12, 45, 45]),
        ),
        adaptive_min_pixels=int(goal_config.get("adaptive_min_pixels", 120)),
        spatial_gate_from_seeds=bool(goal_config.get("spatial_gate_from_seeds", True)),
        seed_spatial_margin_px=float(goal_config.get("seed_spatial_margin_px", 90.0)),
        require_seed_for_color=bool(goal_config.get("require_seed_for_color", True)),
        require_field_overlap=bool(goal_config.get("require_field_overlap", True)),
        field_margin_px=float(goal_config.get("field_margin_px", 18.0)),
        min_area=float(goal_config.get("min_area", 180.0)),
        max_area=float(goal_config.get("max_area", 80_000.0)),
        min_extent=float(goal_config.get("min_extent", 0.18)),
        max_per_frame_per_class=int(goal_config.get("max_per_frame_per_class", 1)),
    )


def _enforce_goal_constraints_for_process(detections: list, config: dict) -> list:
    goal_config = config.get("goal_detection", {})
    return enforce_goal_frame_constraints(
        detections,
        field_detections=detections,
        max_per_frame_per_class=int(goal_config.get("max_per_frame_per_class", 1)),
        require_field_overlap=bool(goal_config.get("require_field_overlap", True)),
        infer_missing_opposite=bool(goal_config.get("infer_missing_opposite", False)),
        inferred_goal_score=float(goal_config.get("inferred_goal_score", 0.28)),
        field_margin_px=float(goal_config.get("field_margin_px", 18.0)),
    )


def _goal_color_profiles(raw_profiles: dict) -> dict[str, dict[str, tuple[int, int, int]]]:
    profiles: dict[str, dict[str, tuple[int, int, int]]] = {}
    for class_name, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            continue
        profiles[str(class_name)] = {
            "hsv_lower": _parse_hsv_bound(None, fallback=profile.get("hsv_lower", [])),
            "hsv_upper": _parse_hsv_bound(None, fallback=profile.get("hsv_upper", [])),
        }
    return profiles


def _team_palette(raw_palette: dict) -> dict[str, tuple[int, int, int]]:
    palette: dict[str, tuple[int, int, int]] = {}
    for team, value in raw_palette.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        palette[str(team)] = (int(value[0]), int(value[1]), int(value[2]))
    return palette


def _context_classes(args: argparse.Namespace) -> str:
    classes = ["field", "robots"]
    if getattr(args, "goals", False):
        classes.extend(["goal_blue", "goal_yellow"])
    return ",".join(classes)


def _run_sweep_for_process(
    args: argparse.Namespace,
    *,
    classes: str,
    prompt_frames: list[int],
    window_size: int,
    end_frame: int,
    out: Path,
    threshold: float,
    dedupe_iou: float,
) -> list:
    sweep_args = argparse.Namespace(
        config=args.config,
        video=args.video,
        out=str(out),
        prompt_frames=",".join(str(frame) for frame in prompt_frames),
        window_size=window_size,
        end_frame=end_frame,
        classes=classes,
        threshold=threshold,
        dedupe_iou=dedupe_iou,
        backend=None,
        model_id=None,
        use_fa3=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        clip_windows=args.clip_windows,
    )
    cmd_run_sam3_sweep(sweep_args)
    return read_detections(out / "detections.jsonl")


def _frame_anchors(end_frame: int, *, start: int, step: int) -> list[int]:
    if end_frame <= 0:
        return []
    if step <= 0:
        raise ValueError("Anchor step must be positive.")
    anchors = list(range(start, end_frame, step))
    if not anchors:
        anchors = [0]
    return anchors


def _filtered_prompts(prompts: dict, classes_csv: str | None) -> dict:
    if not classes_csv:
        return prompts
    classes = {part.strip() for part in classes_csv.split(",") if part.strip()}
    return {class_name: value for class_name, value in prompts.items() if class_name in classes}


def _resolve_ball_color_profile(
    color_config: dict,
    *,
    profile: str | None,
    hsv_lower: str | None,
    hsv_upper: str | None,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    resolved_profile = profile or str(color_config.get("default_profile", "orange"))
    builtin_profiles = {
        "orange": {"hsv_lower": [0, 90, 90], "hsv_upper": [25, 255, 255]},
        "white": {"hsv_lower": [0, 0, 160], "hsv_upper": [180, 80, 255]},
        "yellow": {"hsv_lower": [20, 80, 90], "hsv_upper": [38, 255, 255]},
    }
    profiles = color_config.get("profiles", {})
    profile_config = profiles.get(resolved_profile, {}) if isinstance(profiles, dict) else {}
    if not profile_config:
        profile_config = builtin_profiles.get(resolved_profile, builtin_profiles["orange"])
    lower = _parse_hsv_bound(
        hsv_lower,
        fallback=profile_config.get("hsv_lower", [0, 90, 90]),
    )
    upper = _parse_hsv_bound(
        hsv_upper,
        fallback=profile_config.get("hsv_upper", [25, 255, 255]),
    )
    return resolved_profile, lower, upper


def _parse_hsv_bound(value: str | None, *, fallback: object) -> tuple[int, int, int]:
    if value:
        items = parse_int_list(value)
    elif isinstance(fallback, (list, tuple)):
        items = [int(item) for item in fallback]
    else:
        raise ValueError("HSV bounds must be a 3-item list.")
    if len(items) != 3:
        raise ValueError("HSV bounds must have exactly 3 values: H,S,V.")
    return (items[0], items[1], items[2])


def cmd_field_analysis(args: argparse.Namespace) -> None:
    detections = read_detections(args.tracks)
    if args.video:
        detections = _assign_teams_for_process(args.video, detections, load_config(args.config))
    analysis = analyze_field_tracks(
        detections,
        load_field_calibration(args.calibration),
        fps=args.fps,
        possession_radius_px=args.possession_radius_px,
        field_margin_px=args.in_play_field_margin_px,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        robot_anchor=args.robot_anchor,
    )
    write_json(args.out, analysis)
    if args.csv_out:
        write_field_trajectory_csv(args.csv_out, analysis)
    if args.robot_csv_out:
        write_field_robot_csv(args.robot_csv_out, analysis)
    if args.zone_control_csv_out:
        write_field_zone_control_csv(args.zone_control_csv_out, analysis)
    if args.map_out:
        render_field_map(analysis, args.map_out)
    print(
        json.dumps(
            {
                "out": args.out,
                "csv_out": args.csv_out,
                "robot_csv_out": args.robot_csv_out,
                "zone_control_csv_out": args.zone_control_csv_out,
                "map_out": args.map_out,
                "summary": analysis["summary"],
                "robot_summary": analysis["robot_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_render_field_map(args: argparse.Namespace) -> None:
    out = render_field_map(read_json(args.analysis), args.out, width=args.width)
    print(json.dumps({"map": str(out)}, indent=2))


def cmd_render_calibration_frame(args: argparse.Namespace) -> None:
    calibration = load_field_calibration(args.calibration) if args.calibration else None
    out = render_calibration_frame(
        args.video,
        args.out,
        frame_index=args.frame_index,
        calibration=calibration,
    )
    print(json.dumps({"calibration_frame": str(out)}, indent=2))


def cmd_calibration_check(args: argparse.Namespace) -> None:
    frame_width = args.frame_width
    frame_height = args.frame_height
    if args.video and (frame_width is None or frame_height is None):
        info = video_info(args.video)
        frame_width = int(info.get("width") or 0) or frame_width
        frame_height = int(info.get("height") or 0) or frame_height
    report = calibration_quality_report(
        load_field_calibration(args.calibration),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if args.out:
        write_calibration_quality(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_summarize_run(args: argparse.Namespace) -> None:
    out = write_run_report(
        args.out,
        title=args.title,
        metrics_path=args.metrics,
        events_path=args.events,
        field_analysis_path=args.field_analysis,
        qa_path=args.qa,
        demo_path=args.demo,
        field_map_path=args.field_map,
    )
    print(json.dumps({"report": str(out)}, indent=2))


def cmd_qa_run(args: argparse.Namespace) -> None:
    thresholds = _qa_thresholds(args)
    report = evaluate_run_quality(
        metrics_path=args.metrics,
        events_path=args.events,
        field_analysis_path=args.field_analysis,
        thresholds=thresholds,
    )
    out = write_quality_json(args.out, report)
    report_out = None
    if args.report_out:
        report_out = write_quality_markdown(args.report_out, report)
    print(
        json.dumps(
            {
                "qa": str(out),
                "report": str(report_out) if report_out else None,
                "status": report["status"],
                "quality_score": report["quality_score"],
                "issues": len(report["issues"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_qa_index(args: argparse.Namespace) -> None:
    reports = collect_quality_reports(args.root, pattern=args.pattern)
    out = write_quality_index_json(args.out, reports)
    report_out = None
    if args.report_out:
        report_out = write_quality_index_markdown(args.report_out, reports)
    print(
        json.dumps(
            {
                "qa_index": str(out),
                "report": str(report_out) if report_out else None,
                "runs": len(reports),
                "best": reports[0] if reports else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _qa_thresholds(args: argparse.Namespace) -> dict[str, float]:
    mapping = {
        "min_ball_coverage": args.min_ball_coverage,
        "fail_ball_coverage": args.fail_ball_coverage,
        "max_ball_jump_px_frame": args.max_ball_jump_px_frame,
        "fail_ball_jump_px_frame": args.fail_ball_jump_px_frame,
        "max_out_of_bounds_ratio": args.max_out_of_bounds_ratio,
        "fail_out_of_bounds_ratio": args.fail_out_of_bounds_ratio,
        "max_unknown_team_ratio": args.max_unknown_team_ratio,
        "fail_unknown_team_ratio": args.fail_unknown_team_ratio,
    }
    return {key: value for key, value in mapping.items() if value is not None}


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
        field_margin_px=args.in_play_field_margin_px,
    )
    write_events(args.out, events)
    summary = summarize_events(events)
    if args.summary_out:
        write_json(args.summary_out, summary)
    print(
        json.dumps(
            {
                "events_out": args.out,
                "summary_out": args.summary_out,
                "events": len(events),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_event_summary(args: argparse.Namespace) -> None:
    events = read_json(args.events)
    if not isinstance(events, list):
        raise ValueError(f"Expected JSON array: {args.events}")
    summary = summarize_events(events)
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_metrics(args: argparse.Namespace) -> None:
    summary = summarize_tracks(
        read_detections(args.tracks),
        fps=args.fps,
        possession_radius_px=args.possession_radius_px,
        field_margin_px=args.in_play_field_margin_px,
    )
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_render_demo(args: argparse.Namespace) -> None:
    out = render_demo_video(
        args.video,
        args.tracks,
        args.out,
        events_path=args.events,
        max_seconds=args.max_seconds,
        trail_length=args.trail_length,
        style=args.style,
    )
    print(json.dumps({"video": str(out), "style": args.style}, indent=2))


def cmd_video_info(args: argparse.Namespace) -> None:
    print(json.dumps(video_info(args.video), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
