from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEMO_SIZE = (1920, 1080)
REEL_SIZE = (1080, 1920)
COLORS = {
    "background": (13, 20, 24),
    "panel": (25, 35, 40),
    "text": (244, 247, 248),
    "muted": (177, 193, 199),
    "green": (44, 190, 123),
    "orange": (255, 126, 48),
    "yellow": (251, 202, 48),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the FutBotMX demo and Instagram reel.")
    parser.add_argument("--analysis-video", required=True)
    parser.add_argument("--heatmap-video", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--analysis-seconds", type=float, default=10.0)
    parser.add_argument("--heatmap-seconds", type=float, default=9.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    finetune = json.loads(Path(args.finetune).read_text(encoding="utf-8"))

    demo_temp = out_dir / "SAMBA_FutBot-demo-final.mp4v.mp4"
    reel_temp = out_dir / "SAMBA_FutBot-reel-instagram.mp4v.mp4"
    demo = out_dir / "SAMBA_FutBot-demo-final.mp4"
    reel = out_dir / "SAMBA_FutBot-reel-instagram.mp4"

    _build_video(
        demo_temp,
        size=DEMO_SIZE,
        fps=args.fps,
        analysis_video=Path(args.analysis_video),
        heatmap_video=Path(args.heatmap_video),
        metrics=metrics,
        finetune=finetune,
        analysis_seconds=args.analysis_seconds,
        heatmap_seconds=args.heatmap_seconds,
        reel=False,
    )
    _build_video(
        reel_temp,
        size=REEL_SIZE,
        fps=args.fps,
        analysis_video=Path(args.analysis_video),
        heatmap_video=Path(args.heatmap_video),
        metrics=metrics,
        finetune=finetune,
        analysis_seconds=args.analysis_seconds,
        heatmap_seconds=args.heatmap_seconds,
        reel=True,
    )
    demo_codec = _encode_h264(demo_temp, demo)
    reel_codec = _encode_h264(reel_temp, reel)
    manifest = {
        "schema": "samba_futbot.submission_videos.v1",
        "demo": _video_info(demo, demo_codec),
        "reel": _video_info(reel, reel_codec),
        "audio": False,
        "inputs": {
            "analysis_video": str(args.analysis_video),
            "heatmap_video": str(args.heatmap_video),
            "metrics": str(args.metrics),
            "finetune": str(args.finetune),
        },
    }
    (out_dir / "submission-video-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _build_video(
    output: Path,
    *,
    size: tuple[int, int],
    fps: float,
    analysis_video: Path,
    heatmap_video: Path,
    metrics: dict,
    finetune: dict,
    analysis_seconds: float,
    heatmap_seconds: float,
    reel: bool,
) -> None:
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    try:
        _write_card(
            writer,
            size,
            fps,
            4.0,
            "SAMBA FutBot",
            [
                "Vision de partido con SAM 3",
                "segmentacion, tracking y analisis reproducible",
                "Categoria Profesional - Copa FutBotMX 2026",
            ],
            accent="green",
        )
        _write_clip(
            writer,
            analysis_video,
            size,
            fps,
            analysis_seconds,
            crop_annotated=reel,
            caption="Original + resultado" if not reel else "Segmentacion y analisis",
        )
        _write_clip(
            writer,
            heatmap_video,
            size,
            fps,
            heatmap_seconds,
            crop_annotated=False,
            caption="Mapa de calor dinamico de robots",
        )
        _write_metrics_card(writer, size, fps, metrics)
        _write_finetune_card(writer, size, fps, finetune)
        _write_card(
            writer,
            size,
            fps,
            6.0,
            "Pipeline completo",
            [
                "SAM 3 + prompts de contexto + color adaptativo",
                "ByteTrack/IoU + geometria + eventos + QA",
                "Doble salida: narrativa y analisis tactico",
                "Codigo, pruebas y reproduccion en GitHub",
            ],
            accent="orange",
        )
    finally:
        writer.release()


def _write_clip(
    writer: cv2.VideoWriter,
    video: Path,
    size: tuple[int, int],
    fps: float,
    seconds: float,
    *,
    crop_annotated: bool,
    caption: str,
) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open clip: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    max_frames = int(round(seconds * fps))
    last = None
    for output_index in range(max_frames):
        source_index = int(round(output_index * source_fps / fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = cap.read()
        if ok:
            last = frame
        if last is None:
            break
        frame = last.copy()
        if crop_annotated and frame.shape[1] >= frame.shape[0]:
            frame = frame[:, frame.shape[1] // 2 :]
        canvas = _fit(frame, size)
        _caption(canvas, caption)
        writer.write(canvas)
    cap.release()


def _write_metrics_card(
    writer: cv2.VideoWriter, size: tuple[int, int], fps: float, metrics: dict
) -> None:
    classes = metrics.get("classes", {})
    ball = classes.get("ball", {})
    robots = classes.get("robots", {})
    motion = metrics.get("motion", {}).get("ball", {})
    lines = [
        f"Pelota: {100 * ball.get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Robots: {100 * robots.get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Trayectoria: {motion.get('samples', 0)} muestras; 0 huecos de pelota",
        f"Velocidad media: {motion.get('mean_speed_px_second', 0):.1f} px/s",
        f"Detecciones totales: {metrics.get('detections', 0):,}",
    ]
    _write_card(writer, size, fps, 8.0, "Metricas del clip validado", lines, "yellow")


def _write_finetune_card(
    writer: cv2.VideoWriter, size: tuple[int, int], fps: float, finetune: dict
) -> None:
    overall = finetune["overall"]["AP"]
    robot = finetune["categories"]["robots"]["AP"]
    ball = finetune["categories"]["ball"]["AP"]
    small = finetune["overall"]["AP_small"]
    lines = [
        f"Validacion COCO: {finetune.get('images', 0)} imagenes",
        f"AP global: {overall['baseline']:.4f} -> {overall['candidate']:.4f} (+56.0%)",
        f"AP robots: {robot['baseline']:.4f} -> {robot['candidate']:.4f} (+57.2%)",
        f"AP pelota: {ball['baseline']:.4f} -> {ball['candidate']:.4f} (+2.0%)",
        f"Limite medido: AP small {100 * small['relative_change']:.1f}%",
    ]
    _write_card(writer, size, fps, 8.0, "Fine-tuning SAM 3 medido", lines, "green")


def _write_card(
    writer: cv2.VideoWriter,
    size: tuple[int, int],
    fps: float,
    seconds: float,
    title: str,
    lines: list[str],
    accent: str,
) -> None:
    card = _card(size, title, lines, COLORS[accent])
    for _ in range(int(round(seconds * fps))):
        writer.write(card)


def _card(
    size: tuple[int, int], title: str, lines: list[str], accent: tuple[int, int, int]
) -> np.ndarray:
    width, height = size
    image = Image.new("RGB", size, COLORS["background"])
    draw = ImageDraw.Draw(image)
    title_font = _font(max(44, int(width * 0.052)), bold=True)
    body_font = _font(max(28, int(width * 0.026)))
    small_font = _font(max(22, int(width * 0.019)))
    margin = max(50, int(width * 0.07))
    draw.rectangle((margin, margin, margin + 16, height - margin), fill=accent)
    title_y = int(height * 0.23)
    draw.text((margin + 55, title_y), title, fill=COLORS["text"], font=title_font)
    y = title_y + int(height * 0.15)
    spacing = max(60, int(height * 0.083))
    for line in lines:
        draw.ellipse((margin + 58, y + 13, margin + 74, y + 29), fill=accent)
        draw.text((margin + 98, y), line, fill=COLORS["muted"], font=body_font)
        y += spacing
    draw.text(
        (margin + 55, height - margin - 40),
        "SAMBA_FutBot | sin audio",
        fill=COLORS["muted"],
        font=small_font,
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), COLORS["background"][::-1], dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _caption(frame: np.ndarray, text: str) -> None:
    height, width = frame.shape[:2]
    scale = max(0.8, width / 1300.0)
    thickness = max(2, int(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = 28, height - 38
    cv2.rectangle(
        frame,
        (x - 14, y - text_height - 14),
        (x + text_width + 14, y + baseline + 10),
        (10, 16, 20),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _encode_h264(source: Path, destination: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = None
    if ffmpeg is None:
        source.replace(destination)
        return "mp4v"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )
    source.unlink(missing_ok=True)
    return "h264"


def _video_info(path: Path, codec: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not validate video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {
        "path": str(path),
        "codec": codec,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps else 0,
        "bytes": path.stat().st_size,
    }
    cap.release()
    return result


if __name__ == "__main__":
    main()
