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
FPS = 30.0
BG = (12, 18, 22)
TEXT = (247, 249, 250)
MUTED = (184, 197, 202)
GREEN = (38, 190, 119)
ORANGE = (255, 126, 46)
YELLOW = (250, 204, 52)
RED = (228, 67, 67)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the FutBotMX demo and reel.")
    parser.add_argument("--narrative-video", required=True)
    parser.add_argument("--analysis-video", required=True)
    parser.add_argument("--normal-video", required=True)
    parser.add_argument("--alternate-video", required=True)
    parser.add_argument("--heatmap-video", required=True)
    parser.add_argument("--heatmap-image", required=True)
    parser.add_argument("--field-map", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--field-analysis", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: Path(getattr(args, name)) for name in (
        "narrative_video", "analysis_video", "normal_video", "alternate_video",
        "heatmap_video", "heatmap_image", "field_map", "metrics",
        "field_analysis", "finetune",
    )}
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    metrics = _read_json(inputs["metrics"])
    field = _read_json(inputs["field_analysis"])
    finetune = _read_json(inputs["finetune"])

    products = []
    for reel, filename, size in (
        (False, "SAMBA_FutBot-demo-final.mp4", DEMO_SIZE),
        (True, "SAMBA_FutBot-reel-instagram.mp4", REEL_SIZE),
    ):
        temp = out_dir / f".{filename}.mp4v.mp4"
        output = out_dir / filename
        _build(temp, size, args.fps, inputs, metrics, field, finetune, reel=reel)
        codec = _encode_h264(temp, output)
        products.append(_video_info(output, codec, kind="reel" if reel else "demo"))

    manifest = {
        "schema": "samba_futbot.submission_videos.v2",
        "team": "Pumas",
        "institution": "Universidad Nacional Autónoma de México (UNAM)",
        "members": ["Germán Alday Salazar", "Raúl García Lemus", "Darien Piña Sánchez"],
        "audio": False,
        "products": products,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "evidence_scope": {
            "heatmap": "full source match: 12:56, 23,278 frames",
            "metric_units": "homography-calibrated field: 2.43 x 1.82 m",
            "events": "candidates requiring review unless explicitly confirmed",
        },
    }
    (out_dir / "submission-video-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _build(
    output: Path,
    size: tuple[int, int],
    fps: float,
    inputs: dict[str, Path],
    metrics: dict,
    field: dict,
    finetune: dict,
    *,
    reel: bool,
) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    try:
        _animated_card(writer, size, fps, 4.0, "PUMAS", [
            "SAMBA FutBot | Categoria Profesional",
            "Universidad Nacional Autónoma de México - UNAM",
            "Germán Alday | Raúl García | Darien Piña",
        ], GREEN)
        _clip(writer, inputs["narrative_video"], size, fps, 10 if reel else 13,
              "MODULO NARRATIVO", "Estado de juego, posesion y eventos candidatos",
              crop_right=reel, badge="BALON EN JUEGO | POSESION: CANDIDATA")
        _clip(writer, inputs["analysis_video"], size, fps, 7 if reel else 9,
              "REVISION DE EVENTO", "GOL CANDIDATO 60% | marcador provisional 1*-0",
              start_seconds=2.2, end_seconds=4.3, hold_at_end=1.8, crop_right=reel,
              badge="MARCADOR CANDIDATO | PUMAS 1* - 0 RIVAL")
        _clip(writer, inputs["normal_video"], size, fps, 7 if reel else 9,
              "VISTA DE CAMPO", "Original + segmentacion SAM 3 + tracking",
              crop_right=reel, badge="SAM 3 | MASCARAS + ID + CONFIANZA")
        if not reel:
            _clip(writer, inputs["alternate_video"], size, fps, 7,
                  "SEGUNDA VISTA", "Generalizacion entre angulos de camara")
        _clip(writer, inputs["analysis_video"], size, fps, 8 if reel else 12,
              "MODULO DE ANALISIS", "Distancias al balon, velocidad y trayectorias",
              start_seconds=0.4, crop_right=reel, badge="CANCHA 2.43 x 1.82 m")
        _clip(writer, inputs["heatmap_video"], size, fps, 8 if reel else 13,
              "MAPA DE CALOR DINAMICO", "Partido completo: 12:56 | 23,278 cuadros",
              badge="23,274 CUADROS LEGIBLES | 23,784 MUESTRAS QA")
        if not reel:
            _image_scene(writer, inputs["heatmap_image"], size, fps, 7,
                         "ACTIVIDAD ACUMULADA", "23,784 muestras de robots con QA")
            _image_scene(writer, inputs["field_map"], size, fps, 7,
                         "MAPA TACTICO CALIBRADO",
                         "Cancha 2.43 x 1.82 m | equipos por marcador visual")
        _metrics_card(writer, size, fps, 7 if reel else 9, metrics, field)
        _finetune_card(writer, size, fps, 7 if reel else 8, finetune)
        _animated_card(writer, size, fps, 5.0, "PUMAS | UNAM", [
            "SAM 3 + prompts + tracking + geometria",
            "Narrativa y analisis tactico reproducibles",
            "Codigo, pruebas y evidencia en GitHub",
        ], ORANGE)
    finally:
        writer.release()


def _clip(
    writer: cv2.VideoWriter,
    path: Path,
    size: tuple[int, int],
    fps: float,
    seconds: float,
    section: str,
    caption: str,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    hold_at_end: float = 0.0,
    crop_right: bool = False,
    badge: str | None = None,
) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open clip: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    start_frame = int(round(start_seconds * source_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame)
    if end_seconds is not None:
        source_frames = min(
            source_frames,
            max(1, int(round((end_seconds - start_seconds) * source_fps))),
        )
    moving_frames = max(1, int(round((seconds - hold_at_end) * fps)))
    last: np.ndarray | None = None
    current_source_index = -1
    for index in range(int(round(seconds * fps))):
        if index < moving_frames:
            target = int(round(index * (source_frames - 1) / max(1, moving_frames - 1)))
            while current_source_index < target:
                if not cap.grab():
                    break
                current_source_index += 1
            if current_source_index == target:
                ok, frame = cap.retrieve()
                if ok:
                    last = frame
        if last is None:
            continue
        display = last
        if crop_right and last.shape[1] >= last.shape[0]:
            display = last[:, last.shape[1] // 2 :]
        canvas = _fit(display, size)
        _chrome(
            canvas,
            section,
            caption,
            progress=index / max(1, int(seconds * fps) - 1),
            badge=badge,
        )
        writer.write(canvas)
    cap.release()


def _image_scene(
    writer: cv2.VideoWriter,
    path: Path,
    size: tuple[int, int],
    fps: float,
    seconds: float,
    section: str,
    caption: str,
) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    frames = int(round(seconds * fps))
    for index in range(frames):
        zoom = 1.0 + 0.06 * index / max(1, frames - 1)
        h, w = image.shape[:2]
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x, y = (w - crop_w) // 2, (h - crop_h) // 2
        canvas = _fit(image[y:y + crop_h, x:x + crop_w], size)
        _chrome(canvas, section, caption, progress=index / max(1, frames - 1))
        writer.write(canvas)


def _metrics_card(writer, size, fps, seconds, metrics: dict, field: dict) -> None:
    classes = metrics.get("classes", {})
    summary = field.get("summary", {})
    lines = [
        f"Pelota detectada: {100 * classes.get('ball', {}).get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Robots detectados: {100 * classes.get('robots', {}).get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Velocidad calibrada: media {summary.get('mean_speed_m_s', 0):.2f} m/s | max {summary.get('max_speed_m_s', 0):.2f} m/s",
        f"Trayectoria valida: {summary.get('distance_m', 0):.2f} m | {summary.get('path_samples', 0)} muestras",
        "Posesion y goles se publican con confianza y alcance, no como verdad absoluta",
    ]
    _animated_card(writer, size, fps, seconds, "METRICAS Y QA", lines, YELLOW)


def _finetune_card(writer, size, fps, seconds, data: dict) -> None:
    overall = data["overall"]["AP"]
    robot = data["categories"]["robots"]["AP"]
    ball = data["categories"]["ball"]["AP"]
    small = data["overall"]["AP_small"]
    lines = [
        f"Evaluacion COCO: {data.get('images', 0)} imagenes",
        f"AP global: {overall['baseline']:.3f} -> {overall['candidate']:.3f} (+56.0%)",
        f"AP robots: {robot['baseline']:.3f} -> {robot['candidate']:.3f} (+57.2%)",
        f"AP pelota: {ball['baseline']:.3f} -> {ball['candidate']:.3f} (+2.0%)",
        f"Limite declarado: AP-small {100 * small['relative_change']:.1f}%",
    ]
    _animated_card(writer, size, fps, seconds, "FINE-TUNING MEDIDO", lines, GREEN)


def _animated_card(writer, size, fps, seconds, title, lines, accent) -> None:
    frames = int(round(seconds * fps))
    for index in range(frames):
        reveal = min(1.0, index / max(1, fps * 0.55))
        pulse = 0.92 + 0.08 * np.sin(index / fps * np.pi)
        writer.write(_card(size, title, lines, accent, reveal, pulse))


def _card(size, title, lines, accent, reveal: float, pulse: float) -> np.ndarray:
    width, height = size
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    margin = max(44, int(width * 0.065))
    bar = int((width - 2 * margin) * reveal)
    draw.rectangle((margin, int(height * 0.17), margin + bar, int(height * 0.185)), fill=accent)
    title_font = _font(max(46, int(width * (0.075 if height > width else 0.055))), bold=True)
    body_font = _font(max(27, int(width * (0.041 if height > width else 0.025))))
    small_font = _font(max(21, int(width * 0.018)))
    wrapped = _wrap_lines(draw, lines, body_font, width - 2 * margin - 42)
    draw.text((margin, int(height * 0.22)), title, fill=TEXT, font=title_font)
    y = int(height * 0.37)
    available = height - y - margin - 92
    spacing = min(max(64, int(height * 0.092)), max(48, available // max(1, len(wrapped))))
    visible = int(np.ceil(len(wrapped) * reveal))
    for line, starts_item in wrapped[:visible]:
        if starts_item:
            radius = int(8 * pulse)
            draw.ellipse(
                (margin, y + 11, margin + 2 * radius, y + 11 + 2 * radius),
                fill=accent,
            )
        draw.text((margin + 42, y), line, fill=MUTED, font=body_font)
        y += spacing
    draw.text((margin, height - margin - 28), "SAMBA_FutBot | video sin audio", fill=MUTED, font=small_font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _chrome(
    frame: np.ndarray,
    section: str,
    caption: str,
    *,
    progress: float,
    badge: str | None = None,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    top_height = max(150 if badge else 105, int(h * (0.12 if badge else 0.095)))
    cv2.rectangle(overlay, (0, 0), (w, top_height), (9, 14, 18), -1)
    cv2.rectangle(overlay, (0, h - max(120, int(h * 0.12))), (w, h), (9, 14, 18), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    scale = _cv_fit_scale(section, w - 72, max(0.8, min(1.45, w / 1350)))
    cv2.putText(frame, section, (36, 62), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), max(2, int(scale * 2)), cv2.LINE_AA)
    if badge:
        badge_scale = _cv_fit_scale(badge, w - 72, max(0.64, min(1.0, w / 1700)))
        cv2.putText(
            frame,
            badge,
            (36, min(top_height - 24, 118)),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_scale,
            (76, 224, 157),
            max(2, int(badge_scale * 2)),
            cv2.LINE_AA,
        )
    caption_scale = _cv_fit_scale(caption, w - 72, max(0.72, min(1.25, w / 1550)))
    cv2.putText(frame, caption, (36, h - 53), cv2.FONT_HERSHEY_SIMPLEX,
                caption_scale, (239, 244, 246), max(2, int(caption_scale * 2)), cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 8), (int(w * progress), h), (46, 190, 119), -1)


def _wrap_lines(
    draw: ImageDraw.ImageDraw, lines: list[str], font, max_width: int
) -> list[tuple[str, bool]]:
    wrapped: list[tuple[str, bool]] = []
    for line in lines:
        current = ""
        starts_item = True
        for word in line.split():
            candidate = f"{current} {word}".strip()
            if current and draw.textlength(candidate, font=font) > max_width:
                wrapped.append((current, starts_item))
                current = word
                starts_item = False
            else:
                current = candidate
        if current:
            wrapped.append((current, starts_item))
    return wrapped


def _cv_fit_scale(text: str, max_width: int, preferred: float) -> float:
    scale = preferred
    while scale > 0.48:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.04
    return 0.48


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(frame, (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), BG[::-1], dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _font(size: int, *, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    subprocess.run([ffmpeg, "-y", "-i", str(source), "-an", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(destination)], check=True)
    source.unlink(missing_ok=True)
    return "h264"


def _video_info(path: Path, codec: str, *, kind: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not validate video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {"kind": kind, "path": str(path), "codec": codec,
              "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": fps,
              "frames": frames, "duration_seconds": frames / fps if fps else 0,
              "bytes": path.stat().st_size}
    cap.release()
    return result


if __name__ == "__main__":
    main()
