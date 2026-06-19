from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEMO_SIZE = (1920, 1080)
REEL_SIZE = (1080, 1920)
FPS = 30.0
TOP_BAND = 96
BOTTOM_BAND = 84
BG = (12, 18, 22)
TEXT = (247, 249, 250)
MUTED = (184, 197, 202)
GREEN = (38, 190, 119)
ORANGE = (255, 126, 46)
YELLOW = (250, 204, 52)
CYAN = (44, 186, 218)
RED = (228, 67, 67)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FutBotMX submission videos.")
    parser.add_argument("--narrative-video", required=True)
    parser.add_argument("--analysis-video", required=True)
    parser.add_argument("--normal-video", required=True)
    parser.add_argument("--goal-video", required=True)
    parser.add_argument("--heatmap-video", required=True)
    parser.add_argument("--heatmap-image", required=True)
    parser.add_argument("--field-map", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--field-analysis", required=True)
    parser.add_argument("--prediction-analysis", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--event-summary", required=True)
    parser.add_argument("--ball-filter-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "narrative_video",
        "analysis_video",
        "normal_video",
        "goal_video",
        "heatmap_video",
        "heatmap_image",
        "field_map",
        "metrics",
        "field_analysis",
        "prediction_analysis",
        "finetune",
        "event_summary",
        "ball_filter_report",
    )
    inputs = {name: Path(getattr(args, name)) for name in names}
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    evidence = {
        "metrics": _read_json(inputs["metrics"]),
        "field": _read_json(inputs["field_analysis"]),
        "prediction": _read_json(inputs["prediction_analysis"]),
        "finetune": _read_json(inputs["finetune"]),
        "events": _read_json(inputs["event_summary"]),
        "ball_filter": _read_json(inputs["ball_filter_report"]),
    }
    products = []
    for reel, filename, size in (
        (False, "SAMBA_FutBot-demo-final.mp4", DEMO_SIZE),
        (True, "SAMBA_FutBot-reel-instagram.mp4", REEL_SIZE),
    ):
        temp = out_dir / f".{filename}.mp4v.mp4"
        output = out_dir / filename
        _build(temp, size, args.fps, inputs, evidence, reel=reel)
        codec = _encode_h264(temp, output)
        products.append(_video_info(output, codec, kind="reel" if reel else "demo"))

    manifest = {
        "schema": "samba_futbot.submission_videos.v4",
        "competition": "Copa FutBotMX 2026",
        "challenge": "Reto de Visión por Computadora - Categoría Profesional",
        "team": "Pumas",
        "institution": "Universidad Nacional Autónoma de México (UNAM)",
        "members": ["Germán Alday Salazar", "Raúl García Lemus", "Darien Piña Sánchez"],
        "products": products,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "evidence_scope": {
            "heatmap": "full source match: 12:56, 23,274 readable frames",
            "metric_units": "homography-calibrated field: 2.43 x 1.82 m",
            "segmentation_metrics": "COCO mask AP/AR over 128 annotated images",
            "operational_validation": "one ball per frame, field/referee context, robot duplicate suppression",
            "events": "goal visually validated in video-427 by complete goal-line crossing",
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
    evidence: dict,
    *,
    reel: bool,
) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    try:
        _animated_card(
            writer,
            size,
            fps,
            4,
            "PUMAS | SAMBA FUTBOT",
            [
                "Copa FutBotMX 2026 | Visión por Computadora",
                "Categoría Profesional | Universidad Nacional Autónoma de México",
                "Germán Alday | Raúl García | Darien Piña",
            ],
            GREEN,
        )
        _approach_card(writer, size, fps, 5 if reel else 7)
        _clip(
            writer,
            inputs["narrative_video"],
            size,
            fps,
            10 if reel else 12,
            "MODULO NARRATIVO",
            "Dos robots activos y una pelota unica por cuadro",
            crop_right=reel,
            badge="ESTADO: EN JUEGO | EVENTOS CANDIDATOS",
        )
        _pause_explainer(writer, inputs["analysis_video"], size, fps, 5, reel=reel)
        _goal_scene(
            writer,
            inputs["goal_video"],
            size,
            fps,
            10 if reel else 12,
            reel=reel,
        )
        _clip(
            writer,
            inputs["normal_video"],
            size,
            fps,
            6 if reel else 8,
            "SEGUNDA CAMARA",
            "Original + mascaras SAM 3 + IDs temporales",
            crop_right=reel,
            badge="VISTA NORMAL | SIN DETECCIONES DE PELOTA AMBIGUAS",
        )
        _clip(
            writer,
            inputs["analysis_video"],
            size,
            fps,
            7 if reel else 9,
            "MODULO DE ANALISIS",
            "Distancia robot-pelota en m | velocidad de pelota en m/s",
            start_seconds=7.2,
            end_seconds=10.0,
            crop_right=reel,
            badge="HOMOGRAFIA | CANCHA 2.43 x 1.82 m",
        )
        _prediction_scene(
            writer,
            size,
            fps,
            7 if reel else 8,
            evidence["prediction"],
        )
        _clip(
            writer,
            inputs["heatmap_video"],
            size,
            fps,
            7 if reel else 10,
            "MAPA DE CALOR DINAMICO",
            "Partido completo: 12:56 | actividad acumulada progresiva",
            badge="23,274 CUADROS | 23,784 MUESTRAS CON QA",
        )
        if not reel:
            _image_scene(
                writer,
                inputs["heatmap_image"],
                size,
                fps,
                4,
                "ACTIVIDAD ACUMULADA",
                "Heatmap del partido completo, filtrado por geometria y campo",
            )
            _image_scene(
                writer,
                inputs["field_map"],
                size,
                fps,
                4,
                "MAPA TACTICO CALIBRADO",
                "Coordenadas metricas y trayectoria sobre cancha reglamentaria",
            )
        _sports_card(writer, size, fps, 5 if reel else 6, evidence["events"])
        _physical_metrics_card(
            writer,
            size,
            fps,
            5 if reel else 6,
            evidence["metrics"],
            evidence["field"],
        )
        _operational_validation_card(
            writer,
            size,
            fps,
            7 if reel else 9,
            evidence["ball_filter"],
        )
        _quantitative_validation_card(
            writer,
            size,
            fps,
            7 if reel else 8,
            evidence["finetune"],
        )
        _animated_card(
            writer,
            size,
            fps,
            4,
            "PUMAS | UNAM",
            [
                "SAM 3 + contexto + tracking + geometría",
                "Código, pruebas, métricas y evidencia reproducible",
                "Copa FutBotMX 2026 | Categoría Profesional",
            ],
            ORANGE,
        )
    finally:
        writer.release()


def _approach_card(writer, size, fps, seconds) -> None:
    lines = [
        "SAM 3: máscaras por prompts de texto, puntos, cajas y contexto",
        "Color adaptable: recuperación HSV como evidencia complementaria",
        "Pelota: campo/mano + rechazo de robot + trayectoria temporal única",
        "Robots: IoU + contención + distancia normalizada entre cajas",
        "ByteTrack/IoU + homografía + eventos + compuertas de QA",
    ]
    _animated_card(writer, size, fps, seconds, "ENFOQUE HÍBRIDO", lines, CYAN)


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
    total_frames = int(round(seconds * fps))
    for index in range(total_frames):
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
        display = _right_half(last) if crop_right else last
        canvas = _fit_with_bands(display, size)
        _chrome(canvas, section, caption, index / max(1, total_frames - 1), badge=badge)
        writer.write(canvas)
    cap.release()


def _pause_explainer(writer, path, size, fps, seconds, *, reel: bool) -> None:
    frame = _read_frame(path, 8.8)
    display = _right_half(frame) if reel else frame
    total = int(round(seconds * fps))
    for index in range(total):
        canvas = _fit_with_bands(display, size)
        _chrome(
            canvas,
            "PAUSA EXPLICATIVA",
            "La evidencia visual se separa por objeto, identidad y contexto",
            index / max(1, total - 1),
            badge="MASCARA + ID + DISTANCIA METRICA + PELOTA UNICA",
        )
        targets = (
            [
                (0.63, 0.70, "Robot: mascara e ID"),
                (0.72, 0.67, "Pelota unica"),
                (0.68, 0.69, "Distancia en m"),
            ]
            if not reel
            else [
                (0.48, 0.72, "Robot: mascara e ID"),
                (0.82, 0.68, "Pelota unica"),
                (0.65, 0.70, "Distancia en m"),
            ]
        )
        for callout_index, (tx, ty, label) in enumerate(targets):
            _draw_callout(
                canvas,
                label,
                (int(size[0] * tx), int(size[1] * ty)),
                (32, TOP_BAND + 34 + callout_index * 62),
                CYAN,
            )
        writer.write(canvas)


def _goal_scene(writer, path, size, fps, seconds, *, reel: bool) -> None:
    total = int(round(seconds * fps))
    moving = max(1, total - int(round(2.5 * fps)))
    start, event_time, end = 9.5, 11.75, 13.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
    last = None
    for index in range(total):
        if index < moving:
            source_time = start + (end - start) * index / max(1, moving - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(source_time * source_fps)))
            ok, frame = cap.read()
            if ok:
                last = frame
        if last is None:
            continue
        display = _center_vertical_crop(last) if reel else last
        canvas = _fit_with_bands(display, size)
        after_goal = index >= int((event_time - start) / (end - start) * moving)
        badge = (
            "MARCADOR | AMARILLO 1 - 0 AZUL"
            if after_goal
            else "MARCADOR | AMARILLO 0 - 0 AZUL"
        )
        caption = (
            "Gol validado en video: cruce completo de la linea de meta"
            if after_goal
            else "Disparo detectado por el pipeline hacia la porteria azul"
        )
        _chrome(canvas, "SECUENCIA DE GOL", caption, index / max(1, total - 1), badge=badge)
        if after_goal and index >= moving:
            target = (int(size[0] * (0.61 if not reel else 0.84)), int(size[1] * 0.64))
            _draw_callout(
                canvas,
                "Balon tras el plano de gol",
                target,
                (32, TOP_BAND + 42),
                YELLOW,
            )
        writer.write(canvas)
    cap.release()


def _prediction_scene(writer, size, fps, seconds, analysis: dict) -> None:
    path = analysis.get("path", [])
    if len(path) < 12:
        return
    total = int(round(seconds * fps))
    field = analysis.get("calibration", {}).get("field", {})
    field_length = float(field.get("length_m", 2.43))
    field_width = float(field.get("width_m", 1.82))
    speed_candidates = []
    for index in range(5, len(path)):
        dx = float(path[index]["field_x_m"]) - float(path[index - 5]["field_x_m"])
        dy = float(path[index]["field_y_m"]) - float(path[index - 5]["field_y_m"])
        speed_candidates.append((math.hypot(dx, dy), index))
    peak_index = max(speed_candidates, default=(0.0, len(path) // 2))[1]
    start_index = max(5, peak_index - 30)
    end_index = min(len(path) - 1, max(start_index + 1, peak_index + 30))
    for output_index in range(total):
        current_index = start_index + int(
            (end_index - start_index) * output_index / max(1, total - 1)
        )
        current = path[current_index]
        previous = path[max(0, current_index - 5)]
        dt = max(1, int(current["frame_index"]) - int(previous["frame_index"])) / 30.0
        vx = (float(current["field_x_m"]) - float(previous["field_x_m"])) / dt
        vy = (float(current["field_y_m"]) - float(previous["field_y_m"])) / dt
        canvas = _field_canvas(size)
        _chrome(
            canvas,
            "PREDICCION DE MOVIMIENTO",
            "Modelo cinematico: p(t+dt) = p(t) + v*dt",
            output_index / max(1, total - 1),
            badge=f"VELOCIDAD ESTIMADA {math.hypot(vx, vy):.2f} m/s | HORIZONTE 1.5 s",
        )
        trail = path[max(0, current_index - 35) : current_index + 1]
        trail_points = [
            _field_to_canvas(float(item["field_x_m"]), float(item["field_y_m"]), size, field_length, field_width)
            for item in trail
        ]
        for first, second in zip(trail_points, trail_points[1:]):
            cv2.line(canvas, first, second, (255, 155, 60), 3, cv2.LINE_AA)
        current_point = _field_to_canvas(
            float(current["field_x_m"]),
            float(current["field_y_m"]),
            size,
            field_length,
            field_width,
        )
        cv2.circle(canvas, current_point, 12, (30, 120, 255), -1)
        previous_point = current_point
        predictions = []
        for horizon in (0.5, 1.0, 1.5):
            predicted_x = min(field_length, max(0.0, float(current["field_x_m"]) + vx * horizon))
            predicted_y = min(field_width, max(0.0, float(current["field_y_m"]) + vy * horizon))
            point = _field_to_canvas(predicted_x, predicted_y, size, field_length, field_width)
            cv2.line(canvas, previous_point, point, (80, 230, 255), 2, cv2.LINE_AA)
            confidence = math.exp(-0.45 * horizon)
            cv2.circle(canvas, point, 9, (80, 230, 255), 2)
            predictions.append((horizon, confidence))
            previous_point = point
        legend_x = int(size[0] * 0.13)
        legend_y = TOP_BAND + 70
        legend_width = int(size[0] * (0.30 if size[0] > size[1] else 0.72))
        cv2.rectangle(
            canvas,
            (legend_x - 14, legend_y - 34),
            (legend_x + legend_width, legend_y + 92),
            (18, 34, 24),
            -1,
        )
        for prediction_index, (horizon, confidence) in enumerate(predictions):
            cv2.putText(
                canvas,
                f"Horizonte {horizon:.1f}s | confianza heuristica {confidence:.0%}",
                (legend_x, legend_y + prediction_index * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56 if size[0] > 1200 else 0.45,
                (240, 245, 250),
                2,
                cv2.LINE_AA,
            )
        writer.write(canvas)


def _image_scene(writer, path, size, fps, seconds, section, caption) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    total = int(round(seconds * fps))
    for index in range(total):
        zoom = 1.0 + 0.05 * index / max(1, total - 1)
        h, w = image.shape[:2]
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x, y = (w - crop_w) // 2, (h - crop_h) // 2
        canvas = _fit_with_bands(image[y : y + crop_h, x : x + crop_w], size)
        _chrome(canvas, section, caption, index / max(1, total - 1))
        writer.write(canvas)


def _sports_card(writer, size, fps, seconds, summary: dict) -> None:
    counts = summary.get("counts", {})
    lines = [
        "Alcance: segmento superior limpio + secuencia de gol validada",
        f"Pases válidos: {counts.get('pass', 0)} | disparos detectados: {counts.get('shot', 0)}",
        "Gol mostrado: 1 | marcador: amarillo 1 - 0 azul",
        "Validación del gol: cruce completo de línea y continuidad visual",
        f"Colisiones tras suprimir duplicados: {counts.get('collision', 0)}",
    ]
    _animated_card(writer, size, fps, seconds, "ESTADÍSTICAS DE JUEGO", lines, ORANGE)


def _physical_metrics_card(writer, size, fps, seconds, metrics: dict, field: dict) -> None:
    classes = metrics.get("classes", {})
    summary = field.get("summary", {})
    lines = [
        f"Pelota detectada: {100 * classes.get('ball', {}).get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Robots detectados: {100 * classes.get('robots', {}).get('frame_coverage_ratio', 0):.1f}% de cuadros",
        f"Trayectoria calibrada: {summary.get('distance_m', 0):.2f} m | {summary.get('path_samples', 0):,} muestras",
        f"Velocidad: media {summary.get('mean_speed_m_s', 0):.2f} m/s | máxima {summary.get('max_speed_m_s', 0):.2f} m/s",
        "Escala física: homografía sobre cancha oficial 2.43 × 1.82 m",
    ]
    _animated_card(writer, size, fps, seconds, "MÉTRICAS FÍSICAS", lines, YELLOW)


def _operational_validation_card(writer, size, fps, seconds, report: dict) -> None:
    lines = [
        "Pelota unica: maximo una trayectoria por cuadro",
        f"Candidatos revisados: {report.get('input_balls', 0)} | aceptados: {report.get('kept_balls', 0)}",
        f"Falsos puntos sobre robots eliminados: {report.get('removed_by_reason', {}).get('robot_overlap', 0)}",
        "Contexto obligatorio: cancha verde o mano del arbitro",
        "Robots: IoU + contencion + distancia normalizada entre cajas",
        "Gol: cruce de linea, continuidad temporal y porteria observada",
        "Magnitudes fisicas: homografia, metros y metros por segundo",
    ]
    _animated_card(writer, size, fps, seconds, "VALIDACION OPERATIVA", lines, GREEN)


def _quantitative_validation_card(writer, size, fps, seconds, data: dict) -> None:
    overall = data["overall"]
    robots = data["categories"]["robots"]
    ap_baseline = float(overall["AP"]["baseline"])
    ap_candidate = float(overall["AP"]["candidate"])
    ap50_baseline = float(overall["AP50"]["baseline"])
    ap50_candidate = float(overall["AP50"]["candidate"])
    ap_gain = 100.0 * (ap_candidate / ap_baseline - 1.0)
    ap50_gain = 100.0 * (ap50_candidate / ap50_baseline - 1.0)
    lines = [
        f"Evaluacion COCO de mascaras: {data.get('images', 0)} imagenes anotadas",
        f"Recall global AR@100: {100 * float(overall['AR_100']['candidate']):.1f}%",
        f"Robots AP75: {100 * float(robots['AP75']['candidate']):.1f}%",
        f"Mejora de AP con fine-tuning: +{ap_gain:.1f}%",
        f"Mejora de AP50 con fine-tuning: +{ap50_gain:.1f}%",
        "Protocolo: IoU 0.50:0.95; resultados completos en docs/evidence",
    ]
    _animated_card(writer, size, fps, seconds, "VALIDACION CUANTITATIVA", lines, CYAN)


def _animated_card(writer, size, fps, seconds, title, lines, accent) -> None:
    total = int(round(seconds * fps))
    for index in range(total):
        reveal = min(1.0, index / max(1, fps * 0.55))
        pulse = 0.92 + 0.08 * np.sin(index / fps * np.pi)
        writer.write(_card(size, title, lines, accent, reveal, pulse))


def _card(size, title, lines, accent, reveal: float, pulse: float) -> np.ndarray:
    width, height = size
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    margin = max(44, int(width * 0.065))
    bar = int((width - 2 * margin) * reveal)
    draw.rectangle((margin, int(height * 0.15), margin + bar, int(height * 0.165)), fill=accent)
    title_font = _fitting_font(
        draw,
        title,
        max(44, int(width * (0.068 if height > width else 0.050))),
        width - 2 * margin,
        bold=True,
    )
    body_font = _font(max(26, int(width * (0.039 if height > width else 0.023))))
    small_font = _font(max(20, int(width * 0.017)))
    wrapped = _wrap_lines(draw, lines, body_font, width - 2 * margin - 42)
    draw.text((margin, int(height * 0.20)), title, fill=TEXT, font=title_font)
    y = int(height * 0.34)
    available = height - y - margin - 92
    spacing = min(max(58, int(height * 0.082)), max(44, available // max(1, len(wrapped))))
    visible = int(np.ceil(len(wrapped) * reveal))
    for line, starts_item in wrapped[:visible]:
        if starts_item:
            radius = int(7 * pulse)
            draw.ellipse((margin, y + 11, margin + 2 * radius, y + 11 + 2 * radius), fill=accent)
        draw.text((margin + 38, y), line, fill=MUTED, font=body_font)
        y += spacing
    draw.text(
        (margin, height - margin - 26),
        "Copa FutBotMX 2026 | Reto de Visión por Computadora",
        fill=MUTED,
        font=small_font,
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _fit_with_bands(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    content_h = height - TOP_BAND - BOTTOM_BAND
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, content_h / source_h)
    resized = cv2.resize(
        frame,
        (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), BG[::-1], dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = TOP_BAND + (content_h - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _chrome(frame, section, caption, progress, *, badge=None) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, TOP_BAND), (9, 14, 18), -1)
    cv2.rectangle(frame, (0, height - BOTTOM_BAND), (width, height), (9, 14, 18), -1)
    section_scale = _cv_fit_scale(section, width - 72, max(0.72, min(1.2, width / 1500)))
    cv2.putText(
        frame,
        section,
        (32, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        section_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if badge:
        badge_scale = _cv_fit_scale(badge, width - 72, max(0.52, min(0.76, width / 2100)))
        cv2.putText(
            frame,
            badge,
            (32, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_scale,
            (76, 224, 157),
            2,
            cv2.LINE_AA,
        )
    caption_scale = _cv_fit_scale(caption, width - 72, max(0.60, min(0.94, width / 1750)))
    cv2.putText(
        frame,
        caption,
        (32, height - 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        caption_scale,
        (239, 244, 246),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (0, height - 6), (int(width * progress), height), (46, 190, 119), -1)


def _draw_callout(frame, text, target, label_origin, color) -> None:
    scale = _cv_fit_scale(text, max(180, frame.shape[1] // 2), 0.62)
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = label_origin
    cv2.rectangle(
        frame,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + baseline + 8),
        (10, 18, 22),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(frame, (x + text_w + 8, y - text_h // 2), target, color, 3, tipLength=0.08)


def _field_canvas(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), BG[::-1], dtype=np.uint8)
    left, right = int(width * 0.10), int(width * 0.90)
    top, bottom = TOP_BAND + 24, height - BOTTOM_BAND - 24
    cv2.rectangle(canvas, (left, top), (right, bottom), (55, 132, 65), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (235, 240, 235), 3)
    mid = (left + right) // 2
    cv2.line(canvas, (mid, top), (mid, bottom), (235, 240, 235), 2)
    cv2.circle(canvas, (mid, (top + bottom) // 2), max(28, int((bottom - top) * 0.16)), (235, 240, 235), 2)
    return canvas


def _field_to_canvas(x, y, size, field_length, field_width) -> tuple[int, int]:
    width, height = size
    left, right = int(width * 0.10), int(width * 0.90)
    top, bottom = TOP_BAND + 24, height - BOTTOM_BAND - 24
    return (
        int(round(left + x / field_length * (right - left))),
        int(round(top + y / field_width * (bottom - top))),
    )


def _read_frame(path: Path, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {path} at {seconds:.2f}s")
    return frame


def _right_half(frame: np.ndarray) -> np.ndarray:
    return frame[:, frame.shape[1] // 2 :] if frame.shape[1] >= frame.shape[0] else frame


def _center_vertical_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_width = min(width, max(1, int(round(height * 0.68))))
    center_x = int(round(width * 0.50))
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    return frame[:, left : left + crop_width]


def _wrap_lines(draw, lines, font, max_width) -> list[tuple[str, bool]]:
    wrapped = []
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
    while scale > 0.40:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.04
    return 0.40


def _font(size: int, *, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _fitting_font(draw, text: str, preferred: int, max_width: int, *, bold: bool = False):
    size = preferred
    while size > 24:
        font = _font(size, bold=bold)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 2
    return _font(24, bold=bold)


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


def _video_info(path: Path, codec: str, *, kind: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not validate video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {
        "kind": kind,
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
