from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .io_utils import ensure_parent


DEFAULT_FIELD_LENGTH_M = 2.43
DEFAULT_FIELD_WIDTH_M = 1.82
DEFAULT_CENTER_CIRCLE_DIAMETER_M = 0.60
DEFAULT_PENALTY_AREA_DEPTH_M = 0.25
DEFAULT_PENALTY_AREA_WIDTH_M = 0.80
DEFAULT_GOAL_WIDTH_M = 0.60
DEFAULT_GOAL_DEPTH_M = 0.10


def render_field_map(
    analysis: dict,
    out_path: str | Path,
    *,
    width: int = 1200,
    margin: int = 70,
) -> Path:
    calibration = analysis.get("calibration", {})
    field = calibration.get("field", {})
    field_length = float(field.get("length_m", DEFAULT_FIELD_LENGTH_M))
    field_width = float(field.get("width_m", DEFAULT_FIELD_WIDTH_M))
    if field_length <= 0 or field_width <= 0:
        raise ValueError("Field dimensions must be positive.")

    image_width = int(width)
    field_px_width = image_width - margin * 2
    field_px_height = max(1, int(field_px_width * (field_width / field_length)))
    image_height = field_px_height + margin * 2 + 110

    image = Image.new("RGB", (image_width, image_height), (245, 247, 244))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    field_box = (margin, margin, margin + field_px_width, margin + field_px_height)
    _draw_field(draw, field_box, field, font)
    _draw_heatmap(draw, analysis, field_box)
    _draw_trajectory(draw, analysis, field_box, field_length, field_width)
    _draw_summary(draw, analysis, field_box, font)

    output = ensure_parent(out_path)
    image.save(output)
    return output


def _draw_field(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    field: dict,
    font,
) -> None:
    x1, y1, x2, y2 = box
    field_length = float(field.get("length_m", DEFAULT_FIELD_LENGTH_M))
    field_width = float(field.get("width_m", DEFAULT_FIELD_WIDTH_M))
    draw.rounded_rectangle(
        box,
        radius=12,
        fill=(48, 138, 76, 255),
        outline=(245, 245, 245, 255),
        width=4,
    )
    center = _to_canvas((field_length / 2, field_width / 2), box, field_length, field_width)
    draw.line((center[0], y1, center[0], y2), fill=(245, 245, 245, 220), width=3)
    radius = (
        float(field.get("center_circle_diameter_m", DEFAULT_CENTER_CIRCLE_DIAMETER_M))
        / 2
        / field_width
        * (y2 - y1)
    )
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=(245, 245, 245, 200),
        width=3,
    )
    penalty_depth = float(field.get("penalty_area_depth_m", DEFAULT_PENALTY_AREA_DEPTH_M))
    penalty_width = float(field.get("penalty_area_width_m", DEFAULT_PENALTY_AREA_WIDTH_M))
    goal_width = float(field.get("goal_width_m", DEFAULT_GOAL_WIDTH_M))
    goal_depth = float(field.get("goal_depth_m", DEFAULT_GOAL_DEPTH_M))
    penalty_y1 = y1 + (field_width - penalty_width) / 2 / field_width * (y2 - y1)
    penalty_y2 = y1 + (field_width + penalty_width) / 2 / field_width * (y2 - y1)
    left_penalty_x = x1 + penalty_depth / field_length * (x2 - x1)
    right_penalty_x = x2 - penalty_depth / field_length * (x2 - x1)
    draw.rectangle(
        (x1, penalty_y1, left_penalty_x, penalty_y2),
        outline=(245, 245, 245, 180),
        width=2,
    )
    draw.rectangle(
        (right_penalty_x, penalty_y1, x2, penalty_y2),
        outline=(245, 245, 245, 180),
        width=2,
    )

    goal_y1 = y1 + (field_width - goal_width) / 2 / field_width * (y2 - y1)
    goal_y2 = y1 + (field_width + goal_width) / 2 / field_width * (y2 - y1)
    goal_px_depth = max(4.0, goal_depth / field_length * (x2 - x1))
    for gx, fill in ((x1, (255, 218, 65, 230)), (x2, (0, 160, 220, 230))):
        goal_box = (
            gx - goal_px_depth if gx == x1 else gx,
            goal_y1,
            gx if gx == x1 else gx + goal_px_depth,
            goal_y2,
        )
        draw.rectangle(goal_box, fill=fill, outline=(245, 245, 245, 220), width=2)
        draw.line(
            (gx, goal_y1, gx, goal_y2),
            fill=(255, 235, 120, 255),
            width=6,
        )
    draw.text(
        (x1, y1 - 28),
        "Field map: ball trajectory and zone occupancy",
        fill=(35, 45, 40, 255),
        font=font,
    )


def _draw_heatmap(
    draw: ImageDraw.ImageDraw,
    analysis: dict,
    box: tuple[int, int, int, int],
) -> None:
    grid = analysis.get("grid", {})
    counts = grid.get("sample_counts", [])
    rows = int(grid.get("rows", len(counts) or 1))
    cols = int(grid.get("cols", len(counts[0]) if counts else 1))
    if rows <= 0 or cols <= 0:
        return

    max_count = max((count for row in counts for count in row), default=0)
    x1, y1, x2, y2 = box
    cell_w = (x2 - x1) / cols
    cell_h = (y2 - y1) / rows
    for row in range(rows):
        for col in range(cols):
            count = counts[row][col] if row < len(counts) and col < len(counts[row]) else 0
            alpha = int(35 + 150 * (count / max_count)) if max_count else 20
            color = (255, 196, 62, alpha)
            cell = (
                x1 + col * cell_w,
                y1 + row * cell_h,
                x1 + (col + 1) * cell_w,
                y1 + (row + 1) * cell_h,
            )
            draw.rectangle(cell, fill=color, outline=(255, 255, 255, 70), width=1)


def _draw_trajectory(
    draw: ImageDraw.ImageDraw,
    analysis: dict,
    box: tuple[int, int, int, int],
    field_length: float,
    field_width: float,
) -> None:
    points = [
        _project_record(record, box, field_length, field_width)
        for record in analysis.get("path", [])
        if record.get("inside_field", True)
    ]
    if len(points) >= 2:
        for prev, current in zip(points, points[1:]):
            draw.line(
                (prev[0], prev[1], current[0], current[1]),
                fill=(35, 42, 62, 210),
                width=4,
            )
    for index, point in enumerate(points):
        radius = 5 if index in {0, len(points) - 1} else 3
        color = (75, 185, 95, 255) if index == 0 else (236, 72, 56, 255)
        if 0 < index < len(points) - 1:
            color = (245, 245, 245, 210)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
            outline=(20, 20, 20, 120),
        )


def _draw_summary(
    draw: ImageDraw.ImageDraw,
    analysis: dict,
    box: tuple[int, int, int, int],
    font,
) -> None:
    summary = analysis.get("summary", {})
    x1, _, _, y2 = box
    lines = [
        f"Samples: {summary.get('path_samples', 0)}",
        f"Distance: {float(summary.get('distance_m', 0.0)):.2f} m",
        f"Mean speed: {float(summary.get('mean_speed_m_s', 0.0)):.2f} m/s",
        f"Max speed: {float(summary.get('max_speed_m_s', 0.0)):.2f} m/s",
        f"Zones: {summary.get('unique_zones', 0)}",
    ]
    y = y2 + 22
    for line in lines:
        draw.text((x1, y), line, fill=(35, 45, 40, 255), font=font)
        y += 18


def _project_record(
    record: dict,
    box: tuple[int, int, int, int],
    field_length: float,
    field_width: float,
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    field_x = float(record.get("field_x_m", 0.0))
    field_y = float(record.get("field_y_m", 0.0))
    return _to_canvas((field_x, field_y), box, field_length, field_width)


def _to_canvas(
    point: tuple[float, float],
    box: tuple[int, int, int, int],
    field_length: float,
    field_width: float,
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    px = x1 + (point[0] / field_length) * (x2 - x1)
    py = y1 + (point[1] / field_width) * (y2 - y1)
    return (px, py)
