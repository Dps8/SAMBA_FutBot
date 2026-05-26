from __future__ import annotations

import numpy as np


def dominant_rgb(frame_rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int, int]:
    pixels = frame_rgb[mask > 0] if mask is not None else frame_rgb.reshape(-1, 3)
    if pixels.size == 0:
        return (0, 0, 0)
    median = np.median(pixels, axis=0)
    return tuple(int(v) for v in median[:3])


def nearest_palette_team(
    rgb: tuple[int, int, int], palette: dict[str, tuple[int, int, int]]
) -> tuple[str, float]:
    color = np.asarray(rgb, dtype=np.float64)
    best_team = "unknown"
    best_distance = float("inf")
    for team, value in palette.items():
        distance = float(np.linalg.norm(color - np.asarray(value, dtype=np.float64)))
        if distance < best_distance:
            best_team = team
            best_distance = distance
    return best_team, best_distance
