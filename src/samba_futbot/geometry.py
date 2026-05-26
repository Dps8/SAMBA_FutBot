from __future__ import annotations

import numpy as np


def estimate_homography(
    source_points: list[tuple[float, float]], target_points: list[tuple[float, float]]
) -> np.ndarray:
    """Estimate a 3x3 homography with OpenCV when present, otherwise DLT."""

    if len(source_points) < 4 or len(target_points) < 4:
        raise ValueError("At least four source and target points are required.")
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    try:
        import cv2

        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("OpenCV could not estimate homography.")
        return matrix
    except ImportError:
        return _dlt_homography(src, dst)


def transform_points(points: list[tuple[float, float]], homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1) @ homography.T
    homogeneous[:, 0] /= homogeneous[:, 2]
    homogeneous[:, 1] /= homogeneous[:, 2]
    return homogeneous[:, :2]


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vh[-1].reshape(3, 3)
    return h / h[2, 2]
