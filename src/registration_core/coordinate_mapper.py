from __future__ import annotations

import numpy as np


def map_roi_point_to_template(
    point: tuple[float, float],
    *,
    top_left: tuple[int, int],
    scale: float,
    template_crop_offset: tuple[int, int] = (0, 0),
) -> tuple[float, float]:
    return (
        float(template_crop_offset[0] + top_left[0] + point[0] * scale),
        float(template_crop_offset[1] + top_left[1] + point[1] * scale),
    )


def weighted_mean_position(points: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    weights = np.array([p[2] for p in points], dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-9:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)
