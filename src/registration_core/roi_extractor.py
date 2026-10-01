from __future__ import annotations

import cv2
import numpy as np

from .utils import crop_roi


def extract_map_roi(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    roi = cfg.get("map_roi")
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        return crop_roi(image, (x, y, w, h)), (x, y, w, h)
    return image.copy(), (0, 0, image.shape[1], image.shape[0])


def content_bbox(mask: np.ndarray, padding: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(mask.shape[1], int(xs.max()) + padding + 1)
    y2 = min(mask.shape[0], int(ys.max()) + padding + 1)
    return x1, y1, x2 - x1, y2 - y1


def component_content_bbox(
    mask: np.ndarray,
    padding: int = 16,
    *,
    min_area: int = 300,
    keep_area_ratio: float = 0.08,
    max_components: int = 8,
) -> tuple[int, int, int, int]:
    """Return a bbox around the main observed map components.

    The screenshot ROI can contain floor numbers, bottom buttons, watermarks and
    other small UI fragments that share map-like colors.  For online matching we
    crop around the largest connected structure and similarly-sized neighbors,
    instead of letting those UI fragments stretch the observation patch.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            components.append((area, label))
    if not components:
        return content_bbox(mask, padding=padding)
    components.sort(reverse=True)
    largest = components[0][0]
    keep = np.zeros(mask.shape, dtype=np.uint8)
    kept = 0
    for area, label in components:
        if kept >= max_components:
            break
        if area < largest * float(keep_area_ratio):
            continue
        keep[labels == label] = 255
        kept += 1
    return content_bbox(keep if kept else mask, padding=padding)
