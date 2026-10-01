from __future__ import annotations

import cv2
import numpy as np

from .masked_matcher import LocalizationResult
from .utils import draw_label, imwrite_unicode, to_bgr


def draw_candidates(template_image: np.ndarray, result: LocalizationResult) -> np.ndarray:
    out = template_image.copy()
    for idx, candidate in enumerate(result.top_k_candidates, 1):
        x, y = candidate.top_left
        color = (0, 255, 255) if idx == 1 else (0, 160, 255)
        if result.observation_shape is not None:
            obs_h, obs_w = result.observation_shape
            rw = int(round(obs_w * candidate.scale))
            rh = int(round(obs_h * candidate.scale))
            cv2.rectangle(out, (int(x), int(y)), (int(x + rw), int(y + rh)), color, 2, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
        draw_label(out, f"{idx}:{candidate.score:.3f}", (int(x) + 10, int(y) - 6), color)
    display_player = result.player_position_in_template_crop or result.player_position_in_template
    if display_player is not None:
        px, py = display_player
        cv2.circle(out, (int(round(px)), int(round(py))), 12, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(px)), int(round(py))), 18, (0, 0, 255), 2, cv2.LINE_AA)
        draw_label(out, "player", (int(px) + 16, int(py) - 10), (0, 255, 255))
    return out


def draw_player_candidates(roi: np.ndarray, candidates: list[tuple[float, float, float, tuple[int, int, int, int], float]]) -> np.ndarray:
    out = roi.copy()
    for idx, (area, cx, cy, rect, score) in enumerate(candidates[:12], 1):
        x, y, w, h = rect
        color = (0, 255, 0) if idx == 1 else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 5, color, -1, cv2.LINE_AA)
        draw_label(out, f"{idx}:{area:.0f}/{score:.2f}", (x, max(18, y - 4)), color)
    return out


def _shift_player_candidates(candidates, offset: tuple[int, int], shape: tuple[int, int]):
    ox, oy = offset
    h, w = shape
    shifted = []
    for area, cx, cy, rect, score in candidates:
        x, y, rw, rh = rect
        sx, sy = x - ox, y - oy
        scx, scy = cx - ox, cy - oy
        if sx + rw < 0 or sy + rh < 0 or sx >= w or sy >= h:
            continue
        shifted.append((area, scx, scy, (int(sx), int(sy), int(rw), int(rh)), score))
    return shifted


def save_debug_images(
    output_dir,
    *,
    roi,
    original_roi=None,
    segmentation,
    ignore_mask,
    player_detection,
    template_image,
    result,
    full: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.player_roi is not None and player_detection.center is not None:
        ox = int(round(player_detection.center[0] - result.player_roi[0]))
        oy = int(round(player_detection.center[1] - result.player_roi[1]))
    else:
        ox = 0
        oy = 0
    shifted_candidates = _shift_player_candidates(player_detection.candidates, (ox, oy), roi.shape[:2])
    images = {
        "roi": roi,
        "corridor_mask": segmentation.corridor_mask,
        "room_mask": segmentation.room_mask,
        "map_mask": segmentation.map_mask,
        "visible_mask": segmentation.visible_mask,
        "valid_mask": segmentation.valid_mask if segmentation.valid_mask is not None else segmentation.visible_mask,
        "unknown_mask": segmentation.unknown_mask if segmentation.unknown_mask is not None else np.zeros_like(segmentation.visible_mask),
        "known_background_mask": (
            segmentation.known_background_mask
            if segmentation.known_background_mask is not None
            else np.zeros_like(segmentation.visible_mask)
        ),
        "edge_mask": segmentation.edge_mask,
        "stair_mask": segmentation.stair_mask,
        "ignore_mask": ignore_mask,
        "player_candidates": draw_player_candidates(roi, shifted_candidates),
        "candidates_on_template": draw_candidates(template_image, result),
    }
    if original_roi is not None:
        images["original_roi"] = original_roi
    if not full:
        images = {"candidates_on_template": images["candidates_on_template"]}
    paths: dict[str, str] = {}
    for name, image in images.items():
        path = output_dir / f"{name}.png"
        imwrite_unicode(path, to_bgr(image))
        paths[name] = str(path)
    return paths
