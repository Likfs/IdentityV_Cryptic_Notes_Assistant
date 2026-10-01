"""Bounded full-resolution refinement for a local structure-index candidate."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .template_preprocessor import ProcessedTemplate


_WEIGHTS = np.sqrt(np.float32([0.48, 0.32, 0.20]))


@dataclass(frozen=True)
class LocalRefinement:
    position: tuple[int, int]
    scale: float
    score: float


def refine_local_structure(
    template: ProcessedTemplate,
    observation: SegmentationResult,
    *,
    position: tuple[float, float],
    scales: list[float],
    radius: int,
) -> LocalRefinement | None:
    """Search a small translation window at a few scales using semantic edges.

    Positions are in processed-template pixels.  The local canvas is padded
    with zeroes so observations touching the template border remain searchable.
    """
    radius = max(1, int(radius))
    observed = np.stack((observation.corridor_mask, observation.room_mask,
                         observation.edge_mask), axis=2)
    height, width = template.map_mask.shape
    best: LocalRefinement | None = None
    for scale in scales:
        patch_size = (max(3, int(round(observed.shape[1] * scale))),
                      max(3, int(round(observed.shape[0] * scale))))
        patch = cv2.resize(observed, patch_size,
                           interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        patch *= _WEIGHTS
        if not np.any(patch):
            continue
        predicted_x = int(round(position[0]))
        predicted_y = int(round(position[1]))
        left = predicted_x - radius
        top = predicted_y - radius
        canvas_width = patch.shape[1] + 2 * radius
        canvas_height = patch.shape[0] + 2 * radius
        right = left + canvas_width
        bottom = top + canvas_height
        sx0, sy0 = max(0, left), max(0, top)
        sx1, sy1 = min(width, right), min(height, bottom)
        canvas = np.zeros((canvas_height, canvas_width, 3), np.float32)
        if sx0 < sx1 and sy0 < sy1:
            destination = canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left]
            sl = np.s_[sy0:sy1, sx0:sx1]
            destination[:, :, 0] = template.corridor_mask[sl] * (_WEIGHTS[0] / 255.0)
            destination[:, :, 1] = template.room_mask[sl] * (_WEIGHTS[1] / 255.0)
            destination[:, :, 2] = template.edge_mask[sl] * (_WEIGHTS[2] / 255.0)
        response = cv2.matchTemplate(canvas, patch, cv2.TM_CCORR_NORMED)
        _, score, _, point = cv2.minMaxLoc(response)
        if not np.isfinite(score):
            continue
        found = LocalRefinement((left + point[0], top + point[1]),
                                float(scale), float(score))
        if best is None or found.score > best.score:
            best = found
    return best
