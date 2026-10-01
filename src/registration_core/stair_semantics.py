from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _transition_links_path(cfg: dict[str, Any]) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    configured = cfg.get("stair_transition_links_path")
    candidates = []
    if configured:
        path = Path(str(configured))
        candidates.append(path if path.is_absolute() else project_root / path)
    return next((path for path in candidates if path.exists()), None)


def transition_anchor_points(map_id: str, cfg: dict[str, Any]) -> list[tuple[float, float]]:
    path = _transition_links_path(cfg)
    if path is None:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    base, separator, floor = map_id.rpartition("_")
    if not separator or floor not in {"1st", "2nd"}:
        return []
    items = data.get("maps", {}).get(base, []) if isinstance(data, dict) else []
    points: list[tuple[float, float]] = []
    for item in items if isinstance(items, list) else []:
        value = item.get(floor) if isinstance(item, dict) else None
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(axis, (int, float)) for axis in value[:2]):
            points.append((float(value[0]), float(value[1])))
    return points


def restrict_template_stair_mask(
    stripe_mask: np.ndarray,
    *,
    map_id: str,
    cfg: dict[str, Any],
    crop_offset: tuple[int, int],
) -> np.ndarray:
    prep = cfg.get("preprocess", {})
    stair_cfg = prep.get("stairs", {}) if isinstance(prep, dict) else {}
    if not isinstance(stair_cfg, dict) or not bool(stair_cfg.get("enabled", False)):
        return stripe_mask
    anchors = transition_anchor_points(map_id, cfg)
    if not anchors:
        return stripe_mask
    radius = max(4, int(stair_cfg.get("template_anchor_radius", 38)))
    allowed = np.zeros(stripe_mask.shape, dtype=np.uint8)
    offset_x, offset_y = crop_offset
    for x, y in anchors:
        cv2.circle(allowed, (round(x - offset_x), round(y - offset_y)), radius, 255, -1)
    return cv2.bitwise_and(stripe_mask, allowed)
