from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    corridor_mask: np.ndarray
    room_mask: np.ndarray
    map_mask: np.ndarray
    edge_mask: np.ndarray
    stair_mask: np.ndarray
    visible_mask: np.ndarray
    # Three-state ROI evidence for geometric verification. valid_mask contains
    # visible structure plus reliable known-negative background. Everything
    # else outside ignore_mask remains unknown fog.
    valid_mask: np.ndarray | None = None
    unknown_mask: np.ndarray | None = None
    ignore_mask: np.ndarray | None = None
    known_background_mask: np.ndarray | None = None
    # Screenshot-derived fog ambiguity. Must never become negative evidence.
    uncertain_mask: np.ndarray | None = None


def mask_from_hsv_ranges(
    image_bgr: np.ndarray,
    ranges: list[list[int]],
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for item in ranges or []:
        if len(item) != 6:
            continue
        lo = np.array(item[:3], dtype=np.uint8)
        hi = np.array(item[3:], dtype=np.uint8)
        out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))
    return out


def cleanup_mask(mask: np.ndarray, kernel_size: int = 3, close_iter: int = 1, open_iter: int = 1) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = mask
    if close_iter:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return out


def visible_mask(
    image_bgr: np.ndarray,
    colors_cfg: dict[str, Any],
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark_v = int(colors_cfg.get("dark_background_v_max", 38))
    s_min = int(colors_cfg.get("visible_s_min", 14))
    v_min = int(colors_cfg.get("visible_v_min", 42))
    bright = cv2.compare(val, v_min, cv2.CMP_GT)
    saturated = cv2.compare(sat, s_min, cv2.CMP_GT)
    clearly_bright = cv2.compare(val, dark_v + 35, cv2.CMP_GT)
    mask = cv2.bitwise_and(bright, cv2.bitwise_or(saturated, clearly_bright))
    return cleanup_mask(mask, 3, close_iter=1, open_iter=1)


def edges_from_map_mask(
    map_mask: np.ndarray,
    canny1: int,
    canny2: int,
    *,
    mode: str = "canny",
    cleanup_open_kernel: int = 3,
    cleanup_close_kernel: int = 5,
    semantic_masks: tuple[np.ndarray, ...] | None = None,
    cleanup_min_component_area: int = 24,
    cleanup_max_hole_area: int = 320,
) -> np.ndarray:
    if str(mode).strip().lower() in {"semantic_boundary", "boundary", "clean_boundary"}:
        open_size = max(1, int(cleanup_open_kernel)) | 1
        close_size = max(1, int(cleanup_close_kernel)) | 1
        masks = semantic_masks if semantic_masks else (map_mask,)
        combined_edges = np.zeros_like(map_mask)
        for source_mask in masks:
            cleaned = cv2.morphologyEx(
                source_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
            )
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (cleaned > 0).astype(np.uint8), connectivity=8
            )
            filtered = np.zeros_like(cleaned)
            minimum_area = max(1, int(cleanup_min_component_area))
            for label in range(1, count):
                if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                    filtered[labels == label] = 255

            inverse = cv2.bitwise_not(filtered)
            hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
                (inverse > 0).astype(np.uint8), connectivity=8
            )
            max_hole_area = max(0, int(cleanup_max_hole_area))
            height, width = filtered.shape[:2]
            for label in range(1, hole_count):
                x, y, component_width, component_height, area = hole_stats[label]
                touches_border = x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
                if not touches_border and int(area) <= max_hole_area:
                    filtered[hole_labels == label] = 255
            edges = cv2.morphologyEx(
                filtered,
                cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        return combined_edges
    x, y, width, height = cv2.boundingRect(map_mask)
    if width <= 0 or height <= 0:
        return np.zeros_like(map_mask)
    # Gaussian, Canny, and the closing operation have a combined finite
    # support smaller than this halo, so the cropped result is pixel-identical.
    halo = 4
    x0 = max(0, x - halo)
    y0 = max(0, y - halo)
    x1 = min(map_mask.shape[1], x + width + halo)
    y1 = min(map_mask.shape[0], y + height + halo)
    local_mask = map_mask[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(local_mask, (3, 3), 0)
    local_edges = cv2.Canny(blurred, int(canny1), int(canny2))
    local_edges = cleanup_mask(local_edges, 3, close_iter=1, open_iter=0)
    edges = np.zeros_like(map_mask)
    edges[y0:y1, x0:x1] = local_edges
    return edges


def semantic_component_counts(
    segmentation: SegmentationResult,
    *,
    min_area: int = 60,
) -> dict[str, int]:
    """Count meaningful room and corridor regions in a cropped observation."""
    minimum = max(1, int(min_area))

    def count(mask: np.ndarray) -> int:
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
        return int(sum(area >= minimum for area in stats[1:, cv2.CC_STAT_AREA]))

    rooms = count(segmentation.room_mask)
    corridors = count(segmentation.corridor_mask)
    return {
        "room_components": rooms,
        "corridor_components": corridors,
        "total_components": rooms + corridors,
        "min_component_area": minimum,
    }


def known_background_from_dark_rim(
    image_bgr: np.ndarray,
    map_mask: np.ndarray,
    ignore_mask: np.ndarray,
    colors_cfg: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    """Return only the deep background rim immediately outside visible structure."""
    if cv2.countNonZero(map_mask) == 0:
        return np.zeros_like(map_mask)
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    dark_max = int(colors_cfg.get("known_background_v_max", colors_cfg.get("dark_background_v_max", 45)))
    dark = np.where(value <= dark_max, 255, 0).astype(np.uint8)
    radius = max(1, int(preprocess_cfg.get("known_background_max_distance", 16)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    near_structure = cv2.dilate(map_mask, kernel)
    rim = cv2.bitwise_and(dark, near_structure)
    rim = cv2.bitwise_and(rim, cv2.bitwise_not(map_mask))
    rim = cv2.bitwise_and(rim, cv2.bitwise_not(ignore_mask))
    return cleanup_mask(rim, 3, close_iter=1, open_iter=0)


def stair_mask_from_stripes(image_bgr: np.ndarray, prep: dict[str, Any]) -> np.ndarray:
    """Extract neutral horizontal and vertical stripe evidence from a map ROI."""
    cfg = prep.get("stairs", {})
    out = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return out

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (3, 3), 0),
        int(cfg.get("edge_canny1", 50)),
        int(cfg.get("edge_canny2", 120)),
    )
    line_length = max(2, int(cfg.get("line_min_length", 3)))
    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)),
    )
    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)),
    )
    return cv2.bitwise_or(horizontal, vertical)


def segment_map(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
    *,
    ignore_mask: np.ndarray | None = None,
    infer_known_background: bool = True,
    structure_only: bool = False,
    include_stairs: bool | None = None,
) -> SegmentationResult:
    colors = cfg.get("colors", {})
    prep = cfg.get("preprocess", {})
    kernel_size = int(prep.get("morph_kernel", 3))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    corridor = mask_from_hsv_ranges(image_bgr, colors.get("corridor_hsv", []), hsv=hsv)
    room = mask_from_hsv_ranges(image_bgr, colors.get("room_hsv", []), hsv=hsv)
    should_extract_stairs = not structure_only and (include_stairs is None or include_stairs)
    processing_bbox: tuple[int, int, int, int] | None = None
    can_crop = (
        ignore_mask is not None
        and not should_extract_stairs
        and bool(cfg.get("local_segmentation_static_ignore_crop_enabled", True))
    )
    if can_crop:
        raw_structure = cv2.bitwise_or(corridor, room)
        raw_structure = cv2.bitwise_and(raw_structure, cv2.bitwise_not(ignore_mask))
        x, y, width, height = cv2.boundingRect(raw_structure)
        if width > 0 and height > 0:
            background_radius = (
                max(1, int(prep.get("known_background_max_distance", 16)))
                if infer_known_background and not structure_only
                else 0
            )
            halo = max(8, background_radius + 4, 4 * (kernel_size // 2))
            x0 = max(0, x - halo)
            y0 = max(0, y - halo)
            x1 = min(image_bgr.shape[1], x + width + halo)
            y1 = min(image_bgr.shape[0], y + height + halo)
            processing_bbox = (x0, y0, x1, y1)
            corridor = corridor[y0:y1, x0:x1]
            room = room[y0:y1, x0:x1]
            image_bgr = image_bgr[y0:y1, x0:x1]
            hsv = hsv[y0:y1, x0:x1]
            ignore_mask_local = ignore_mask[y0:y1, x0:x1]
        else:
            ignore_mask_local = ignore_mask
    else:
        ignore_mask_local = ignore_mask
    stairs = stair_mask_from_stripes(image_bgr, prep) if should_extract_stairs else np.zeros(
        image_bgr.shape[:2], dtype=np.uint8
    )
    corridor = cleanup_mask(corridor, kernel_size)
    room = cleanup_mask(room, kernel_size)
    vis = visible_mask(image_bgr, colors, hsv=hsv)
    if ignore_mask_local is not None:
        inv_ignore = cv2.bitwise_not(ignore_mask_local)
        corridor = cv2.bitwise_and(corridor, inv_ignore)
        room = cv2.bitwise_and(room, inv_ignore)
        stairs = cv2.bitwise_and(stairs, inv_ignore)
        vis = cv2.bitwise_and(vis, inv_ignore)
    else:
        ignore_mask_local = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    map_mask = cv2.bitwise_or(corridor, room)
    map_mask = cv2.bitwise_and(map_mask, vis)
    edge = edges_from_map_mask(
        map_mask,
        int(prep.get("edge_canny1", 45)),
        int(prep.get("edge_canny2", 135)),
        mode=str(prep.get("edge_mode", "canny")),
        cleanup_open_kernel=int(prep.get("edge_cleanup_open_kernel", 3)),
        cleanup_close_kernel=int(prep.get("edge_cleanup_close_kernel", 5)),
        semantic_masks=(corridor, room),
        cleanup_min_component_area=int(prep.get("edge_cleanup_min_component_area", 24)),
        cleanup_max_hole_area=int(prep.get("edge_cleanup_max_hole_area", 320)),
    )
    # known_background is observation-side negative evidence. A clean template
    # is complete structure and must never infer this channel from its colors.
    if infer_known_background and not structure_only:
        known_background = known_background_from_dark_rim(
            image_bgr,
            map_mask,
            ignore_mask_local,
            colors,
            prep,
            hsv=hsv,
        )
    else:
        known_background = np.zeros_like(map_mask)
    valid = cv2.bitwise_or(map_mask, known_background)
    unknown = cv2.bitwise_not(cv2.bitwise_or(valid, ignore_mask_local))
    if processing_bbox is not None:
        x0, y0, x1, y1 = processing_bbox
        full_shape = ignore_mask.shape

        def expand(mask: np.ndarray) -> np.ndarray:
            out = np.zeros(full_shape, dtype=mask.dtype)
            out[y0:y1, x0:x1] = mask
            return out

        corridor = expand(corridor)
        room = expand(room)
        map_mask = expand(map_mask)
        edge = expand(edge)
        stairs = expand(stairs)
        vis = expand(vis)
        known_background = expand(known_background)
        valid = cv2.bitwise_or(map_mask, known_background)
        unknown = cv2.bitwise_not(cv2.bitwise_or(valid, ignore_mask))
    elif ignore_mask is None:
        ignore_mask = ignore_mask_local
    return SegmentationResult(
        corridor,
        room,
        map_mask,
        edge,
        stairs,
        vis,
        valid,
        unknown,
        ignore_mask,
        known_background,
    )


def apply_additional_ignore_mask(
    image_bgr: np.ndarray,
    segmentation: SegmentationResult,
    ignore_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    infer_known_background: bool = True,
) -> SegmentationResult:
    """Apply a late-arriving ignore mask without repeating HSV segmentation."""
    colors = cfg.get("colors", {})
    prep = cfg.get("preprocess", {})
    inv_ignore = cv2.bitwise_not(ignore_mask)
    corridor = cv2.bitwise_and(segmentation.corridor_mask, inv_ignore)
    room = cv2.bitwise_and(segmentation.room_mask, inv_ignore)
    stairs = cv2.bitwise_and(segmentation.stair_mask, inv_ignore)
    vis = cv2.bitwise_and(segmentation.visible_mask, inv_ignore)
    map_mask = cv2.bitwise_and(cv2.bitwise_or(corridor, room), vis)
    edge = edges_from_map_mask(
        map_mask,
        int(prep.get("edge_canny1", 45)),
        int(prep.get("edge_canny2", 135)),
        mode=str(prep.get("edge_mode", "canny")),
        cleanup_open_kernel=int(prep.get("edge_cleanup_open_kernel", 3)),
        cleanup_close_kernel=int(prep.get("edge_cleanup_close_kernel", 5)),
        semantic_masks=(corridor, room),
        cleanup_min_component_area=int(prep.get("edge_cleanup_min_component_area", 24)),
        cleanup_max_hole_area=int(prep.get("edge_cleanup_max_hole_area", 320)),
    )
    known_background = (
        known_background_from_dark_rim(image_bgr, map_mask, ignore_mask, colors, prep)
        if infer_known_background
        else np.zeros_like(map_mask)
    )
    if segmentation.uncertain_mask is not None:
        known_background[segmentation.uncertain_mask > 0] = 0
    valid = cv2.bitwise_or(map_mask, known_background)
    unknown = cv2.bitwise_not(cv2.bitwise_or(valid, ignore_mask))
    return SegmentationResult(
        corridor, room, map_mask, edge, stairs, vis, valid, unknown, ignore_mask, known_background,
        segmentation.uncertain_mask,
    )


def add_known_background_evidence(
    image_bgr: np.ndarray,
    segmentation: SegmentationResult,
    cfg: dict[str, Any],
) -> SegmentationResult:
    """Add the negative-evidence channel only when geometric review needs it."""
    ignore_mask = segmentation.ignore_mask
    if ignore_mask is None:
        ignore_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    known_background = known_background_from_dark_rim(
        image_bgr,
        segmentation.map_mask,
        ignore_mask,
        cfg.get("colors", {}),
        cfg.get("preprocess", {}),
    )
    if segmentation.uncertain_mask is not None:
        known_background[segmentation.uncertain_mask > 0] = 0
    valid = cv2.bitwise_or(segmentation.map_mask, known_background)
    unknown = cv2.bitwise_not(cv2.bitwise_or(valid, ignore_mask))
    return SegmentationResult(
        segmentation.corridor_mask,
        segmentation.room_mask,
        segmentation.map_mask,
        segmentation.edge_mask,
        segmentation.stair_mask,
        segmentation.visible_mask,
        valid,
        unknown,
        ignore_mask,
        known_background,
        segmentation.uncertain_mask,
    )
