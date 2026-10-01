from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color_segmenter import SegmentationResult, edges_from_map_mask, segment_map
from .roi_extractor import component_content_bbox
from .stair_semantics import restrict_template_stair_mask
from .utils import ensure_dir, imread_unicode, imwrite_unicode, load_json, save_json


_PROCESSED_TEMPLATE_CACHE: dict[tuple[str, str], "ProcessedTemplate"] = {}


def _clean_template_region_mask(
    mask: np.ndarray,
    *,
    min_component_area: int,
    max_hole_area: int,
) -> np.ndarray:
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
            filtered[labels == label] = 255
    inverse = cv2.bitwise_not(filtered)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((inverse > 0).astype(np.uint8), 8)
    height, width = filtered.shape[:2]
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
        if not touches_border and int(area) <= max_hole_area:
            filtered[labels == label] = 255
    return filtered


def _apply_clean_template_semantics(
    image: np.ndarray,
    masks: SegmentationResult,
    prep: dict,
) -> SegmentationResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_width = max(2, min(20, min(image.shape[:2]) // 20))
    border_pixels = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_lab = np.median(border_pixels, axis=0)
    background_distance = np.linalg.norm(lab - background_lab, axis=2)
    foreground = np.where(
        background_distance >= float(prep.get("clean_template_background_lab_distance", 12.0)),
        255,
        0,
    ).astype(np.uint8)
    value_min = int(prep.get("clean_template_value_min", 45))
    corridor = cv2.inRange(
        hsv,
        np.array([int(prep.get("clean_template_corridor_h_min", 80)), 2, value_min], dtype=np.uint8),
        np.array([int(prep.get("clean_template_corridor_h_max", 150)), 90, 255], dtype=np.uint8),
    )
    room = cv2.inRange(
        hsv,
        np.array([0, int(prep.get("clean_template_room_s_min", 25)), value_min], dtype=np.uint8),
        np.array([int(prep.get("clean_template_room_h_max", 45)), 255, 255], dtype=np.uint8),
    )
    corridor = cv2.bitwise_and(corridor, foreground)
    room = cv2.bitwise_and(room, foreground)
    minimum_area = max(1, int(prep.get("edge_cleanup_min_component_area", 24)))
    max_hole_area = max(0, int(prep.get("clean_template_max_hole_area", 1200)))
    corridor = _clean_template_region_mask(
        corridor,
        min_component_area=minimum_area,
        max_hole_area=max_hole_area,
    )
    room = _clean_template_region_mask(
        room,
        min_component_area=minimum_area,
        max_hole_area=max_hole_area,
    )
    corridor = cv2.bitwise_and(corridor, cv2.bitwise_not(room))
    map_mask = cv2.bitwise_or(corridor, room)
    edge = edges_from_map_mask(
        map_mask,
        int(prep.get("edge_canny1", 45)),
        int(prep.get("edge_canny2", 135)),
        mode="semantic_boundary",
        cleanup_open_kernel=int(prep.get("edge_cleanup_open_kernel", 3)),
        cleanup_close_kernel=int(prep.get("edge_cleanup_close_kernel", 7)),
        semantic_masks=(corridor, room),
        cleanup_min_component_area=minimum_area,
        cleanup_max_hole_area=max_hole_area,
    )
    zeros = np.zeros_like(map_mask)
    return SegmentationResult(
        corridor_mask=corridor,
        room_mask=room,
        map_mask=map_mask,
        edge_mask=edge,
        stair_mask=masks.stair_mask,
        visible_mask=map_mask.copy(),
        valid_mask=map_mask.copy(),
        unknown_mask=cv2.bitwise_not(map_mask),
        ignore_mask=zeros,
        known_background_mask=zeros,
    )


@dataclass
class ProcessedTemplate:
    map_id: str
    image: np.ndarray
    corridor_mask: np.ndarray
    room_mask: np.ndarray
    map_mask: np.ndarray
    edge_mask: np.ndarray
    stair_mask: np.ndarray
    distance_transform: np.ndarray
    crop_offset: tuple[int, int]
    raw_path: str
    corridor_float: np.ndarray | None = None
    room_float: np.ndarray | None = None


def _apply_bbox(image: np.ndarray, masks: SegmentationResult, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, SegmentationResult]:
    x, y, w, h = bbox
    cropped = image[y : y + h, x : x + w].copy()
    return cropped, SegmentationResult(
        corridor_mask=masks.corridor_mask[y : y + h, x : x + w].copy(),
        room_mask=masks.room_mask[y : y + h, x : x + w].copy(),
        map_mask=masks.map_mask[y : y + h, x : x + w].copy(),
        edge_mask=masks.edge_mask[y : y + h, x : x + w].copy(),
        stair_mask=masks.stair_mask[y : y + h, x : x + w].copy(),
        visible_mask=masks.visible_mask[y : y + h, x : x + w].copy(),
        valid_mask=masks.valid_mask[y : y + h, x : x + w].copy() if masks.valid_mask is not None else None,
        unknown_mask=masks.unknown_mask[y : y + h, x : x + w].copy() if masks.unknown_mask is not None else None,
        ignore_mask=masks.ignore_mask[y : y + h, x : x + w].copy() if masks.ignore_mask is not None else None,
        known_background_mask=(
            masks.known_background_mask[y : y + h, x : x + w].copy()
            if masks.known_background_mask is not None
            else None
        ),
    )


def preprocess_template(
    path: Path,
    cfg: dict,
    output_dir: Path,
    *,
    map_id: str | None = None,
) -> ProcessedTemplate:
    image = imread_unicode(path)
    if image is None:
        raise RuntimeError(f"无法读取模板图: {path}")
    # Clean templates contain complete structure but no game-side known
    # background. That negative evidence exists only in a captured ROI.
    masks = segment_map(image, cfg, infer_known_background=False)
    prep = cfg.get("preprocess", {})
    if bool(prep.get("clean_template_semantic_segmentation", False)):
        masks = _apply_clean_template_semantics(image, masks, prep)
    if bool(prep.get("disable_template_crop", False)):
        bbox = (0, 0, image.shape[1], image.shape[0])
    else:
        padding = int(prep.get("crop_padding", 16))
        min_area = int(prep.get("min_map_area", 300))
        bbox = component_content_bbox(masks.map_mask, padding=padding, min_area=min_area)
    image, masks = _apply_bbox(image, masks, bbox)
    masks.stair_mask = restrict_template_stair_mask(
        masks.stair_mask,
        map_id=map_id or path.stem,
        cfg=cfg,
        crop_offset=(bbox[0], bbox[1]),
    )
    dist = cv2.distanceTransform(cv2.bitwise_not(masks.edge_mask), cv2.DIST_L2, 3)
    map_id = map_id or path.stem
    target = ensure_dir(output_dir / map_id)
    imwrite_unicode(target / "image.png", image)
    imwrite_unicode(target / "corridor_mask.png", masks.corridor_mask)
    imwrite_unicode(target / "room_mask.png", masks.room_mask)
    imwrite_unicode(target / "map_mask.png", masks.map_mask)
    imwrite_unicode(target / "edge_mask.png", masks.edge_mask)
    imwrite_unicode(target / "stair_mask.png", masks.stair_mask)
    np.save(target / "distance_transform.npy", dist.astype(np.float32))
    save_json(
        target / "meta.json",
        {
            "map_id": map_id,
            "raw_path": str(path),
            "crop_offset": [int(bbox[0]), int(bbox[1])],
            "shape": [int(image.shape[0]), int(image.shape[1])],
        },
    )
    return load_processed_template(output_dir, map_id)


def load_processed_template(processed_dir: Path, map_id: str) -> ProcessedTemplate:
    resolved_dir = processed_dir.resolve()
    cache_key = (str(resolved_dir), str(map_id))
    cached = _PROCESSED_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    target = resolved_dir / map_id
    meta = load_json(target / "meta.json")
    image = imread_unicode(target / "image.png")
    corridor = imread_unicode(target / "corridor_mask.png", cv2.IMREAD_GRAYSCALE)
    room = imread_unicode(target / "room_mask.png", cv2.IMREAD_GRAYSCALE)
    map_mask = imread_unicode(target / "map_mask.png", cv2.IMREAD_GRAYSCALE)
    edge = imread_unicode(target / "edge_mask.png", cv2.IMREAD_GRAYSCALE)
    stairs = imread_unicode(target / "stair_mask.png", cv2.IMREAD_GRAYSCALE)
    dist = np.load(target / "distance_transform.npy").astype(np.float32)
    if any(v is None for v in (image, corridor, room, map_mask, edge, stairs)):
        raise RuntimeError(f"预处理模板不完整: {target}")
    offset = meta.get("crop_offset", [0, 0])
    template = ProcessedTemplate(
        map_id=str(meta.get("map_id", map_id)),
        image=image,
        corridor_mask=corridor,
        room_mask=room,
        map_mask=map_mask,
        edge_mask=edge,
        stair_mask=stairs,
        distance_transform=dist,
        crop_offset=(int(offset[0]), int(offset[1])),
        raw_path=str(meta.get("raw_path", "")),
        corridor_float=corridor.astype(np.float32) / 255.0,
        room_float=room.astype(np.float32) / 255.0,
    )
    _PROCESSED_TEMPLATE_CACHE[cache_key] = template
    return template


def preprocess_template_dir(template_dir: Path, processed_dir: Path, cfg: dict) -> list[str]:
    ensure_dir(processed_dir)
    prefixes_raw = cfg.get("include_map_prefixes") or []
    include_prefixes = {str(item) for item in prefixes_raw} if isinstance(prefixes_raw, list) else set()
    map_ids: list[str] = []
    for path in sorted(template_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        if include_prefixes and not any(path.stem.startswith(prefix) for prefix in include_prefixes):
            continue
        # Some source assets carry a manual review suffix such as
        # ``map_23_1st(03).jpg``. Runtime lookup uses ``map_23_1st``; keep the
        # source filename for provenance but normalize the generated cache ID.
        map_id = path.stem
        if map_id.startswith("map_") and map_id.endswith(")") and "(" in map_id:
            map_id = map_id.rsplit("(", 1)[0]
        preprocess_template(path, cfg, processed_dir, map_id=map_id)
        map_ids.append(map_id)
    return map_ids
