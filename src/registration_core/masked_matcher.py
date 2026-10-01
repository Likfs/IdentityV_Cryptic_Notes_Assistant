from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .coordinate_mapper import map_roi_point_to_template
from .template_preprocessor import ProcessedTemplate


_MATCH_CHANNEL_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cv-match-channel")


@dataclass
class MatchCandidate:
    top_left: tuple[int, int]
    scale: float
    score: float
    corridor_score: float
    room_score: float
    edge_score: float
    stair_score: float
    semantic_consistency: float | None = None
    semantic_extra_structure_rate: float | None = None
    directional_chamfer_score: float | None = None
    edge_precision: float | None = None
    edge_recall: float | None = None
    review_score: float | None = None
    stage1_score: float | None = None


@dataclass
class LocalizationResult:
    map_id: str
    best_position: tuple[int, int] | None
    best_scale: float | None
    confidence: float
    low_confidence: bool
    top_k_candidates: list[MatchCandidate]
    player_roi: tuple[float, float] | None
    player_position_in_template: tuple[float, float] | None
    player_position_in_template_crop: tuple[float, float] | None = None
    observation_shape: tuple[int, int] | None = None
    geometric_review_applied: bool | None = None
    geometric_review_stage1_margin: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["top_k_candidates"] = [asdict(c) for c in self.top_k_candidates]
        return data


def _scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if abs(float(scale) - 1.0) < 1e-9:
        return mask
    return cv2.resize(
        mask,
        (max(1, int(round(mask.shape[1] * scale))), max(1, int(round(mask.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )


def _scale_masks(masks: list[np.ndarray], scale: float) -> list[np.ndarray]:
    """Resize same-sized binary channels in one OpenCV operation."""
    if not masks:
        return []
    stacked = np.stack(masks, axis=2)
    scaled = cv2.resize(
        stacked,
        (
            max(1, int(round(stacked.shape[1] * scale))),
            max(1, int(round(stacked.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    if scaled.ndim == 2:
        scaled = scaled[:, :, None]
    return [np.ascontiguousarray(scaled[:, :, index]) for index in range(scaled.shape[2])]


def _safe_match(
    template: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    template_float: np.ndarray | None = None,
    binary_fast: bool = False,
) -> np.ndarray | None:
    if template.shape[0] < patch.shape[0] or template.shape[1] < patch.shape[1]:
        return None
    if binary_fast and mask is patch and cv2.countNonZero(patch) >= 20:
        # For binary T,P with mask=P: CCORR_NORMED = sqrt(overlap / |P|).
        # One unmasked correlation replaces the masked norm convolutions.
        overlap = _support_correlation(template, patch)
        overlap = np.rint(overlap / np.float32(255 * 255))
        return np.sqrt(np.clip(overlap / cv2.countNonZero(patch), 0.0, 1.0))
    patch_f = (patch.astype(np.float32) / 255.0)
    template_f = template_float if template_float is not None else (template.astype(np.float32) / 255.0)
    if mask is not None and cv2.countNonZero(mask) >= 20:
        return cv2.matchTemplate(template_f, patch_f, cv2.TM_CCORR_NORMED, mask=mask)
    return cv2.matchTemplate(template_f, patch_f, cv2.TM_CCORR_NORMED)


def _support_correlation(template: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Remove zero-only patch borders without removing a single output position.

    Shift the template view by the same amount; its end includes exactly the
    original response extent plus the reduced patch size. This is an exact
    correlation identity, not a smaller pose search or a confidence shortcut.
    """
    x, y, width, height = cv2.boundingRect(patch if patch.dtype == np.uint8 else np.uint8(patch != 0))
    response_h = template.shape[0]-patch.shape[0]+1
    response_w = template.shape[1]-patch.shape[1]+1
    if not width or not height:
        return np.zeros((response_h, response_w), np.float32)
    region = template[y:y+response_h+height-1, x:x+response_w+width-1]
    return cv2.matchTemplate(region, patch[y:y+height, x:x+width], cv2.TM_CCORR)


def _safe_match_in_windows(
    template: np.ndarray,
    patch: np.ndarray,
    windows: list[tuple[int, int, int, int]] | None,
    mask: np.ndarray | None = None,
    *,
    template_float: np.ndarray | None = None,
    binary_fast: bool = False,
) -> np.ndarray | None:
    """Evaluate matchTemplate only inside inclusive response-coordinate windows."""
    if windows is None:
        return _safe_match(template, patch, mask, template_float=template_float, binary_fast=binary_fast)
    response_h = template.shape[0] - patch.shape[0] + 1
    response_w = template.shape[1] - patch.shape[1] + 1
    if response_h <= 0 or response_w <= 0:
        return None
    response = np.zeros((response_h, response_w), dtype=np.float32)
    for x0, y0, x1, y1 in windows:
        template_region = template[y0 : y1 + patch.shape[0], x0 : x1 + patch.shape[1]]
        float_region = (
            template_float[y0 : y1 + patch.shape[0], x0 : x1 + patch.shape[1]]
            if template_float is not None
            else None
        )
        local = _safe_match(template_region, patch, mask, template_float=float_region, binary_fast=binary_fast)
        if local is not None:
            response[y0 : y1 + 1, x0 : x1 + 1] = local
    return response


def _score_at_response(response: np.ndarray | None, x: int, y: int) -> float:
    if response is None or y < 0 or x < 0 or y >= response.shape[0] or x >= response.shape[1]:
        return 0.0
    value = float(response[y, x])
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _chamfer_response(
    template_distance: np.ndarray,
    unknown_edge: np.ndarray,
    scale: float,
    *,
    windows: list[tuple[int, int, int, int]] | None = None,
    clipped_distance: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return a dense Chamfer score so edge evidence participates in candidate generation."""
    scaled_edge = _scale_mask(unknown_edge, scale)
    if template_distance.shape[0] < scaled_edge.shape[0] or template_distance.shape[1] < scaled_edge.shape[1]:
        return None
    edge = (scaled_edge > 0).astype(np.float32)
    edge_count = int(edge.sum())
    if edge_count < 20:
        return None
    if clipped_distance is None:
        clipped_distance = np.clip(template_distance, 0.0, 30.0).astype(np.float32)
    if windows is None:
        distance_sum = _support_correlation(clipped_distance, edge)
    else:
        response_h = clipped_distance.shape[0] - edge.shape[0] + 1
        response_w = clipped_distance.shape[1] - edge.shape[1] + 1
        distance_sum = np.zeros((response_h, response_w), dtype=np.float32)
        for x0, y0, x1, y1 in windows:
            region = clipped_distance[y0 : y1 + edge.shape[0], x0 : x1 + edge.shape[1]]
            distance_sum[y0 : y1 + 1, x0 : x1 + 1] = _support_correlation(region, edge)
    mean_distance = np.maximum(0.0, distance_sum / float(edge_count))
    return np.exp(-mean_distance / 8.0).astype(np.float32)


def _orientation_masks(edge_mask: np.ndarray) -> list[np.ndarray]:
    """Split edge normals into four unoriented 45-degree bins."""
    binary = (edge_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return [np.zeros_like(edge_mask) for _ in range(4)]
    gx = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.mod(np.rint(angle / (np.pi / 4.0)).astype(np.int32), 4)
    return [np.where((binary > 0) & (bins == idx), 255, 0).astype(np.uint8) for idx in range(4)]


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    if values.size == 0:
        return 0.0
    values = values.astype(np.float32, copy=False).reshape(-1)
    trim = min(values.size // 2, int(values.size * max(0.0, min(0.45, trim_ratio))))
    if trim:
        keep = values.size-trim
        values = np.partition(values, keep-1)[:keep]
    return float(values.mean()) if values.size else 0.0


def _directional_distance_fields(edge_mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    masks = _orientation_masks(edge_mask)
    distances: list[np.ndarray | None] = []
    for mask in masks:
        if cv2.countNonZero(mask) < 4:
            distances.append(None)
        else:
            distances.append(cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3))
    return masks, distances


def _template_directional_distance_fields(
    template: ProcessedTemplate,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    local_fields = getattr(template, "_directional_field_views", None)
    if local_fields is not None:
        return local_fields
    cached = getattr(template, "_directional_distance_cache", None)
    if cached is None:
        cached = _directional_distance_fields(template.edge_mask)
        template._directional_distance_cache = cached
    return cached


def _semantic_consistency(
    *,
    template: ProcessedTemplate,
    observation: SegmentationResult,
    scale: float,
    top_left: tuple[int, int],
    valid_mask: np.ndarray,
    observed_label: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    """Compare complete template structure against known ROI evidence.

    ROI known-background pixels carry label 0. Template structure covering
    those pixels is therefore penalized; template blank space is neutral. The
    template itself has no known-background channel.
    """
    if observed_label is None:
        scaled_corridor = _scale_mask(observation.corridor_mask, scale) > 0
        scaled_room = _scale_mask(observation.room_mask, scale) > 0
        observed_label = np.where(scaled_room, 2, np.where(scaled_corridor, 1, 0))
    height, width = observed_label.shape
    x, y = top_left
    template_corridor = template.corridor_mask[y : y + height, x : x + width] > 0
    template_room = template.room_mask[y : y + height, x : x + width] > 0
    if template_corridor.shape != observed_label.shape or template_room.shape != observed_label.shape:
        return None, None
    valid = valid_mask > 0
    template_label = np.where(template_room, 2, np.where(template_corridor, 1, 0))
    observed_structure = (observed_label > 0) & valid
    template_structure = (template_label > 0) & valid
    observed_count = int(observed_structure.sum())
    template_count = int(template_structure.sum())
    if observed_count < 20:
        return None, None
    exact_structure = int(((observed_label == template_label) & observed_structure).sum())
    denominator = observed_count + template_count
    consistency = (2.0 * exact_structure / denominator) if denominator else None
    extra = int((template_structure & ~observed_structure).sum())
    extra_rate = extra / float(max(1, template_count))
    return consistency, extra_rate


def _directional_chamfer_review(
    *,
    template_edge_masks: list[np.ndarray],
    template_distance_fields: list[np.ndarray | None],
    observation_edge_masks: list[np.ndarray],
    observation_distance_fields: list[np.ndarray | None],
    valid_interior: np.ndarray,
    top_left: tuple[int, int],
    cfg: dict,
) -> tuple[float | None, float | None, float | None]:
    """Bidirectional, orientation-aware Chamfer inside the reliable viewport."""
    x, y = top_left
    height, width = valid_interior.shape
    max_distance = max(1.0, float(cfg.get("geometric_review_max_distance", 20.0)))
    trim_ratio = float(cfg.get("geometric_review_trim_ratio", 0.15))
    hit_distance = max(0.5, float(cfg.get("geometric_review_hit_distance", 3.0)))
    forward_values: list[np.ndarray] = []
    reverse_values: list[np.ndarray] = []
    forward_hits = 0
    forward_total = 0
    reverse_hits = 0
    reverse_total = 0
    for idx in range(4):
        obs_edge = observation_edge_masks[idx] > 0
        template_mask = template_edge_masks[idx][y : y + height, x : x + width] > 0
        if template_mask.shape != obs_edge.shape:
            return None, None, None
        template_distance = template_distance_fields[idx]
        if obs_edge.any():
            forward_total += int(obs_edge.sum())
            if template_distance is None:
                values = np.full(int(obs_edge.sum()), max_distance, dtype=np.float32)
            else:
                values = template_distance[y : y + height, x : x + width][obs_edge]
                values = np.minimum(values, max_distance)
            forward_values.append(values)
            forward_hits += int((values <= hit_distance).sum())
        reverse_points = template_mask & (valid_interior > 0)
        if reverse_points.any():
            reverse_total += int(reverse_points.sum())
            observation_distance = observation_distance_fields[idx]
            if observation_distance is None:
                values = np.full(int(reverse_points.sum()), max_distance, dtype=np.float32)
            else:
                values = np.minimum(observation_distance[reverse_points], max_distance)
            reverse_values.append(values)
            reverse_hits += int((values <= hit_distance).sum())
    if forward_total < 20 or reverse_total < 20:
        return None, None, None
    forward = _trimmed_mean(np.concatenate(forward_values), trim_ratio)
    reverse = _trimmed_mean(np.concatenate(reverse_values), trim_ratio)
    precision = forward_hits / float(forward_total)
    recall = reverse_hits / float(reverse_total)
    f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
    alpha = max(0.0, min(1.0, float(cfg.get("geometric_review_forward_weight", 0.60))))
    score = f1 * float(np.exp(-(alpha * forward + (1.0 - alpha) * reverse) / max_distance))
    return score, precision, recall


def _build_geometric_review_context(
    template: ProcessedTemplate,
    observation: SegmentationResult,
    scale: float,
    cfg: dict,
) -> dict[str, object]:
    valid = observation.valid_mask if observation.valid_mask is not None else observation.visible_mask
    scaled_valid = _scale_mask(valid, scale)
    scaled_structure = _scale_mask(observation.map_mask, scale)
    support_radius = max(0, int(cfg.get("geometric_review_edge_support_radius", 4)))
    if support_radius:
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_radius * 2 + 1, support_radius * 2 + 1),
        )
        edge_support = cv2.dilate(scaled_structure, support_kernel)
    else:
        edge_support = scaled_structure
    erosion = max(0, int(cfg.get("geometric_review_valid_erosion", 3)))
    if erosion:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1))
        valid_interior = cv2.erode(edge_support, kernel)
    else:
        valid_interior = edge_support
    scaled_edge = _scale_mask(observation.edge_mask, scale)
    scaled_edge = cv2.bitwise_and(scaled_edge, scaled_valid)
    template_edge_masks, template_distance_fields = _template_directional_distance_fields(template)
    observation_edge_masks, observation_distance_fields = _directional_distance_fields(scaled_edge)
    return {
        "observed_label": np.where(_scale_mask(observation.room_mask, scale) > 0, 2,
                                   np.where(_scale_mask(observation.corridor_mask, scale) > 0, 1, 0)),
        "valid": scaled_valid,
        "valid_interior": valid_interior,
        "template_edge_masks": template_edge_masks,
        "template_distance_fields": template_distance_fields,
        "observation_edge_masks": observation_edge_masks,
        "observation_distance_fields": observation_distance_fields,
    }


def _geometric_review(
    *,
    template: ProcessedTemplate,
    observation: SegmentationResult,
    scale: float,
    top_left: tuple[int, int],
    context: dict[str, object],
    cfg: dict,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    semantic, extra_rate = _semantic_consistency(
        template=template,
        observation=observation,
        scale=scale,
        top_left=top_left,
        valid_mask=context["valid"],  # type: ignore[arg-type]
        observed_label=context.get("observed_label"),
    )
    directional, precision, recall = _directional_chamfer_review(
        template_edge_masks=context["template_edge_masks"],  # type: ignore[arg-type]
        template_distance_fields=context["template_distance_fields"],  # type: ignore[arg-type]
        observation_edge_masks=context["observation_edge_masks"],  # type: ignore[arg-type]
        observation_distance_fields=context["observation_distance_fields"],  # type: ignore[arg-type]
        valid_interior=context["valid_interior"],  # type: ignore[arg-type]
        top_left=top_left,
        cfg=cfg,
    )
    if semantic is None and directional is None:
        return None, semantic, extra_rate, directional, precision, recall
    semantic_weight = max(0.0, float(cfg.get("geometric_review_semantic_weight", 0.55)))
    directional_weight = max(0.0, float(cfg.get("geometric_review_directional_weight", 0.45)))
    total = semantic_weight + directional_weight
    review = 0.0
    if semantic is not None and semantic_weight:
        review += semantic * semantic_weight
    if directional is not None and directional_weight:
        review += directional * directional_weight
    if total:
        review /= total
    return review, semantic, extra_rate, directional, precision, recall


def _pad_template_for_matching(template: ProcessedTemplate, padding: int) -> ProcessedTemplate:
    """Add a search-only border without changing reported template coordinates."""
    padding = max(0, int(padding))
    if padding == 0:
        return template
    # Cache lifetime follows the immutable source template. A global id-keyed
    # cache could return another layout after Python reuses a released id.
    cache = getattr(template, "_padded_matching_cache", None)
    if cache is None:
        cache = {}
        template._padded_matching_cache = cache
    cached = cache.get(padding)
    if cached is not None:
        return cached

    def pad(image: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    edge_mask = pad(template.edge_mask)
    distance_transform = cv2.distanceTransform(cv2.bitwise_not(edge_mask), cv2.DIST_L2, 3)
    corridor_mask = pad(template.corridor_mask)
    room_mask = pad(template.room_mask)
    padded = ProcessedTemplate(
        map_id=template.map_id,
        image=pad(template.image),
        corridor_mask=corridor_mask,
        room_mask=room_mask,
        map_mask=pad(template.map_mask),
        edge_mask=edge_mask,
        stair_mask=pad(template.stair_mask),
        distance_transform=distance_transform,
        crop_offset=template.crop_offset,
        raw_path=template.raw_path,
        corridor_float=corridor_mask.astype(np.float32) / 255.0,
        room_float=room_mask.astype(np.float32) / 255.0,
    )
    cache[padding] = padded
    return padded


def warm_matching_template(template: ProcessedTemplate, padding: int = 24) -> None:
    """Move immutable distance-field work into the existing template preload."""
    padded = _pad_template_for_matching(template, padding)
    _template_directional_distance_fields(padded)
    if getattr(padded, "_clipped_chamfer_distance", None) is None:
        padded._clipped_chamfer_distance = np.clip(padded.distance_transform, 0, 30).astype(np.float32)


def review_pose_candidates(template: ProcessedTemplate, observation: SegmentationResult,
                           poses: list[tuple[tuple[float, float], float]], cfg: dict) -> list[MatchCandidate]:
    """Score a short pose list with the same dense semantics as the template matcher."""
    padding = int(cfg.get("candidate_border_padding", 24))
    padded = _pad_template_for_matching(template, padding)
    contexts = {}
    candidates = []
    strength = float(cfg.get("geometric_review_strength", 0.60))
    weights = np.float64([cfg.get("corridor_weight", .48), cfg.get("room_weight", .32),
                          cfg.get("edge_weight", .20)])
    weights /= weights.sum()
    for position, scale in poses:
        x, y = int(round(position[0])) + padding, int(round(position[1])) + padding
        key = round(float(scale), 6)
        if key not in contexts:
            masks = _scale_masks([observation.corridor_mask, observation.room_mask,
                                  observation.edge_mask], float(scale))
            contexts[key] = (masks, _build_geometric_review_context(padded, observation, float(scale), cfg))
        masks, context = contexts[key]
        h, w = masks[0].shape
        if x < 0 or y < 0 or x + w > padded.map_mask.shape[1] or y + h > padded.map_mask.shape[0]:
            continue
        scores = []
        for expected, observed in zip((padded.corridor_mask, padded.room_mask), masks[:2]):
            selected = observed > 0
            count = int(np.count_nonzero(selected))
            overlap = np.count_nonzero(expected[y:y+h, x:x+w][selected])
            scores.append(float(np.sqrt(overlap / count)) if count >= 20 else 0.0)
        edges = masks[2] > 0
        values = padded.distance_transform[y:y+h, x:x+w][edges]
        if cfg.get("proposal_requires_strong_geometry", False):
            observed_count = sum(cv2.countNonZero(mask) for mask in masks[:2])
            correct_count = sum(np.count_nonzero(expected[y:y+h, x:x+w][observed > 0])
                                for expected, observed in zip((padded.corridor_mask, padded.room_mask), masks[:2]))
            if len(values) < 40 or np.median(values) > 8 or correct_count / max(1, observed_count) < .82:
                continue
        scores.append(float(np.exp(-np.minimum(values, 30).mean() / 8)) if len(values) >= 20 else 0.0)
        stage1 = float(np.dot(weights, scores))
        review, semantic, extra, directional, precision, recall = _geometric_review(
            template=padded, observation=observation, scale=float(scale),
            top_left=(x, y), context=context, cfg=cfg)
        score = stage1 if review is None else (1-strength)*stage1 + strength*review
        candidates.append(MatchCandidate((x-padding, y-padding), float(scale), score,
                                          *scores, 0.0, semantic, extra, directional,
                                          precision, recall, review, stage1))
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _top_locations(response: np.ndarray, top_k: int, suppress_radius: int) -> list[tuple[float, tuple[int, int]]]:
    local = np.nan_to_num(response, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
    found: list[tuple[float, tuple[int, int]]] = []
    for _ in range(max(top_k * 3, top_k)):
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(local)
        if max_val < 0:
            break
        found.append((float(max_val), (int(max_loc[0]), int(max_loc[1]))))
        x, y = max_loc
        r = max(8, suppress_radius)
        local[max(0, y - r) : min(local.shape[0], y + r + 1), max(0, x - r) : min(local.shape[1], x + r + 1)] = -1.0
        if len(found) >= top_k:
            break
    return found


def _select_geometric_review_seeds(
    seeds: list[tuple[float, float, tuple[int, int], dict[str, object]]],
    *,
    review_count: int,
    spatial_separation: int = 24,
) -> list[tuple[float, float, tuple[int, int], dict[str, object]]]:
    """Keep only distinct high-confidence stage-one peaks for costly review."""
    selected: list[tuple[float, float, tuple[int, int], dict[str, object]]] = []
    for seed in sorted(seeds, key=lambda item: item[0], reverse=True):
        _score, scale, top_left, _responses = seed
        if any(
            abs(scale - old_scale) < 1e-6
            and abs(top_left[0] - old_top_left[0]) + abs(top_left[1] - old_top_left[1]) < spatial_separation
            for _old_score, old_scale, old_top_left, _old_responses in selected
        ):
            continue
        selected.append(seed)
        if len(selected) >= review_count:
            break
    return selected


def _stage1_margin_from_seeds(
    seeds: list[tuple[float, float, tuple[int, int], dict[str, object]]],
    *,
    spatial_separation: int = 24,
) -> float | None:
    """Return the stage-one gap between the two distinct best locations."""
    distinct = _select_geometric_review_seeds(
        sorted(seeds, key=lambda item: item[0], reverse=True),
        review_count=2,
        spatial_separation=spatial_separation,
    )
    if len(distinct) < 2:
        return None
    return max(0.0, float(distinct[0][0] - distinct[1][0]))


def _allowed_player_points(cfg: dict) -> list[tuple[float, float]]:
    value = cfg.get("allowed_player_template_points")
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if (
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ):
            points.append((float(item[0]), float(item[1])))
    return points


def _restrict_response_to_player_points(
    response: np.ndarray,
    *,
    player_roi: tuple[float, float] | None,
    scale: float,
    allowed_points: list[tuple[float, float]],
    radius: float,
) -> np.ndarray:
    if player_roi is None or not allowed_points:
        return response
    radius = max(1.0, float(radius))
    radius2 = radius * radius
    local = np.full(response.shape, -1.0, dtype=response.dtype)
    player_x = float(player_roi[0]) * float(scale)
    player_y = float(player_roi[1]) * float(scale)
    height, width = response.shape[:2]
    for point_x, point_y in allowed_points:
        center_x = float(point_x) - player_x
        center_y = float(point_y) - player_y
        x0 = max(0, int(np.floor(center_x - radius)))
        x1 = min(width - 1, int(np.ceil(center_x + radius)))
        y0 = max(0, int(np.floor(center_y - radius)))
        y1 = min(height - 1, int(np.ceil(center_y + radius)))
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = ((xx - center_x) * (xx - center_x) + (yy - center_y) * (yy - center_y)) <= radius2
        patch = local[y0 : y1 + 1, x0 : x1 + 1]
        source = response[y0 : y1 + 1, x0 : x1 + 1]
        patch[mask] = source[mask]
    return local


def _merge_response_windows(
    windows: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for candidate in windows:
        x0, y0, x1, y1 = candidate
        changed = True
        while changed:
            changed = False
            remaining: list[tuple[int, int, int, int]] = []
            for old_x0, old_y0, old_x1, old_y1 in merged:
                if x0 <= old_x1 + 1 and old_x0 <= x1 + 1 and y0 <= old_y1 + 1 and old_y0 <= y1 + 1:
                    x0 = min(x0, old_x0)
                    y0 = min(y0, old_y0)
                    x1 = max(x1, old_x1)
                    y1 = max(y1, old_y1)
                    changed = True
                else:
                    remaining.append((old_x0, old_y0, old_x1, old_y1))
            merged = remaining
        merged.append((x0, y0, x1, y1))
    return merged


def _allowed_response_windows(
    response_shape: tuple[int, int],
    *,
    player_roi: tuple[float, float] | None,
    scale: float,
    allowed_points: list[tuple[float, float]],
    radius: float,
) -> list[tuple[int, int, int, int]] | None:
    """Return square bounds enclosing the existing circular player constraint."""
    if player_roi is None or not allowed_points:
        return None
    height, width = response_shape
    radius = max(1.0, float(radius))
    player_x = float(player_roi[0]) * float(scale)
    player_y = float(player_roi[1]) * float(scale)
    windows: list[tuple[int, int, int, int]] = []
    for point_x, point_y in allowed_points:
        center_x = float(point_x) - player_x
        center_y = float(point_y) - player_y
        x0 = max(0, int(np.floor(center_x - radius)))
        x1 = min(width - 1, int(np.ceil(center_x + radius)))
        y0 = max(0, int(np.floor(center_y - radius)))
        y1 = min(height - 1, int(np.ceil(center_y + radius)))
        if x0 <= x1 and y0 <= y1:
            windows.append((x0, y0, x1, y1))
    return _merge_response_windows(windows)


def _candidate_seed_response(
    template: ProcessedTemplate,
    obs: SegmentationResult,
    scale: float,
    *,
    scaled_corridor: np.ndarray | None = None,
    scaled_room: np.ndarray | None = None,
    scaled_edge: np.ndarray | None = None,
    parallel_channels: bool = True,
    stair_seed_weight: float = 0.0,
    stair_min_score: float = 0.0,
    scaled_stairs: np.ndarray | None = None,
    response_windows: list[tuple[int, int, int, int]] | None = None,
    binary_fast: bool = False,
) -> tuple[np.ndarray | None, dict[str, np.ndarray | None]]:
    if scaled_corridor is None:
        scaled_corridor = _scale_mask(obs.corridor_mask, scale)
    if scaled_room is None:
        scaled_room = _scale_mask(obs.room_mask, scale)
    if scaled_edge is None:
        scaled_edge = _scale_mask(obs.edge_mask, scale)
    # 只用当前已观测到的语义结构作为 match mask，避免迷雾/背景参与匹配。
    jobs = {}
    if cv2.countNonZero(scaled_corridor) >= 20:
        jobs["corridor"] = lambda: _safe_match_in_windows(
            template.corridor_mask,
            scaled_corridor,
            response_windows,
            scaled_corridor,
            template_float=template.corridor_float,
            binary_fast=binary_fast,
        )
    if cv2.countNonZero(scaled_room) >= 20:
        jobs["room"] = lambda: _safe_match_in_windows(
            template.room_mask,
            scaled_room,
            response_windows,
            scaled_room,
            template_float=template.room_float,
            binary_fast=binary_fast,
        )
    clipped = None
    if binary_fast:
        clipped = getattr(template, "_clipped_chamfer_distance", None)
        if clipped is None:
            clipped = np.clip(template.distance_transform, 0.0, 30.0).astype(np.float32)
            template._clipped_chamfer_distance = clipped
    jobs["edge"] = lambda: _chamfer_response(
        template.distance_transform,
        scaled_edge,
        1.0,
        windows=response_windows,
        clipped_distance=clipped,
    )
    if parallel_channels and len(jobs) > 1:
        futures = {name: _MATCH_CHANNEL_EXECUTOR.submit(job) for name, job in jobs.items()}
        matched = {name: future.result() for name, future in futures.items()}
    else:
        matched = {name: job() for name, job in jobs.items()}
    stair_resp = None
    if stair_seed_weight > 0 and cv2.countNonZero(obs.stair_mask) >= 12:
        if scaled_stairs is None:
            scaled_stairs = _scale_mask(obs.stair_mask, scale)
        stair_resp = _safe_match(template.stair_mask, scaled_stairs)
    patch_h, patch_w = scaled_corridor.shape
    response_h = template.map_mask.shape[0] - patch_h + 1
    response_w = template.map_mask.shape[1] - patch_w + 1
    if response_h <= 0 or response_w <= 0:
        return None, {}
    # Candidate ranking is assembled later from the configured fixed weights.
    # This array supplies only the response shape; absent semantic channels do
    # not transfer their weight to the channel that happened to be detected.
    seed = np.zeros((response_h, response_w), dtype=np.float32)
    if stair_resp is not None:
        weight = max(0.0, min(0.35, float(stair_seed_weight)))
        # A staircase can be outside the current map viewport. It is positive
        # evidence when present, but its absence must not suppress a good map match.
        stair_evidence = np.maximum(0.0, stair_resp - float(stair_min_score))
        seed = np.minimum(1.0, seed + weight * stair_evidence)
    return seed, {
        "corridor": matched.get("corridor"),
        "room": matched.get("room"),
        "edge": matched.get("edge"),
        "stairs": stair_resp,
    }


def _configured_scales(cfg: dict) -> np.ndarray:
    """Return either explicit scale hypotheses or the configured regular grid."""
    explicit = cfg.get("scale_values")
    if isinstance(explicit, (list, tuple)):
        values: list[float] = []
        for raw in explicit:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0 and not any(abs(value - old) < 1e-7 for old in values):
                values.append(value)
        if values:
            return np.asarray(values, dtype=np.float32)
    fixed_scale = cfg.get("fixed_scale")
    if fixed_scale is not None:
        return np.array([float(fixed_scale)], dtype=np.float32)
    scale_min = float(cfg.get("scale_min", 0.85))
    scale_max = float(cfg.get("scale_max", 1.15))
    scale_step = max(1e-4, float(cfg.get("scale_step", 0.05)))
    return np.arange(scale_min, scale_max + 1e-9, scale_step, dtype=np.float32)


def match_single_frame(
    *,
    template: ProcessedTemplate,
    observation: SegmentationResult,
    cfg: dict,
    player_roi: tuple[float, float] | None,
) -> LocalizationResult:
    top_k = int(cfg.get("top_k", 8))
    candidate_pool_size = max(top_k, int(cfg.get("candidate_pool_size", top_k)))
    candidate_border_padding = max(0, int(cfg.get("candidate_border_padding", 0)))
    geometric_review_enabled = bool(cfg.get("geometric_review_enabled", False))
    review_skip_margin = max(0.0, float(cfg.get("geometric_review_skip_if_stage1_margin_at_least", 0.0)))
    configured_review_count = int(cfg.get("geometric_review_candidate_count", top_k))
    geometric_review_candidate_count = (
        max(top_k, configured_review_count) if configured_review_count > 0 else None
    )
    geometric_review_strength = max(0.0, min(1.0, float(cfg.get("geometric_review_strength", 0.50))))
    matching_template = _pad_template_for_matching(template, candidate_border_padding)
    weights = {
        "corridor": float(cfg.get("corridor_weight", 0.45)),
        "room": float(cfg.get("room_weight", 0.30)),
        "edge": float(cfg.get("edge_weight", 0.20)),
        "stairs": float(cfg.get("stair_weight", 0.0)),
    }
    # Keep configured weights fixed for every observation. Missing evidence
    # contributes zero instead of redistributing its weight to another channel.
    base_total = max(1e-6, sum(weight for name, weight in weights.items() if name != "stairs"))
    stair_min_score = max(0.0, min(0.95, float(cfg.get("stair_min_score", 0.10))))
    scales = _configured_scales(cfg)
    allowed_points = _allowed_player_points(cfg)
    # Bindings are stored in full-template coordinates. Match responses use the
    # processed template crop, so shift the allowed centers into that space.
    crop_x, crop_y = template.crop_offset
    allowed_points = [
        (x - float(crop_x) + candidate_border_padding, y - float(crop_y) + candidate_border_padding)
        for x, y in allowed_points
    ]
    allowed_radius = float(cfg.get("allowed_player_template_radius", 160.0))
    search_origin = (0, 0)
    if (cfg.get("first_sparse_optimized_enabled", False)
            and cfg.get("pre_match_search_windows_enabled", True)
            and allowed_points and player_roi is not None and len(scales)):
        # Windowed correlation formerly still allocated and scanned full-map
        # response arrays. Crop their domain too, preserving every allowed pose.
        h, w = observation.map_mask.shape
        px, py = player_roi
        left = min(ax - px * float(s) - allowed_radius for ax, _ in allowed_points for s in scales)
        top = min(ay - py * float(s) - allowed_radius for _, ay in allowed_points for s in scales)
        right = max(ax + (w - px) * float(s) + allowed_radius for ax, _ in allowed_points for s in scales)
        bottom = max(ay + (h - py) * float(s) + allowed_radius for _, ay in allowed_points for s in scales)
        x0, y0 = max(0, int(np.floor(left)) - 2), max(0, int(np.floor(top)) - 2)
        x1 = min(matching_template.map_mask.shape[1], int(np.ceil(right)) + 3)
        y1 = min(matching_template.map_mask.shape[0], int(np.ceil(bottom)) + 3)
        if x1 > x0 and y1 > y0:
            region = np.s_[y0:y1, x0:x1]
            masks, distances = _template_directional_distance_fields(matching_template)
            values = {name: getattr(matching_template, name)[region] for name in (
                "image", "corridor_mask", "room_mask", "map_mask", "edge_mask",
                "stair_mask", "distance_transform", "corridor_float", "room_float")
                if getattr(matching_template, name) is not None}
            matching_template = replace(matching_template, **values)
            matching_template._directional_field_views = (
                [mask[region] for mask in masks],
                [distance[region] if distance is not None else None for distance in distances],
            )
            allowed_points = [(ax - x0, ay - y0) for ax, ay in allowed_points]
            search_origin = (x0, y0)
    seeds: list[tuple[float, float, tuple[int, int], dict[str, object]]] = []
    for scale in scales:
        scale_value = float(scale)
        mask_names = ["corridor", "room", "edge"]
        masks = [observation.corridor_mask, observation.room_mask, observation.edge_mask]
        if weights["stairs"] > 0:
            mask_names.append("stairs")
            masks.append(observation.stair_mask)
        scaled_masks = dict(zip(mask_names, _scale_masks(masks, scale_value)))
        scaled_height, scaled_width = scaled_masks["corridor"].shape
        response_shape = (
            matching_template.map_mask.shape[0] - scaled_height + 1,
            matching_template.map_mask.shape[1] - scaled_width + 1,
        )
        response_windows = None
        if bool(cfg.get("pre_match_search_windows_enabled", True)) and response_shape[0] > 0 and response_shape[1] > 0:
            response_windows = _allowed_response_windows(
                response_shape,
                player_roi=player_roi,
                scale=scale_value,
                allowed_points=allowed_points,
                radius=allowed_radius,
            )
        semantic_seed, responses = _candidate_seed_response(
            matching_template,
            observation,
            scale_value,
            scaled_corridor=scaled_masks["corridor"],
            scaled_room=scaled_masks["room"],
            scaled_edge=scaled_masks["edge"],
            parallel_channels=bool(cfg.get("parallel_match_channels_enabled", True)),
            stair_seed_weight=weights["stairs"],
            stair_min_score=stair_min_score,
            scaled_stairs=scaled_masks.get("stairs"),
            response_windows=response_windows,
            binary_fast=bool(cfg.get("first_sparse_optimized_enabled", False)),
        )
        if semantic_seed is None:
            continue
        combined = np.zeros_like(semantic_seed, dtype=np.float32)
        for name in ("corridor", "room", "edge"):
            component = responses.get(name)
            if component is not None and weights[name] > 0:
                combined += float(weights[name] / base_total) * component
        stair_response = responses.get("stairs")
        if stair_response is not None and weights["stairs"] > 0:
            stair_evidence = np.maximum(0.0, stair_response - stair_min_score) / max(1e-6, 1.0 - stair_min_score)
            combined += float(weights["stairs"]) * stair_evidence
        combined = np.clip(combined, 0.0, 1.0)
        combined = _restrict_response_to_player_points(
            combined,
            player_roi=player_roi,
            scale=scale_value,
            allowed_points=allowed_points,
            radius=allowed_radius,
        )
        patch_w = int(round(observation.map_mask.shape[1] * scale))
        patch_h = int(round(observation.map_mask.shape[0] * scale))
        suppress = max(12, min(patch_w, patch_h) // 6)
        responses["combined"] = combined
        for seed_score, loc in _top_locations(combined, candidate_pool_size, suppress):
            seeds.append((seed_score, scale_value, loc, responses))
    stage1_margin = None
    if geometric_review_enabled and review_skip_margin > 0.0:
        stage1_margin = _stage1_margin_from_seeds(seeds)
        if stage1_margin is not None and stage1_margin >= review_skip_margin:
            geometric_review_enabled = False
    if geometric_review_enabled and geometric_review_candidate_count is not None:
        seeds = _select_geometric_review_seeds(
            seeds,
            review_count=geometric_review_candidate_count,
        )
    if geometric_review_enabled:
        for _seed, scale, _top_left, responses in seeds:
            if "geometric_context" not in responses:
                responses["geometric_context"] = _build_geometric_review_context(
                    matching_template,
                    observation,
                    scale,
                    cfg,
                )
    candidates: list[MatchCandidate] = []
    for _seed, scale, top_left, responses in seeds:
        x, y = top_left
        corridor_score = _score_at_response(responses.get("corridor"), x, y)
        room_score = _score_at_response(responses.get("room"), x, y)
        edge = _score_at_response(responses.get("edge"), x, y)
        stair_score = _score_at_response(responses.get("stairs"), x, y)
        stage1_score = _score_at_response(responses.get("combined"), x, y)  # type: ignore[arg-type]
        review_score = None
        semantic_consistency = None
        semantic_extra_structure_rate = None
        directional_chamfer_score = None
        edge_precision = None
        edge_recall = None
        if geometric_review_enabled:
            context = responses.get("geometric_context")
            if isinstance(context, dict):
                (
                    review_score,
                    semantic_consistency,
                    semantic_extra_structure_rate,
                    directional_chamfer_score,
                    edge_precision,
                    edge_recall,
                ) = _geometric_review(
                    template=matching_template,
                    observation=observation,
                    scale=scale,
                    top_left=top_left,
                    context=context,
                    cfg=cfg,
                )
        score = stage1_score if review_score is None else (
            (1.0 - geometric_review_strength) * stage1_score + geometric_review_strength * review_score
        )
        candidates.append(
            MatchCandidate(
                top_left=(top_left[0] - candidate_border_padding + search_origin[0],
                          top_left[1] - candidate_border_padding + search_origin[1]),
                scale=scale,
                score=float(score),
                corridor_score=corridor_score,
                room_score=room_score,
                edge_score=edge,
                stair_score=stair_score,
                semantic_consistency=semantic_consistency,
                semantic_extra_structure_rate=semantic_extra_structure_rate,
                directional_chamfer_score=directional_chamfer_score,
                edge_precision=edge_precision,
                edge_recall=edge_recall,
                review_score=review_score,
                stage1_score=stage1_score,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    unique: list[MatchCandidate] = []
    for candidate in candidates:
        if any(abs(candidate.scale - old.scale) < 1e-6 and abs(candidate.top_left[0] - old.top_left[0]) + abs(candidate.top_left[1] - old.top_left[1]) < 24 for old in unique):
            continue
        unique.append(candidate)
        if len(unique) >= top_k:
            break
    if not unique:
        return LocalizationResult(template.map_id, None, None, 0.0, True, [], player_roi, None, None, observation.map_mask.shape)
    best = unique[0]
    second_score = unique[1].score if len(unique) > 1 else 0.0
    confidence = float(max(0.0, best.score - second_score))
    low_conf = confidence < float(cfg.get("confidence_threshold", 0.08))
    player_template = None
    player_template_crop = None
    if player_roi is not None:
        player_template_crop = map_roi_point_to_template(player_roi, top_left=best.top_left, scale=best.scale, template_crop_offset=(0, 0))
        player_template = map_roi_point_to_template(player_roi, top_left=best.top_left, scale=best.scale, template_crop_offset=template.crop_offset)
    elif bool(cfg.get("player_is_map_center", False)):
        center = (observation.map_mask.shape[1] / 2.0, observation.map_mask.shape[0] / 2.0)
        player_template_crop = map_roi_point_to_template(center, top_left=best.top_left, scale=best.scale, template_crop_offset=(0, 0))
        player_template = map_roi_point_to_template(center, top_left=best.top_left, scale=best.scale, template_crop_offset=template.crop_offset)
    return LocalizationResult(
        template.map_id,
        best.top_left,
        best.scale,
        confidence,
        low_conf,
        unique,
        player_roi,
        player_template,
        player_template_crop,
        observation.map_mask.shape,
        geometric_review_applied=geometric_review_enabled,
        geometric_review_stage1_margin=stage1_margin,
    )
