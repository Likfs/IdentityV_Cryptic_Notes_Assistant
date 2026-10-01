from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np


from src.registration.multiscale_structure.gpu.core.color_segmenter import (
    SegmentationResult,
    add_known_background_evidence,
    segment_map,
)
from src.registration.multiscale_structure.gpu.core.cpu_pyramid_matcher import (
    CpuPyramidMatchResult,
    CpuPyramidTiming,
    build_cpu_pyramid_template,
    match_single_frame_cpu_pyramid,
)
from src.registration.multiscale_structure.gpu.core.local_structure_index import (
    LocalStructureIndex,
    build_or_load_local_structure_index,
)
from src.registration.multiscale_structure.gpu.core.local_structure_refiner import refine_local_structure
from src.registration.multiscale_structure.gpu.core.geometric_pair_index import GeometricPairIndex
from src.registration.multiscale_structure.gpu.core.masked_matcher import (
    LocalizationResult, MatchCandidate, match_single_frame, warm_matching_template,
)
from src.registration.multiscale_structure.gpu.core.roi_extractor import component_content_bbox
from src.registration.multiscale_structure.gpu.core.pose_evidence import build_regional_evidence, analyze_pose, warm_regional_template
from src.registration.multiscale_structure.gpu.core.sparse_pose_review import observation_anchor, review_sparse_pose
from src.registration.multiscale_structure.gpu.core.credible_geometry import screenshot_evidence, review_geometry, pose_matrix
from src.registration.multiscale_structure.gpu.core.fog_evidence import filter_corridor_fog
from src.registration.multiscale_structure.gpu.core.structural_features import extract_structural_features
from src.registration.multiscale_structure.gpu.core.template_preprocessor import ProcessedTemplate, preprocess_template
from src.registration.multiscale_structure.gpu.core.utils import imread_unicode
from src.registration.multiscale_structure.gpu.registration.temporal_evidence import TemporalEvidence


MAX_EDGE_MEDIAN_ERROR_PX = 10.0


def register_screenshot(screenshot, prepared, prior_screenshot_to_clean=None,
                        temporal_evidence=None, soft_pose_hint=None):
    from .recovery import recover, recover_unfiltered_search
    result = _register_screenshot_once(screenshot,prepared,prior_screenshot_to_clean,temporal_evidence,soft_pose_hint)
    try:
        result = recover(screenshot,prepared,result,_register_screenshot_once,
                         prior_screenshot_to_clean,temporal_evidence,soft_pose_hint)
        return recover_unfiltered_search(screenshot, prepared, result, _register_screenshot_once,
                                         prior_screenshot_to_clean, temporal_evidence, soft_pose_hint)
    except Exception as exc:
        result.validation['recovery_error'] = str(exc)
        return result


def _background_only_history_conflict(validation: dict, details: dict) -> bool:
    """Permit route review when only uncertain current-background evidence disagrees."""
    compared = details.get("compared_pixels", 0)
    edge = validation.get("edge_median_error_px")
    return bool(
        compared >= 500
        and details.get("conflict_rate") is not None
        and details["conflict_rate"] > 0.20
        and details.get("hard_conflict_pixels", compared) / compared <= 0.05
        and validation.get("observed_structure_pixels", 0) >= 5000
        and validation.get("structure_precision", 0.0) >= 0.95
        and validation.get("semantic_precision", 0.0) >= 0.88
        and edge is not None and edge <= 5.0
    )


def runtime_registration_config() -> dict:
    return {
        "corridor_fog_filter_enabled": True,
        "corridor_structure_enabled": True,
        "registration_recovery_enabled": True,
        "scale_boundary_recovery_enabled": True,
        "reliable_wall_recovery_enabled": True,
        "local_projection_enabled": True,
        "first_sparse_optimized_enabled": True,
        "first_sparse_point_refinement_enabled": True,
        # Candidate replacement remains experimental: the 63-template known-pose
        # suite found additional incorrect accepts despite strong local scores.
        "first_sparse_candidate_pool_enabled": False,
        "first_sparse_geometry_enabled": False,
        "regional_pose_diagnostics_enabled": True,
        "regional_pose_guard_enabled": True,
        "sparse_pose_review_enabled": True,
        "credible_sparse_geometry_enabled": True,
        # Strict replacement remains opt-in until it preserves known-pose recall.
        "credible_sparse_geometry_strict": False,
        "staged_fine_search_enabled": False,
        "pending_pan_enabled": False,
        "scale_min": 0.35,
        "scale_max": 1.25,
        "scale_step": 0.01,
        "top_k": 12,
        "candidate_pool_size": 72,
        "candidate_border_padding": 24,
        "corridor_weight": 0.48,
        "room_weight": 0.32,
        "edge_weight": 0.20,
        "stair_weight": 0.0,
        "parallel_match_channels_enabled": True,
        "pre_match_search_windows_enabled": False,
        "confidence_threshold": 0.035,
        "geometric_review_enabled": True,
        "geometric_review_candidate_count": 48,
        "geometric_review_strength": 0.60,
        "geometric_review_semantic_weight": 0.55,
        "geometric_review_directional_weight": 0.45,
        "geometric_review_forward_weight": 0.60,
        "geometric_review_trim_ratio": 0.15,
        "geometric_review_hit_distance": 3.0,
        "geometric_review_max_distance": 20.0,
        "geometric_review_valid_erosion": 2,
        "geometric_review_edge_support_radius": 4,
        "cpu_pyramid_factor": 0.4,
        "cpu_pyramid_coarse_scale_step": 0.05,
        "cpu_pyramid_coarse_top_k": 4,
        "cpu_pyramid_coarse_candidate_pool_size": 4,
        "cpu_pyramid_fine_scale_radius": 0.05,
        "cpu_pyramid_fine_scale_step": 0.01,
        "cpu_pyramid_fine_position_radius": 100.0,
        "cpu_pyramid_fine_candidate_pool_size": 8,
        "cpu_pyramid_fine_review_candidate_count": 16,
        "local_structure_index_enabled": True,
        "local_structure_index_min_score": 0.78,
        "local_structure_index_coarse_min_score": 0.74,
        "local_structure_index_refine_scale_step": 0.01,
        "local_structure_index_refine_position_radius": 12.0,
        "local_structure_direct_refine_enabled": True,
        "local_structure_index_max_background_conflict": 0.15,
        "local_segmentation_static_ignore_crop_enabled": False,
        "preprocess": {
            "crop_padding": 10,
            "min_map_area": 300,
            "edge_canny1": 45,
            "edge_canny2": 135,
            "morph_kernel": 3,
            "known_background_max_distance": 16,
            "clean_template_semantic_segmentation": True,
            "clean_template_min_component_area": 80,
            "clean_template_max_hole_area": 320,
            "stairs": {"enabled": False},
        },
        "colors": {
            "corridor_hsv": [[95, 10, 72, 128, 92, 215]],
            "room_hsv": [[6, 40, 45, 28, 230, 190]],
            "visible_s_min": 12,
            "visible_v_min": 48,
            "dark_background_v_max": 45,
            "known_background_v_max": 60,
        },
    }


@dataclass
class PreparedRegistrationTemplate:
    layout: str
    floor: str
    path: Path
    raw_image: np.ndarray
    processed: ProcessedTemplate
    pyramid: ProcessedTemplate
    local_structure_index: LocalStructureIndex
    config: dict
    registration_scale: float
    preprocess_ms: float
    pyramid_ms: float
    local_structure_index_ms: float


@dataclass
class RuntimeRegistrationResult:
    accepted: bool
    rejection_reasons: list[str]
    clean_to_screen: np.ndarray | None
    screenshot_to_clean: np.ndarray | None
    observation_bbox: tuple[int, int, int, int] | None
    match: CpuPyramidMatchResult | None
    validation: dict
    timings: dict
    observation_image: np.ndarray | None
    observation_mask: np.ndarray | None
    alignment_preview: np.ndarray | None
    observation_segmentation: SegmentationResult | None = None
    source_segmentation: SegmentationResult | None = None
    search_segmentation: SegmentationResult | None = None

    def diagnostics(self) -> dict:
        matched = self.match.result.to_dict() if self.match is not None else None
        return {
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
            "clean_to_screen": self.clean_to_screen.tolist() if self.clean_to_screen is not None else None,
            "screenshot_to_clean": (
                self.screenshot_to_clean.tolist() if self.screenshot_to_clean is not None else None
            ),
            "observation_bbox": list(self.observation_bbox) if self.observation_bbox else None,
            "match": matched,
            "validation": self.validation,
            "timings_ms": self.timings,
        }


@dataclass(frozen=True)
class SoftPoseHint:
    screenshot_to_clean: np.ndarray
    observation_bbox: tuple[int, int, int, int]
    observation_mask: np.ndarray


def validate_route_overlay(
    registration: RuntimeRegistrationResult,
    layer_bgra: np.ndarray,
    structure_mask: np.ndarray | None = None,
) -> dict:
    """Check the projected route geometry against the map already visible on screen."""
    if registration.observation_bbox is None or registration.observation_mask is None:
        return {"accepted": False, "rejection_reasons": ["缺少可见地图结构，无法验证路线投影"]}
    if layer_bgra.ndim != 3 or layer_bgra.shape[2] != 4:
        raise ValueError("路线层必须是 BGRA 四通道图片")
    x, y, width, height = registration.observation_bbox
    observed = registration.observation_mask > 0
    if (
        observed.shape != (height, width)
        or x < 0
        or y < 0
        or x + width > layer_bgra.shape[1]
        or y + height > layer_bgra.shape[0]
    ):
        raise ValueError("可见地图区域与路线层尺寸不一致")
    if structure_mask is not None:
        if structure_mask.ndim != 2 or structure_mask.shape != layer_bgra.shape[:2]:
            raise ValueError("路线结构校验图与路线层尺寸不一致")
        route = structure_mask[y : y + height, x : x + width] > 127
    else:
        route = segment_map(
            layer_bgra[y : y + height, x : x + width, :3],
            runtime_registration_config(),
            infer_known_background=False,
            include_stairs=False,
        ).map_mask > 0
    observed_count = int(np.count_nonzero(observed))
    if observed_count < 1500:
        return {
            "accepted": False,
            "rejection_reasons": ["可见结构过少，无法验证路线投影"],
            "observed_pixels": observed_count,
        }
    covered = int(np.count_nonzero(observed & route))
    distances = cv2.distanceTransform((~route).astype(np.uint8), cv2.DIST_L2, 3)[observed]
    coverage = covered / observed_count
    p95 = float(np.percentile(distances, 95))
    reasons = []
    if coverage < 0.60:
        reasons.append(f"路线结构覆盖率过低：{coverage:.1%}")
    if p95 > 12.0:
        reasons.append(f"可见地图到路线结构的 P95 距离过高：{p95:.1f}px")
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "observed_pixels": observed_count,
        "covered_pixels": covered,
        "structure_coverage": float(coverage),
        "distance_p95_px": p95,
    }


def _scaled_processed_template(template: ProcessedTemplate, scale: float) -> ProcessedTemplate:
    if scale >= 0.999:
        return template

    def resize(mask: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
        return cv2.resize(mask, None, fx=scale, fy=scale, interpolation=interpolation)

    image = resize(template.image, cv2.INTER_AREA)
    corridor = resize(template.corridor_mask)
    room = resize(template.room_mask)
    map_mask = resize(template.map_mask)
    edge = resize(template.edge_mask)
    stairs = resize(template.stair_mask)
    return ProcessedTemplate(
        map_id=f"{template.map_id}_runtime_{scale:.4f}",
        image=image,
        corridor_mask=corridor,
        room_mask=room,
        map_mask=map_mask,
        edge_mask=edge,
        stair_mask=stairs,
        distance_transform=cv2.distanceTransform(cv2.bitwise_not(edge), cv2.DIST_L2, 3),
        crop_offset=(
            int(round(template.crop_offset[0] * scale)),
            int(round(template.crop_offset[1] * scale)),
        ),
        raw_path=template.raw_path,
        corridor_float=corridor.astype(np.float32) / 255.0,
        room_float=room.astype(np.float32) / 255.0,
    )


def _crop_segmentation(
    segmentation: SegmentationResult,
    bbox: tuple[int, int, int, int],
) -> SegmentationResult:
    x, y, width, height = bbox
    values = {}
    for field in fields(SegmentationResult):
        value = getattr(segmentation, field.name)
        values[field.name] = None if value is None else value[y : y + height, x : x + width].copy()
    return SegmentationResult(**values)


def prepare_registration_template(
    layout: str,
    floor: str,
    template_path: Path,
    cache_root: Path,
    *,
    config_override: dict | None = None,
) -> PreparedRegistrationTemplate:
    config = runtime_registration_config()
    if config_override:
        config.update(config_override)
    started = time.perf_counter()
    full_processed = preprocess_template(
        template_path.resolve(),
        config,
        cache_root.resolve(),
        map_id=f"{layout}_{floor}",
    )
    preprocess_ms = (time.perf_counter() - started) * 1000.0
    longest_side = max(full_processed.image.shape[:2])
    registration_scale = min(1.0, 1400.0 / max(1.0, float(longest_side)))
    processed = _scaled_processed_template(full_processed, registration_scale)
    factor = float(config["cpu_pyramid_factor"])
    pyramid, pyramid_ms = build_cpu_pyramid_template(processed, factor=factor)
    index_started = time.perf_counter()
    local_structure_index = build_or_load_local_structure_index(
        processed, template_path.resolve(), cache_root.resolve(),
    )
    local_structure_index_ms = (time.perf_counter() - index_started) * 1000.0
    if config.get("first_sparse_optimized_enabled", False):
        warm_started = time.perf_counter()
        warm_matching_template(processed, int(config["candidate_border_padding"]))
        warm_matching_template(pyramid, int(config["candidate_border_padding"]))
        preprocess_ms += (time.perf_counter() - warm_started) * 1000.0
    if config.get("regional_pose_diagnostics_enabled", True):
        warm_started = time.perf_counter()
        warm_regional_template(processed)
        preprocess_ms += (time.perf_counter() - warm_started) * 1000.0
    raw = imread_unicode(template_path.resolve())
    if raw is None:
        raise RuntimeError(f"无法读取干净底图：{template_path}")
    return PreparedRegistrationTemplate(
        layout=layout,
        floor=floor,
        path=template_path.resolve(),
        raw_image=raw,
        processed=processed,
        pyramid=pyramid,
        local_structure_index=local_structure_index,
        config=config,
        registration_scale=registration_scale,
        preprocess_ms=preprocess_ms,
        pyramid_ms=pyramid_ms,
        local_structure_index_ms=local_structure_index_ms,
    )


def _mapped_structure_fraction(matrix: np.ndarray, segmentation: SegmentationResult,
                               size: tuple[int, int]) -> float:
    """Fraction of source structure retained, independent of raster area/zoom."""
    width, height = size
    points = getattr(segmentation, "_coverage_points", None)
    bounds = cv2.boundingRect(segmentation.map_mask)
    x, y, w, h = bounds
    if not w or not h:
        return 0.0
    corners = np.float64([[x, y], [x+w-1, y], [x, y+h-1], [x+w-1, y+h-1]])
    mapped = corners @ matrix[:, :2].T + matrix[:, 2]
    if (mapped[:, 0].min() >= 0 and mapped[:, 0].max() < width
            and mapped[:, 1].min() >= 0 and mapped[:, 1].max() < height):
        return 1.0
    if points is None:
        yy, xx = np.nonzero(segmentation.map_mask)
        points = np.column_stack((xx, yy)).astype(np.float64)
        segmentation._coverage_points = points
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < width)
              & (mapped[:, 1] >= 0) & (mapped[:, 1] < height))
    return float(inside.mean())


def _validation(
    screenshot_to_processed: np.ndarray,
    screenshot_segmentation: SegmentationResult,
    template: ProcessedTemplate,
    *,
    origin: tuple[int, int] = (0, 0),
) -> dict:
    # Cache only on this immutable observation, never on template ids or globally.
    cached = getattr(screenshot_segmentation, "_validation_channels", None)
    if cached is None:
        channels = np.stack((screenshot_segmentation.corridor_mask,
                             screenshot_segmentation.room_mask,
                             screenshot_segmentation.edge_mask), axis=2)
        bounds = cv2.boundingRect(cv2.bitwise_or(screenshot_segmentation.map_mask,
                                                screenshot_segmentation.edge_mask))
        cached = (channels, bounds)
        screenshot_segmentation._validation_channels = cached
    channels, (bx, by, bw, bh) = cached
    matrix = np.asarray(screenshot_to_processed, np.float64).copy()
    matrix[:, 2] += matrix[:, :2] @ np.float64(origin)
    retained = _mapped_structure_fraction(
        matrix, screenshot_segmentation, (template.image.shape[1], template.image.shape[0]))
    corners = np.float64([[bx, by], [bx+bw, by], [bx, by+bh], [bx+bw, by+bh]])
    mapped_corners = corners @ matrix[:, :2].T + matrix[:, 2]
    left, top = np.maximum(0, np.floor(mapped_corners.min(axis=0))-2).astype(int)
    right, bottom = np.minimum([template.image.shape[1], template.image.shape[0]],
                                np.ceil(mapped_corners.max(axis=0))+3).astype(int)
    if right <= left or bottom <= top or bw == 0 or bh == 0:
        return {"observed_structure_pixels": 1, "mapped_structure_fraction": retained, "structure_precision": 0.,
                "semantic_precision": 0., "edge_median_error_px": None, "edge_p95_error_px": None}
    matrix[:, 2] -= (left, top)
    warped = cv2.warpAffine(channels, matrix, (right-left, bottom-top), flags=cv2.INTER_NEAREST)
    corridor, room, edge = cv2.split(warped)
    region = np.s_[top:bottom, left:right]
    mapped = cv2.bitwise_or(corridor, room)
    observed = max(1, cv2.countNonZero(mapped))
    structure_precision = cv2.countNonZero(cv2.bitwise_and(mapped, template.map_mask[region])) / observed
    semantic_precision = (
        cv2.countNonZero(cv2.bitwise_and(corridor, template.corridor_mask[region]))
        + cv2.countNonZero(cv2.bitwise_and(room, template.room_mask[region]))
    ) / observed
    edge_values = template.distance_transform[region][edge > 0]
    return {
        "observed_structure_pixels": int(observed),
        "mapped_structure_fraction": retained,
        "structure_precision": float(structure_precision),
        "semantic_precision": float(semantic_precision),
        "edge_median_error_px": float(np.median(edge_values)) if edge_values.size else None,
        "edge_p95_error_px": float(np.percentile(edge_values, 95)) if edge_values.size else None,
    }


def _same_location_scale_consensus(
    reviewed: list[tuple], observation_shape: tuple[int, int],
    validation: dict, direction: dict, visible_pixels: int,
) -> tuple[bool, float]:
    """Allow a strong sparse match when close competitors only vary in zoom."""
    if len(reviewed) < 2 or not 15000 <= visible_pixels < 30000:
        return False, 0.0
    edge = validation.get("edge_median_error_px")
    p95 = validation.get("edge_p95_error_px")
    if not (validation.get("mapped_structure_fraction", 0.0) >= 0.80
            and validation.get("structure_precision", 0.0) >= 0.97
            and validation.get("semantic_precision", 0.0) >= 0.90
            and edge is not None and edge <= 4.0
            and p95 is not None and p95 <= 30.0
            and direction.get("directional_comparable_pixels", 0) >= 100
            and (direction.get("directional_precision") or 0.0) >= 0.85
            and direction.get("topology_observed_pixels", 0) >= 150
            and (direction.get("topology_precision") or 0.0) >= 0.30):
        return False, 0.0
    winner = reviewed[0][0]
    height, width = observation_shape
    center = np.float32([width / 2.0, height / 2.0])
    anchor = np.float32(winner.top_left) + float(winner.scale) * center
    plausible = [item[0] for item in reviewed
                 if item[0].score >= winner.score - 0.035]
    if len(plausible) < 2:
        return False, 0.0
    max_scale_difference = 0.0
    for candidate in plausible:
        point = np.float32(candidate.top_left) + float(candidate.scale) * center
        if float(np.linalg.norm(point - anchor)) > 10.0:
            return False, 0.0
        max_scale_difference = max(
            max_scale_difference, abs(float(candidate.scale / winner.scale) - 1.0))
    return 0.04 < max_scale_difference <= 0.08, max_scale_difference


def _edge_direction_validation(
    screenshot_to_processed: np.ndarray,
    screenshot_segmentation: SegmentationResult,
    prepared: PreparedRegistrationTemplate,
) -> dict:
    """Compare only observed horizontal/vertical walls at index resolution."""
    factor = prepared.local_structure_index.factor
    use_cache = prepared.config.get("first_sparse_optimized_enabled", False)
    cached = getattr(screenshot_segmentation, "_direction_validation_cache", None) if use_cache else None
    if cached is None or cached[0] != factor:
        small_room = cv2.resize(screenshot_segmentation.room_mask, None, fx=factor, fy=factor,
                                interpolation=cv2.INTER_NEAREST)
        small_corridor = cv2.resize(screenshot_segmentation.corridor_mask, None, fx=factor, fy=factor,
                                    interpolation=cv2.INTER_NEAREST)
        small = cv2.bitwise_or(small_room, small_corridor)
        smooth = cv2.GaussianBlur(small, (3, 3), 0.6)
        dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.Canny(small, 40, 120) > 0
        horizontal = np.uint8(edge & (np.abs(dy) > 1.35 * np.abs(dx)))
        vertical = np.uint8(edge & (np.abs(dx) > 1.35 * np.abs(dy)))
        observed_features = extract_structural_features(small_room, small_corridor).channels
        cached = (factor, horizontal, vertical, observed_features)
        if use_cache:
            screenshot_segmentation._direction_validation_cache = cached
    _, horizontal, vertical, observed_features = cached
    matrix = screenshot_to_processed.copy()
    matrix[:, 2] *= np.float32(factor)
    shape = prepared.local_structure_index.feature_channels.shape[:2]
    size = (shape[1], shape[0])
    mapped_h = cv2.warpAffine(horizontal, matrix, size, flags=cv2.INTER_NEAREST) > 0
    mapped_v = cv2.warpAffine(vertical, matrix, size, flags=cv2.INTER_NEAREST) > 0
    features = prepared.local_structure_index.feature_channels
    expected_channels = getattr(prepared, "_direction_expected_channels", None) if use_cache else None
    if expected_channels is None:
        kernel = np.ones((5, 5), np.uint8)
        expected_channels = [cv2.dilate(np.uint8(features[:, :, channel] > 0.1), kernel) > 0
                             for channel in range(11)]
        if use_cache:
            prepared._direction_expected_channels = expected_channels
    expected_h, expected_v = expected_channels[:2]
    correct = int(np.count_nonzero(mapped_h & expected_h)
                  + np.count_nonzero(mapped_v & expected_v))
    wrong = int(np.count_nonzero(mapped_h & expected_v & ~expected_h)
                + np.count_nonzero(mapped_v & expected_h & ~expected_v))
    comparable = correct + wrong
    # Only positive observed landmarks count. Missing features in fog remain
    # unknown, so they never penalize a template candidate.
    topology_observed = 0
    topology_supported = 0
    for channel in range(2, 11):
        if np.count_nonzero(observed_features[:, :, channel]) == 0:
            continue
        mapped = cv2.warpAffine(np.uint8(observed_features[:, :, channel] > 0.1),
                                matrix, size, flags=cv2.INTER_NEAREST) > 0
        expected = expected_channels[channel]
        topology_observed += int(np.count_nonzero(mapped))
        topology_supported += int(np.count_nonzero(mapped & expected))
    return {"directional_comparable_pixels": comparable,
            "directional_precision": correct / comparable if comparable else None,
            "topology_observed_pixels": topology_observed,
            "topology_precision": (topology_supported / topology_observed
                                   if topology_observed else None)}


def _known_background_conflict(
    screenshot_to_processed: np.ndarray,
    known_background_crop: np.ndarray | None,
    crop_origin: tuple[int, int],
    template: ProcessedTemplate,
) -> tuple[int, float | None]:
    if known_background_crop is None:
        return 0, None
    crop_transform = screenshot_to_processed.copy()
    crop_transform[:, 2] += crop_transform[:, :2] @ np.float32(crop_origin)
    mapped = cv2.warpAffine(
        known_background_crop, crop_transform,
        (template.map_mask.shape[1], template.map_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    known = cv2.countNonZero(mapped)
    if known == 0:
        return 0, None
    collision = cv2.countNonZero(cv2.bitwise_and(mapped, template.map_mask))
    return known, collision / known


def _alignment_preview(
    screenshot: np.ndarray,
    screenshot_to_processed: np.ndarray,
    segmentation: SegmentationResult,
    template: ProcessedTemplate,
) -> np.ndarray:
    size = (template.image.shape[1], template.image.shape[0])
    warped = cv2.warpAffine(screenshot, screenshot_to_processed, size, flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(segmentation.map_mask, screenshot_to_processed, size, flags=cv2.INTER_NEAREST)
    warped_edge = cv2.warpAffine(segmentation.edge_mask, screenshot_to_processed, size, flags=cv2.INTER_NEAREST)
    output = template.image.copy()
    visible = warped_mask > 0
    output[visible] = cv2.addWeighted(output, 0.35, warped, 0.65, 0)[visible]
    output[warped_edge > 0] = (0, 0, 255)
    return output


def _register_screenshot_once(
    screenshot: np.ndarray,
    prepared: PreparedRegistrationTemplate,
    prior_screenshot_to_clean: np.ndarray | None = None,
    temporal_evidence: TemporalEvidence | None = None,
    soft_pose_hint: SoftPoseHint | None = None,
    *, _scale_bounds: tuple[float,float] | None = None,
    _source_segmentation: SegmentationResult | None = None,
) -> RuntimeRegistrationResult:
    total_started = time.perf_counter()
    config = dict(prepared.config)
    if prior_screenshot_to_clean is not None:
        for option in ("first_sparse_optimized_enabled", "first_sparse_point_refinement_enabled",
                       "first_sparse_candidate_pool_enabled", "first_sparse_geometry_enabled"):
            config[option] = False
    # 干净底图既有 1024px 资产，也有 1920px 资产。匹配尺度表示“截图坐标到
    # 干净底图坐标”的倍率，因此必须随资产分辨率变化，不能使用固定上限。
    width_ratio = prepared.processed.image.shape[1] / max(1.0, float(screenshot.shape[1]))
    height_ratio = prepared.processed.image.shape[0] / max(1.0, float(screenshot.shape[0]))
    nominal_scale = float(np.sqrt(width_ratio * height_ratio))
    # A tiny scale can fit one exposed room while projecting the whole route far
    # away from the map. Real captures for this profile cluster near nominal.
    config["scale_min"] = max(0.25, nominal_scale * 0.60)
    # The in-game map changes zoom while it recenters around revealed rooms.
    # A 1.42x nominal cap excluded valid captures (layout_18/b1 at 1.60x),
    # leaving only a displaced, low-edge-quality candidate. Keep the strict
    # geometric validation below to reject accidental matches in this range.
    config["scale_max"] = min(2.75, nominal_scale * 1.80)
    config["cpu_pyramid_coarse_scale_step"] = max(0.04, nominal_scale * 0.035)
    config["cpu_pyramid_fine_scale_radius"] = max(0.05, nominal_scale * 0.045)
    config["cpu_pyramid_fine_scale_step"] = max(0.008, nominal_scale * 0.007)
    if _scale_bounds is not None:
        config["scale_min"],config["scale_max"] = _scale_bounds
    try:
        started = time.perf_counter()
        segmentation = (_source_segmentation if _source_segmentation is not None else
                        segment_map(screenshot, config, infer_known_background=False, include_stairs=False))
        segmentation_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        search_segmentation = segmentation
        filter_search = (config.get('corridor_fog_filter_enabled', True)
                         and prior_screenshot_to_clean is None and soft_pose_hint is None)
        if filter_search:
            search_segmentation = getattr(segmentation, '_fog_search_segmentation', None)
            if search_segmentation is None:
                search_segmentation = filter_corridor_fog(screenshot, segmentation, config)
                segmentation._fog_search_segmentation = search_segmentation
        fog_filter_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        # Search confidence must not redefine verification's color references,
        # evidence count, or coordinate frame.
        bbox = component_content_bbox(
            segmentation.map_mask,
            padding=int(config["preprocess"]["crop_padding"]),
            min_area=int(config["preprocess"]["min_map_area"]),
        )
        x, y, width, height = bbox
        observation_image = screenshot[y : y + height, x : x + width].copy()
        observation = _crop_segmentation(segmentation, bbox)
        observation = add_known_background_evidence(observation_image, observation, config)
        search_observation = observation
        if search_segmentation is not segmentation:
            search_observation = add_known_background_evidence(observation_image,
                _crop_segmentation(search_segmentation, bbox), config)
        crop_ms = (time.perf_counter() - started) * 1000.0
        visible_pixels = cv2.countNonZero(segmentation.map_mask)
        visible_edges = cv2.countNonZero(segmentation.edge_mask)
        search_trace: list[dict] = []
        index_coarse_spatial_margin: float | None = None
        index_margin_status = "not_evaluated"
        sparse_review = None
        refined_transform = None
        credible_evidence = None
        credible_cache = {}
        fresh_pending_seeds = []
        multi_observation_confirmed = False
        regional_evidence = None
        regional_elapsed_ms = 0.0
        index_candidates = []

        def record_search(result: RuntimeRegistrationResult, tier: str, elapsed_ms: float) -> RuntimeRegistrationResult:
            search_trace.append({
                "tier": tier, "accepted": result.accepted,
                "elapsed_ms": round(elapsed_ms, 2),
                "reasons": list(result.rejection_reasons),
            })
            result.timings["history_search_trace"] = list(search_trace)
            result.timings["total"] = (time.perf_counter() - total_started) * 1000.0
            return result

        def finish(
            screenshot_to_processed: np.ndarray,
            matched: CpuPyramidMatchResult | None,
            mode: str,
            *,
            strict: bool,
            sparse_unique: bool = False,
            scale_only_consensus: bool = False,
        ) -> RuntimeRegistrationResult:
            nonlocal regional_evidence, regional_elapsed_ms, credible_evidence
            geometry = None
            geometry_confirmed = False
            if (config.get("credible_sparse_geometry_enabled", True) and prior_screenshot_to_clean is None
                    and 5000 <= visible_pixels < 30000):
                if credible_evidence is None:
                    credible_evidence = screenshot_evidence(observation_image, observation)
                local = screenshot_to_processed.copy()
                local[:, 2] += local[:, :2] @ np.float32([x, y])
                seeds = [local]
                seeds.extend(np.float32([[c.scale, 0, c.position[0]], [0, c.scale, c.position[1]]])
                             for c in index_candidates)
                if matched is not None:
                    seeds.extend(np.float32([[c.scale, 0, c.top_left[0]], [0, c.scale, c.top_left[1]]])
                                 for c in matched.result.top_k_candidates)
                    if mode.startswith("global"):
                        factor = float(config["cpu_pyramid_factor"])
                        seeds.extend(np.float32([[c.scale, 0, c.top_left[0]/factor], [0, c.scale, c.top_left[1]/factor]])
                                     for c in matched.coarse_result.top_k_candidates)
                for pending_pose in fresh_pending_seeds:
                    pending_local = pending_pose.copy()
                    pending_local[:, 2] += pending_local[:, :2] @ np.float32([x, y])
                    seeds.append(pending_local)
                key = tuple(tuple(np.round(p.ravel(), 3)) for p in seeds)
                if key not in credible_cache:
                    credible_cache[key] = review_geometry(prepared.processed, credible_evidence, seeds,
                        scale_min=config["scale_min"], scale_max=config["scale_max"],
                        viewport=screenshot.shape[:2], origin=(x,y))
                proposal, geometry = credible_cache[key]
                if proposal is not None and geometry["accepted"]:
                    candidate = proposal.copy()
                    candidate[:, 2] -= candidate[:, :2] @ np.float32([x,y])
                    check = _validation(candidate, segmentation, prepared.processed)
                    geometry = {**geometry, "proposal_structure_precision": check["structure_precision"],
                                "proposal_mapped_fraction": check["mapped_structure_fraction"]}
                    if check["structure_precision"] >= .97 and check["mapped_structure_fraction"] >= .95:
                        geometry_confirmed = True
                        screenshot_to_processed = candidate
                        if matched is not None:
                            matched = replace(matched, result=replace(matched.result,
                                best_position=tuple(float(v) for v in proposal[:, 2]), best_scale=float(proposal[0,0])))
            validation = _validation(screenshot_to_processed, segmentation, prepared.processed)
            validation["search_scale_bounds"] = [config["scale_min"],config["scale_max"]]
            trusted_observation = observation
            if geometry is not None:
                raw_validation = dict(validation)
                validation["credible_geometry"] = geometry
                validation["position_status"] = geometry["position_status"]
                validation["scale_status"] = geometry["scale_status"]
                if geometry_confirmed:
                    validation["raw_mask_validation"] = raw_validation
                    validation["semantic_precision"] = geometry["semantic_precision"]
                    validation["edge_median_error_px"] = float(geometry["wall_median_screen_px"] * screenshot_to_processed[0,0])
                    validation["edge_p95_error_px"] = float(geometry["wall_p95_screen_px"] * screenshot_to_processed[0,0])
                    validation["edge_error_basis"] = "credible_wall_segments_in_processed_template_pixels"
                    validation["sparse_refinement_confirmed"] = True
                    validation["credible_geometry_confirmed"] = True
                    corridor, room = credible_evidence.trusted_masks
                    trusted_map = corridor | room
                    trusted_valid = trusted_map | credible_evidence.background_mask
                    trusted_observation = replace(observation, corridor_mask=corridor, room_mask=room,
                        map_mask=trusted_map, known_background_mask=credible_evidence.background_mask,
                        valid_mask=trusted_valid, unknown_mask=cv2.bitwise_not(trusted_valid))
            validation["visible_structure_pixels"] = visible_pixels
            validation["visible_edge_pixels"] = visible_edges
            screenshot_to_clean = screenshot_to_processed.copy()
            screenshot_to_clean[:, 2] += np.float32(prepared.processed.crop_offset)
            screenshot_to_clean /= np.float32(prepared.registration_scale)
            clean_to_screen = cv2.invertAffineTransform(screenshot_to_clean)
            reasons = []
            minimum_pixels = 1500 if strict or sparse_unique else 5000
            minimum_structure = 0.95 if sparse_unique else 0.90 if strict else 0.78
            minimum_semantic = 0.90 if sparse_unique else 0.82 if strict else 0.65
            maximum_edge = (4.0 if sparse_unique or mode == "temporal_reuse"
                            else MAX_EDGE_MEDIAN_ERROR_PX)
            if validation["observed_structure_pixels"] < minimum_pixels:
                reasons.append("有效地图结构像素过少，无法可靠确认配准")
            if validation["structure_precision"] < minimum_structure:
                reasons.append(f"结构精度过低：{validation['structure_precision']:.1%}")
            if validation["semantic_precision"] < minimum_semantic:
                reasons.append(f"房间/走廊语义精度过低：{validation['semantic_precision']:.1%}")
            edge_median = validation["edge_median_error_px"]
            if edge_median is None or edge_median > maximum_edge:
                reasons.append(f"边缘中位误差过高：{edge_median if edge_median is not None else '无数据'}")
            if prior_screenshot_to_clean is None and visible_pixels < 30000:
                if validation["mapped_structure_fraction"] < 0.65:
                    validation["mapped_fraction_rejected"] = True
                    reasons.append("候选位置只覆盖了少量可见结构")
                direction = _edge_direction_validation(
                    screenshot_to_processed, segmentation, prepared)
                validation.update(direction)
                if (not geometry_confirmed and direction["directional_comparable_pixels"] >= 40
                        and direction["directional_precision"] < 0.55):
                    reasons.append("可见边缘的方向与干净底图不一致")
                if (not geometry_confirmed and direction["topology_observed_pixels"] >= 50
                        and direction["topology_precision"] < 0.15):
                    reasons.append("可见转角、门口或走廊连接与干净底图不一致")
            if temporal_evidence is not None and temporal_evidence.matches(prepared):
                history_details = temporal_evidence.conflict_details(
                    corridor=trusted_observation.corridor_mask,
                    room=trusted_observation.room_mask,
                    background=trusted_observation.known_background_mask,
                    transform=screenshot_to_processed,
                    origin=(x, y),
                )
                conflict = history_details["conflict_rate"]
                validation["history_compared_pixels"] = history_details["compared_pixels"]
                validation["history_conflict_rate"] = conflict
                validation["history_hard_conflict_rate"] = (
                    history_details["hard_conflict_pixels"] / history_details["compared_pixels"]
                    if history_details["compared_pixels"] else None
                )
                validation["history_conflict_components"] = {
                    key: history_details[key] for key in (
                        "semantic_pixels", "structure_on_background_pixels",
                        "background_on_structure_pixels",
                    )
                }
                if conflict is not None and conflict > 0.20:
                    if _background_only_history_conflict(validation, history_details):
                        validation["history_background_soft_override"] = True
                    else:
                        reasons.append(f"与本局已确认结构冲突：{conflict:.1%}")
            if prior_screenshot_to_clean is None and visible_pixels < 30000:
                margin = index_coarse_spatial_margin
                exceptionally_clear = (
                    visible_pixels >= 20000
                    and validation["structure_precision"] >= 0.98
                    and validation["semantic_precision"] >= 0.97
                    and edge_median is not None and edge_median <= 10.0
                )
                validation["sparse_spatial_margin"] = margin
                validation["sparse_margin_status"] = index_margin_status
                validation["sparse_exceptionally_clear"] = exceptionally_clear
                validation["multi_observation_confirmed"] = multi_observation_confirmed
                refined_confirmed = geometry_confirmed or (refined_transform is not None
                                     and np.allclose(screenshot_to_processed, refined_transform, atol=1e-4))
                requires_sparse_confirmation = (not multi_observation_confirmed and not exceptionally_clear
                                                and not scale_only_consensus
                                                and (margin is None or margin < .08))
                if requires_sparse_confirmation and not refined_confirmed:
                    validation["sparse_spatial_rejected"] = True
                    reasons.append("首次稀疏配准的位置证据尚未确认，请继续探索")
                if sparse_review is not None:
                    validation["sparse_pose_review"] = sparse_review
                    if (sparse_review["position_status"] == "confirmed" and not refined_confirmed
                            and requires_sparse_confirmation):
                        reasons = [r for r in reasons if r != "首次稀疏配准的位置证据尚未确认，请继续探索"]
                        validation.pop("sparse_spatial_rejected", None)
                        validation["sparse_scale_rejected"] = True
                        reasons.append("位置已确认，但缩放精度仍不足，请继续探索")
                    elif sparse_review["position_status"] == "ambiguous" and not refined_confirmed:
                        validation["sparse_spatial_rejected"] = True
                        reasons = [r for r in reasons if r != "首次稀疏配准的位置证据尚未确认，请继续探索"]
                        reasons.append("可见结构对应多个可能位置，请继续探索")
                    if refined_confirmed:
                        validation["sparse_refinement_confirmed"] = True
                geometry_risk = (geometry is not None and geometry["position_status"] == "confirmed"
                                 and geometry.get("wall_support_fraction", 0) >= .75
                                 and geometry.get("wall_median_screen_px", float("inf")) <= 2
                                 and geometry.get("viewport_disagreement_p95_px", 0) > 8)
                if geometry is not None and not geometry_confirmed and (
                        config.get("credible_sparse_geometry_strict", False) or geometry_risk):
                    reasons = [r for r in reasons if r not in (
                        "首次稀疏配准的位置证据尚未确认，请继续探索",
                        "可见结构对应多个可能位置，请继续探索")]
                    validation.pop("sparse_spatial_rejected", None)
                    validation["sparse_scale_rejected"] = True
                    if geometry["position_status"] == "confirmed":
                        reasons.append("位置已确认，但缩放精度仍不足，请继续探索")
                    elif geometry["position_status"] == "ambiguous":
                        validation["sparse_spatial_rejected"] = True
                        reasons.append("可见结构存在多个可信位置或尺度分支，请继续探索")
                    else:
                        validation["sparse_spatial_rejected"] = True
                        reasons.append("可靠墙段或语义证据不足，尚不能确认位置与缩放，请继续探索")
            if scale_only_consensus:
                validation["scale_only_relaxed"] = True
            if (prior_screenshot_to_clean is None
                    and config.get("regional_pose_diagnostics_enabled", True)):
                regional_started = time.perf_counter()
                if regional_evidence is None:
                    regional_evidence = build_regional_evidence(observation)
                local_pose = screenshot_to_processed.copy()
                local_pose[:, 2] += local_pose[:, :2] @ np.float32([x, y])
                alternatives = []
                if matched is not None:
                    alternatives = [np.float32([[c.scale, 0, c.top_left[0]],
                                                [0, c.scale, c.top_left[1]]])
                                    for c in matched.result.top_k_candidates[:12]]
                    if mode.startswith("global"):
                        factor = float(config["cpu_pyramid_factor"])
                        alternatives.extend(np.float32([[c.scale, 0, c.top_left[0]/factor],
                                                         [0, c.scale, c.top_left[1]/factor]])
                                            for c in matched.coarse_result.top_k_candidates)
                alternatives.extend(np.float32([[c.scale, 0, c.position[0]],
                                                 [0, c.scale, c.position[1]]])
                                    for c in index_candidates[:8])
                regional = analyze_pose(prepared.processed, regional_evidence, local_pose, alternatives)
                validation["regional_evidence"] = regional
                contradictory = (regional["region_count"] >= 4
                                 and regional.get("contradictory_regions", 0) >= 2
                                 and regional.get("contradictory_regions", 0) / regional["region_count"] >= .35)
                contradictory = contradictory or regional["strong_spatial_competitors"] > 0
                validation["regional_guard_would_reject"] = contradictory
                if contradictory and not geometry_confirmed and config.get("regional_pose_guard_enabled", False):
                    validation["regional_pose_rejected"] = True
                    reasons.append("多个可见区域与候选位置矛盾，请继续探索")
                regional_elapsed_ms += (time.perf_counter()-regional_started)*1000
            if (reasons and geometry is not None and temporal_evidence is not None
                    and temporal_evidence.matches(prepared)):
                hypotheses = []
                for branch in geometry.get("branches", []):
                    pending_local = pose_matrix(np.float32([*branch["anchor"], branch["scale"]]), credible_evidence.center)
                    pending_local[:, 2] -= pending_local[:, :2] @ np.float32([x,y])
                    hypotheses.append(pending_local)
                temporal_evidence.remember_candidates(hypotheses, segmentation.map_mask)
            preview = (_alignment_preview(screenshot, screenshot_to_processed, segmentation, prepared.processed)
                       if not reasons or not strict else None)
            reasons = list(dict.fromkeys(reasons))
            if geometry is not None:
                validation["geometry_acceptance_policy"] = ("strict" if config.get("credible_sparse_geometry_strict", False)
                                                            else "verified_correction_and_scale_risk")
                validation["acceptance_basis"] = ("credible_geometry" if geometry_confirmed
                                                  else "rejected" if reasons else "legacy_evidence")
            timing = matched.timing if matched is not None else None
            validation['fog_filter'] = {
                'enabled': bool(config.get('corridor_fog_filter_enabled', True)),
                'applied': bool(filter_search),
                'uncertain_pixels': (cv2.countNonZero(search_segmentation.uncertain_mask)
                                     if search_segmentation.uncertain_mask is not None else 0),
                'scope': 'candidate_search_only',
            }
            return RuntimeRegistrationResult(
                accepted=not reasons,
                rejection_reasons=reasons,
                clean_to_screen=clean_to_screen,
                screenshot_to_clean=screenshot_to_clean,
                observation_bbox=bbox,
                match=matched,
                validation=validation,
                timings={
                    "method": mode,
                    "template_preprocess_once": prepared.preprocess_ms,
                    "template_pyramid_once": prepared.pyramid_ms,
                    "local_structure_index_once": prepared.local_structure_index_ms,
                    "template_registration_scale": prepared.registration_scale,
                    "segmentation": segmentation_ms,
                    "fog_filter": fog_filter_ms,
                    "crop_and_evidence": crop_ms,
                    "regional_evidence": regional_elapsed_ms,
                    "sparse_pose_review": sparse_review["elapsed_ms"] if sparse_review is not None else 0.0,
                    "credible_geometry": geometry["elapsed_ms"] if geometry is not None else 0.0,
                    "observation_pyramid": timing.observation_pyramid_ms if timing else 0.0,
                    "coarse_match": timing.coarse_match_ms if timing else 0.0,
                    "fine_match": timing.fine_match_ms if timing else 0.0,
                    "structure_match": timing.total_match_ms if timing else 0.0,
                    "sparse_refinement": timing.sparse_refinement if timing else {},
                    "history_search_trace": list(search_trace),
                    "total": (time.perf_counter() - total_started) * 1000.0,
                },
                observation_image=observation_image,
                observation_mask=observation.map_mask,
                alignment_preview=preview,
                observation_segmentation=trusted_observation,
                source_segmentation=segmentation,
                search_segmentation=search_segmentation,
            )

        # An unresolved pose is not a prior. It can only become one when a
        # later, nearly stationary map view reveals genuinely new structure.
        if (prior_screenshot_to_clean is None and temporal_evidence is not None
                and temporal_evidence.matches(prepared) and visible_pixels < 30000):
            new_mask = temporal_evidence.pending_new_mask(
                segmentation.map_mask, allow_translation=bool(config.get("pending_pan_enabled", False)
                    or config.get("credible_sparse_geometry_enabled", True)))
            pending = temporal_evidence.pending_poses
            if new_mask is not None and pending is not None:
                fresh_pending_seeds = [p.copy() for p in pending.transforms]
                fresh = _crop_segmentation(segmentation, bbox)
                local_new = new_mask[y:y + height, x:x + width]
                fresh.corridor_mask = cv2.bitwise_and(fresh.corridor_mask, local_new)
                fresh.room_mask = cv2.bitwise_and(fresh.room_mask, local_new)
                fresh.map_mask = cv2.bitwise_or(fresh.corridor_mask, fresh.room_mask)
                fresh.edge_mask = cv2.bitwise_and(fresh.edge_mask,
                                                  cv2.dilate(local_new, np.ones((5, 5), np.uint8)))
                evaluations = []
                for pose in pending.transforms:
                    quality = _validation(pose, fresh, prepared.processed, origin=(x, y))
                    edge = quality["edge_median_error_px"]
                    score = (0.5 * quality["structure_precision"]
                             + 0.5 * quality["semantic_precision"])
                    if edge is None or edge > 10.0:
                        score -= 0.20
                    evaluations.append((score, pose, quality))
                evaluations.sort(key=lambda item: item[0], reverse=True)
                best_score, best_pose, best_quality = evaluations[0]
                second_score = evaluations[1][0] if len(evaluations)>1 else best_score
                if (best_quality["observed_structure_pixels"] >= 1200
                        and best_quality["structure_precision"] >= 0.96
                        and best_quality["semantic_precision"] >= 0.92
                        and best_quality["edge_median_error_px"] is not None
                        and best_quality["edge_median_error_px"] <= 4.0
                        and best_score - second_score >= 0.10):
                    multi_observation_confirmed = True
                    confirmed = finish(best_pose, None, "multi_observation", strict=True)
                    confirmed.validation["hypothesis_count"] = len(evaluations)
                    confirmed.validation["new_structure_pixels"] = int(cv2.countNonZero(new_mask))
                    confirmed.validation["new_evidence_score_margin"] = best_score - second_score
                    if confirmed.accepted:
                        temporal_evidence.clear_pending()
                        return record_search(confirmed, "multi_observation",
                                             (time.perf_counter() - total_started) * 1000.0)
                    multi_observation_confirmed = False

        # A relaxed zoom estimate may seed a later capture, but it never
        # becomes trusted structural history. Recheck it against this frame.
        if prior_screenshot_to_clean is None and soft_pose_hint is not None:
            soft = np.asarray(soft_pose_hint.screenshot_to_clean, dtype=np.float32)
            old_x, old_y, old_w, old_h = soft_pose_hint.observation_bbox
            old_mask = soft_pose_hint.observation_mask
            same_shape = (old_mask.shape == (old_h, old_w)
                          and old_x >= 0 and old_y >= 0
                          and old_x + old_w <= screenshot.shape[1]
                          and old_y + old_h <= screenshot.shape[0])
            overlap = 0.0
            if same_shape:
                current_region = segmentation.map_mask[
                    old_y:old_y + old_h, old_x:old_x + old_w] > 0
                old_binary = old_mask > 0
                intersection = int(np.count_nonzero(old_binary & current_region))
                union = int(np.count_nonzero(old_binary)) + visible_pixels - intersection
                overlap = intersection / max(1, union)
            if (overlap >= 0.96 and soft.shape == (2, 3) and np.isfinite(soft).all()
                    and abs(float(soft[0, 1])) < 1e-4
                    and abs(float(soft[1, 0])) < 1e-4
                    and abs(float(soft[0, 0] - soft[1, 1])) < 1e-3):
                soft_processed = soft * np.float32(prepared.registration_scale)
                soft_processed[:, 2] -= np.float32(prepared.processed.crop_offset)
                quality = _validation(soft_processed, segmentation, prepared.processed)
                edge = quality["edge_median_error_px"]
                if (15000 <= visible_pixels < 30000
                        and quality["mapped_structure_fraction"] >= 0.80
                        and quality["structure_precision"] >= 0.97
                        and quality["semantic_precision"] >= 0.90
                        and edge is not None and edge <= 4.0
                        and quality["edge_p95_error_px"] is not None
                        and quality["edge_p95_error_px"] <= 30.0):
                    direction = _edge_direction_validation(
                        soft_processed, segmentation, prepared)
                    if (direction["directional_comparable_pixels"] >= 100
                            and (direction["directional_precision"] or 0.0) >= 0.85
                            and direction["topology_observed_pixels"] >= 150
                            and (direction["topology_precision"] or 0.0) >= 0.30):
                        reused = finish(soft_processed, None, "soft_pose_reuse",
                                        strict=True, scale_only_consensus=True)
                        if reused.accepted:
                            reused.validation["soft_pose_reuse"] = True
                            reused.validation["soft_observation_overlap"] = overlap
                            return record_search(reused, "soft_pose_reuse",
                                                 (time.perf_counter() - total_started) * 1000.0)

        if prior_screenshot_to_clean is not None:
            prior = np.asarray(prior_screenshot_to_clean, dtype=np.float32)
            if (prior.shape == (2, 3) and np.isfinite(prior).all() and
                    abs(float(prior[0, 1])) < 1e-4 and abs(float(prior[1, 0])) < 1e-4 and
                    abs(float(prior[0, 0] - prior[1, 1])) < 1e-3):
                prior_scale = float(prior[0, 0]) * prepared.registration_scale
                if config["scale_min"] <= prior_scale <= config["scale_max"]:
                    prior_processed = prior.copy()
                    prior_processed *= np.float32(prepared.registration_scale)
                    prior_processed[:, 2] -= np.float32(prepared.processed.crop_offset)
                    tier_started = time.perf_counter()
                    reused = record_search(finish(prior_processed, None, "temporal_reuse", strict=True),
                                           "reuse", (time.perf_counter() - tier_started) * 1000.0)
                    if reused.accepted:
                        return reused
                    fast_config = dict(config)
                    fast_config.update({
                        "scale_min": prior_scale,
                        "scale_max": prior_scale,
                        "cpu_pyramid_fine_scale_radius": 0.0,
                        "cpu_pyramid_fine_position_radius": 12.0,
                        "cpu_pyramid_fine_candidate_pool_size": 2,
                        "geometric_review_enabled": False,
                        "top_k": 2,
                    })
                    tier_started = time.perf_counter()
                    fast_match = match_single_frame_cpu_pyramid(
                        template=prepared.processed,
                        pyramid_template=prepared.pyramid,
                        observation=search_observation,
                        cfg=fast_config,
                        pyramid_factor=float(fast_config["cpu_pyramid_factor"]),
                        pyramid_template_once_ms=prepared.pyramid_ms,
                    )
                    fast_result = fast_match.result
                    tracked: RuntimeRegistrationResult | None = None
                    sparse_limit = max(5000, int(screenshot.shape[0] * screenshot.shape[1] * 0.025))
                    sparse_observation = cv2.countNonZero(segmentation.map_mask) < sparse_limit
                    if fast_result.best_position is not None and fast_result.best_scale is not None:
                        scale = float(fast_result.best_scale)
                        fast_transform = np.float32([
                            [scale, 0.0, float(fast_result.best_position[0]) - scale * x],
                            [0.0, scale, float(fast_result.best_position[1]) - scale * y],
                        ])
                        tracked = finish(fast_transform, fast_match, "temporal_search", strict=True)
                        if tracked.accepted:
                            if sparse_observation:
                                center = np.float32([x + width / 2, y + height / 2])
                                previous_point = prior[:, :2] @ center + prior[:, 2]
                                tracked_point = (tracked.screenshot_to_clean[:, :2] @ center +
                                                 tracked.screenshot_to_clean[:, 2])
                                movement = float(np.linalg.norm(tracked_point - previous_point))
                                strongly_reconfirmed = (
                                    tracked.validation.get("history_compared_pixels", 0) >= 1000
                                    and (tracked.validation.get("history_conflict_rate") or 0.0) <= 0.03
                                )
                                motion_limit = (20.0 if visible_pixels < 10000
                                                and not strongly_reconfirmed else 120.0)
                                tracked.validation["history_movement_px"] = movement
                                tracked.validation["history_motion_limit_px"] = motion_limit
                                if movement > motion_limit:
                                    tracked.accepted = False
                                    tracked.rejection_reasons.append("稀疏画面匹配位置偏离历史位置过远")
                                    if motion_limit == 20.0:
                                        tracked.validation["sparse_motion_rejected"] = True
                        tracked = record_search(tracked, "small_translation",
                                                (time.perf_counter() - tier_started) * 1000.0)
                        if tracked.accepted:
                            return tracked
                    else:
                        search_trace.append({"tier": "small_translation", "accepted": False,
                                             "elapsed_ms": round((time.perf_counter() - tier_started) * 1000.0, 2),
                                             "reasons": ["固定尺度小范围搜索无候选"]})

                    # Keep the center of the observed patch fixed in template
                    # coordinates while probing zoom changes. The first wider
                    # tier remains local; only a well-exposed view may recenter.
                    factor = float(config["cpu_pyramid_factor"])
                    center_screen = np.float32([x + width / 2.0, y + height / 2.0])
                    center_template = prior_processed[:, :2] @ center_screen + prior_processed[:, 2]
                    reuse_edge = reused.validation.get("edge_median_error_px")
                    nearly_aligned = (reuse_edge is not None and reuse_edge <= 10.0
                                      and reused.validation.get("structure_precision", 0.0) >= 0.85)
                    local_radius = 16 if nearly_aligned else 24
                    recenter_radius = 48 if nearly_aligned else 64
                    if temporal_evidence is not None and temporal_evidence.matches(prepared):
                        predicted = temporal_evidence.predicted_center(center_screen)
                        if predicted is not None:
                            search_trace.append({"tier": "predict_center", "accepted": True,
                                                 "elapsed_ms": 0.0,
                                                 "movement_px": round(float(np.linalg.norm(
                                                     predicted - center_template)), 2),
                                                 "reasons": []})
                            center_template = predicted
                    for tier, radius, multipliers in (
                        ("local_zoom", local_radius, (0.98, 1.0, 1.02)),
                        ("recenter_zoom", recenter_radius, (0.94, 0.97, 1.0, 1.03, 1.06)),
                    ):
                        if visible_pixels < 5000 or visible_edges < 100:
                            search_trace.append({"tier": tier, "accepted": False, "elapsed_ms": 0.0,
                                                 "reasons": ["可见结构不足，跳过扩展搜索"]})
                            continue
                        if tier == "recenter_zoom" and sparse_observation:
                            search_trace.append({"tier": tier, "accepted": False, "elapsed_ms": 0.0,
                                                 "reasons": ["可见结构不足，跳过大范围重新居中"]})
                            continue
                        tier_started = time.perf_counter()
                        candidates = []
                        for multiplier in multipliers:
                            candidate_scale = prior_scale * multiplier
                            if not config["scale_min"] <= candidate_scale <= config["scale_max"]:
                                continue
                            anchor = center_template - candidate_scale * np.float32([width / 2.0, height / 2.0])
                            refined = refine_local_structure(
                                prepared.pyramid, search_observation,
                                position=(float(anchor[0] * factor), float(anchor[1] * factor)),
                                scales=[candidate_scale * factor], radius=radius,
                            )
                            if refined is not None:
                                candidates.append(refined)
                        candidates.sort(key=lambda item: item.score, reverse=True)
                        accepted_candidate = None
                        rejected_candidate = None
                        for candidate in candidates[:3]:
                            scale = candidate.scale / factor
                            position = np.float32(candidate.position) / factor
                            transform = np.float32([
                                [scale, 0.0, position[0] - scale * x],
                                [0.0, scale, position[1] - scale * y],
                            ])
                            checked = finish(transform, None, "temporal_" + tier, strict=True)
                            if checked.accepted:
                                new_center = checked.screenshot_to_clean[:, :2] @ center_screen + checked.screenshot_to_clean[:, 2]
                                old_center = prior[:, :2] @ center_screen + prior[:, 2]
                                movement = float(np.linalg.norm(new_center - old_center))
                                strongly_reconfirmed = (
                                    checked.validation.get("history_compared_pixels", 0) >= 1000
                                    and (checked.validation.get("history_conflict_rate") or 0.0) <= 0.03
                                )
                                motion_limit = (20.0 if sparse_observation and visible_pixels < 10000
                                                and not strongly_reconfirmed else
                                                120.0 if sparse_observation else
                                                160.0 if tier == "local_zoom" else 320.0)
                                checked.validation["history_movement_px"] = movement
                                checked.validation["history_motion_limit_px"] = motion_limit
                                if movement > motion_limit:
                                    checked.accepted = False
                                    checked.rejection_reasons.append("候选位置偏离历史位置过远")
                                    if motion_limit == 20.0:
                                        checked.validation["sparse_motion_rejected"] = True
                                elif tier == "recenter_zoom":
                                    checked.validation["history_motion_classification"] = "map_recentered"
                            if checked.accepted:
                                accepted_candidate = checked
                                break
                            rejected_candidate = checked
                        if accepted_candidate is not None:
                            return record_search(accepted_candidate, tier,
                                                 (time.perf_counter() - tier_started) * 1000.0)
                        search_trace.append({"tier": tier, "accepted": False,
                                             "elapsed_ms": round((time.perf_counter() - tier_started) * 1000.0, 2),
                                             "reasons": (rejected_candidate.rejection_reasons if rejected_candidate
                                                         else ["可用尺度范围内无候选"])})
                    # With only a tiny island exposed, an unconstrained global
                    # search can match the same room elsewhere and pass a weak
                    # overlap check. Keep the trusted history or show no overlay.
                    if sparse_observation:
                        reason = "可见结构太少且历史位置无法确认，已停止全局搜索"
                        if tracked is not None:
                            tracked.rejection_reasons.append(reason)
                            tracked.timings["history_search_trace"] = list(search_trace)
                            tracked.timings["total"] = (time.perf_counter() - total_started) * 1000.0
                            return tracked
                        return RuntimeRegistrationResult(
                            accepted=False,
                            rejection_reasons=[reason],
                            clean_to_screen=None,
                            screenshot_to_clean=None,
                            observation_bbox=bbox,
                            match=fast_match,
                            validation={},
                            timings={
                                "method": "temporal_insufficient_evidence",
                                "segmentation": segmentation_ms,
                                "crop_and_evidence": crop_ms,
                                "structure_match": fast_match.timing.total_match_ms,
                                "history_search_trace": list(search_trace),
                                "total": (time.perf_counter() - total_started) * 1000.0,
                            },
                            observation_image=observation_image,
                            observation_mask=observation.map_mask,
                            alignment_preview=None,
                        )

        # A few isolated pixels cannot establish a unique whole-map pose. Avoid
        # spending hundreds of milliseconds on an unconstrained scale search.
        if prior_screenshot_to_clean is None and (visible_pixels < 2000 or visible_edges < 80):
            return RuntimeRegistrationResult(
                accepted=False,
                rejection_reasons=["可见结构不足以确定唯一位置，请再探索一小段地图"],
                clean_to_screen=None,
                screenshot_to_clean=None,
                observation_bbox=bbox,
                match=None,
                validation={"visible_structure_pixels": visible_pixels, "visible_edge_pixels": visible_edges},
                timings={"method": "insufficient_evidence", "segmentation": segmentation_ms,
                         "crop_and_evidence": crop_ms,
                         "total": (time.perf_counter() - total_started) * 1000.0},
                observation_image=observation_image,
                observation_mask=observation.map_mask,
                alignment_preview=None,
            )

        index_query_ms = 0.0
        index_coarse_ms = 0.0
        index_fine_ms = 0.0
        index_feature_ms = 0.0
        index_coarse_score = 0.0
        index_coarse_spatial_margin = None
        index_fine_scale_count = 0
        index_refine_ms = 0.0
        index_candidate_count = 0
        index_history_rejected_count = 0
        index_skip_reason = "disabled"
        index_reference: tuple[np.ndarray, dict] | None = None
        index_candidates = []
        if config.get("local_structure_index_enabled", False):
            index_query = prepared.local_structure_index.query(
                search_observation,
                scale_min=float(config["scale_min"]),
                scale_max=float(config["scale_max"]),
                scale_step=float(config["cpu_pyramid_coarse_scale_step"]),
                coarse_min_score=float(config["local_structure_index_coarse_min_score"]),
                fine_min_score=float(config["local_structure_index_min_score"]),
            )
            index_candidates = index_query.candidates
            if (temporal_evidence is not None and temporal_evidence.matches(prepared)
                    and temporal_evidence.updates):
                consistent = []
                for candidate in index_candidates:
                    candidate_transform = np.float32([
                        [candidate.scale, 0.0, candidate.position[0] - candidate.scale * x],
                        [0.0, candidate.scale, candidate.position[1] - candidate.scale * y],
                    ])
                    _compared, conflict = temporal_evidence.conflict(
                        corridor=observation.corridor_mask, room=observation.room_mask,
                        background=observation.known_background_mask,
                        transform=candidate_transform, origin=(x, y),
                    )
                    if conflict is not None and conflict > 0.20:
                        index_history_rejected_count += 1
                    else:
                        consistent.append(candidate)
                index_candidates = consistent
            index_query_ms = index_query.total_ms
            index_coarse_ms = index_query.coarse_ms
            index_fine_ms = index_query.fine_ms
            index_feature_ms = index_query.feature_ms
            index_coarse_score = index_query.coarse_score
            index_coarse_spatial_margin = index_query.coarse_spatial_margin
            index_margin_status = index_query.coarse_margin_status
            index_fine_scale_count = index_query.fine_scale_count
            index_candidate_count = len(index_candidates)
            if (prior_screenshot_to_clean is None and temporal_evidence is not None
                    and temporal_evidence.matches(prepared) and visible_pixels < 30000
                    and len(index_candidates) >= 2 and
                    (index_coarse_spatial_margin is None or index_coarse_spatial_margin < 0.08)):
                plausible = []
                for candidate in index_candidates[:8]:
                    if candidate.score < max(0.70, index_candidates[0].score - 0.10):
                        continue
                    transform = np.float32([
                        [candidate.scale, 0.0, candidate.position[0] - candidate.scale * x],
                        [0.0, candidate.scale, candidate.position[1] - candidate.scale * y],
                    ])
                    quality = _validation(transform, segmentation, prepared.processed)
                    edge = quality["edge_median_error_px"]
                    if (quality["structure_precision"] >= 0.90
                            and quality["semantic_precision"] >= 0.85
                            and edge is not None and edge <= 6.0):
                        plausible.append(transform)
                temporal_evidence.remember_candidates(plausible, segmentation.map_mask)
            if (index_candidates and index_candidates[0].score >=
                    float(config["local_structure_index_min_score"])):
                first = index_candidates[0]
                first_transform = np.float32([
                    [first.scale, 0.0, first.position[0] - first.scale * x],
                    [0.0, first.scale, first.position[1] - first.scale * y],
                ])
                first_validation = _validation(first_transform, segmentation, prepared.processed)
                index_reference = (first_transform, first_validation)
                first_edge = first_validation["edge_median_error_px"]
                ambiguous = False
                for competitor in index_candidates[1:8]:
                    if competitor.score < first.score - 0.035:
                        break
                    visible_center = observation_anchor(observation)
                    if np.linalg.norm(np.subtract(first.position, competitor.position)
                                      + (first.scale-competitor.scale)*visible_center) < 35.0:
                        continue
                    competitor_transform = np.float32([
                        [competitor.scale, 0.0, competitor.position[0] - competitor.scale * x],
                        [0.0, competitor.scale, competitor.position[1] - competitor.scale * y],
                    ])
                    competing = _validation(competitor_transform, segmentation, prepared.processed)
                    edge = competing["edge_median_error_px"]
                    if (competing["structure_precision"] >= max(
                            0.85, first_validation["structure_precision"] - 0.02)
                            and competing["semantic_precision"] >= max(
                                0.80, first_validation["semantic_precision"] - 0.02)
                            and edge is not None and first_edge is not None
                            and edge <= min(MAX_EDGE_MEDIAN_ERROR_PX, first_edge + 1.5)):
                        ambiguous = True
                        break
                if not ambiguous:
                    center = (width / 2.0, height / 2.0)
                    anchor = [
                        first.position[0] + first.scale * center[0] + prepared.processed.crop_offset[0],
                        first.position[1] + first.scale * center[1] + prepared.processed.crop_offset[1],
                    ]
                    fine_step = float(config["local_structure_index_refine_scale_step"])
                    local_scales = sorted({
                        round(float(np.clip(first.scale + offset * fine_step,
                                            config["scale_min"], config["scale_max"])), 4)
                        for offset in (-1, 0, 1)
                    })
                    local_config = dict(config)
                    local_config.update({
                        "scale_values": local_scales,
                        "top_k": 4,
                        "candidate_pool_size": 4,
                        # The full-size structure/semantic/edge validation below
                        # remains mandatory. Skip the expensive bidirectional
                        # review during this tiny local search.
                        "geometric_review_enabled": False,
                        "pre_match_search_windows_enabled": True,
                        "allowed_player_template_points": [anchor],
                        "allowed_player_template_radius": float(
                            config["local_structure_index_refine_position_radius"]
                        ),
                    })
                    refine_started = time.perf_counter()
                    if config.get("local_structure_direct_refine_enabled", True):
                        refine_factor = float(config["cpu_pyramid_factor"])
                        direct = refine_local_structure(
                            prepared.pyramid, search_observation,
                            position=(first.position[0] * refine_factor,
                                      first.position[1] * refine_factor),
                            scales=[value * refine_factor for value in local_scales],
                            radius=max(2, int(round(float(
                                config["local_structure_index_refine_position_radius"]
                            )) * refine_factor)),
                        )
                        direct_position = (
                            (int(round(direct.position[0] / refine_factor)),
                             int(round(direct.position[1] / refine_factor)))
                            if direct is not None else None
                        )
                        direct_scale = (direct.scale / refine_factor
                                        if direct is not None else None)
                        candidates = ([MatchCandidate(
                            top_left=direct_position, scale=direct_scale,
                            score=direct.score, corridor_score=0.0, room_score=0.0,
                            edge_score=0.0, stair_score=0.0,
                        )] if direct is not None else [])
                        local_result = LocalizationResult(
                            map_id=prepared.processed.map_id,
                            best_position=direct_position,
                            best_scale=direct_scale,
                            confidence=direct.score if direct is not None else 0.0,
                            low_confidence=direct is None,
                            top_k_candidates=candidates,
                            player_roi=None,
                            player_position_in_template=None,
                        )
                    else:
                        local_result = match_single_frame(
                            template=prepared.processed, observation=search_observation,
                            cfg=local_config, player_roi=center,
                        )
                    index_refine_ms = (time.perf_counter() - refine_started) * 1000.0
                    if local_result.best_position is not None and local_result.best_scale is not None:
                        scale = float(local_result.best_scale)
                        local_transform = np.float32([
                            [scale, 0.0, float(local_result.best_position[0]) - scale * x],
                            [0.0, scale, float(local_result.best_position[1]) - scale * y],
                        ])
                        local_timing = CpuPyramidTiming(
                            pyramid_template_once_ms=prepared.pyramid_ms,
                            observation_pyramid_ms=0.0,
                            coarse_match_ms=index_query_ms,
                            fine_match_ms=index_refine_ms,
                            total_match_ms=index_query_ms + index_refine_ms,
                            coarse_scale_count=0,
                            fine_scale_count=len(local_scales),
                        )
                        local_match = CpuPyramidMatchResult(local_result, local_result, local_timing)
                        fast = finish(local_transform, local_match, "local_structure_index", strict=True)
                        fast.timings["index_query"] = index_query_ms
                        fast.timings["index_coarse"] = index_coarse_ms
                        fast.timings["index_fine"] = index_fine_ms
                        fast.timings["index_feature"] = index_feature_ms
                        fast.timings["index_candidate_source"] = first.source
                        fast.timings["index_feature_score"] = first.feature_score
                        fast.timings["index_coarse_score"] = index_coarse_score
                        fast.timings["index_coarse_spatial_margin"] = index_coarse_spatial_margin
                        fast.timings["index_fine_scale_count"] = index_fine_scale_count
                        fast.timings["index_refine"] = index_refine_ms
                        fast.validation["index_candidate_count"] = index_candidate_count
                        fast.validation["index_history_rejected_count"] = index_history_rejected_count
                        if fast.accepted:
                            if fast.validation.get("credible_geometry_confirmed"):
                                # The common verifier used screenshot-owned negatives;
                                # the old fog ring is not additional empty-space evidence.
                                return fast
                            known_count, background_conflict = _known_background_conflict(
                                local_transform, observation.known_background_mask,
                                (x, y), prepared.processed,
                            )
                            fast.validation["known_background_pixels"] = known_count
                            fast.validation["known_background_conflict_rate"] = background_conflict
                            if (known_count >= 500 and background_conflict is not None
                                    and background_conflict <= float(
                                        config["local_structure_index_max_background_conflict"]
                                    )):
                                return fast
                            index_skip_reason = "known_background_conflict"
                        else:
                            index_skip_reason = "refinement_failed"
                    else:
                        index_skip_reason = "refinement_failed"
                else:
                    index_skip_reason = "ambiguous_candidates"
            else:
                index_skip_reason = (
                    "coarse_low_score" if index_fine_ms == 0.0 else "low_candidate_score"
                )

        if (prior_screenshot_to_clean is None and visible_pixels < 20000
                and (visible_pixels < 5000 or not config.get("sparse_pose_review_enabled", True))
                and (index_coarse_spatial_margin is None
                     or index_coarse_spatial_margin < 0.08)):
            # The sparse uniqueness rule below would reject every global pose
            # here, regardless of its score. Avoid an expensive futile search.
            return RuntimeRegistrationResult(
                accepted=False,
                rejection_reasons=["首次稀疏配准的位置证据尚未确认，请继续探索"],
                clean_to_screen=None,
                screenshot_to_clean=None,
                observation_bbox=bbox,
                match=None,
                validation={"visible_structure_pixels": visible_pixels,
                            "visible_edge_pixels": visible_edges,
                            "sparse_spatial_margin": index_coarse_spatial_margin,
                            "sparse_margin_status": index_margin_status,
                            "sparse_spatial_rejected": True,
                            "index_candidate_count": index_candidate_count,
                            "pending_hypothesis_count": (
                                len(temporal_evidence.pending_poses.transforms)
                                if temporal_evidence is not None
                                and temporal_evidence.pending_poses is not None else 0)},
                timings={"method": "sparse_ambiguous", "segmentation": segmentation_ms,
                         "crop_and_evidence": crop_ms, "index_query": index_query_ms,
                         "index_coarse": index_coarse_ms, "index_fine": index_fine_ms,
                         "index_feature": index_feature_ms,
                         "total": (time.perf_counter() - total_started) * 1000.0},
                observation_image=observation_image,
                observation_mask=observation.map_mask,
                alignment_preview=None,
                observation_segmentation=observation,
            )

        if config.get("first_sparse_candidate_pool_enabled", False):
            config["first_sparse_initial_seeds"] = [
                (candidate.position, candidate.scale) for candidate in index_candidates[:6]
                if candidate.score >= max(.72, index_candidates[0].score - .08)
            ]
            if config.get("first_sparse_geometry_enabled", False):
                geometry = getattr(prepared, "_geometric_pair_index", None)
                if geometry is None:
                    geometry = GeometricPairIndex.build(
                        prepared.local_structure_index.feature_channels,
                        prepared.local_structure_index.factor)
                    prepared._geometric_pair_index = geometry
                features = prepared.local_structure_index._observation_features(search_observation)
                geometric_candidates = geometry.query(
                    features, prepared.local_structure_index.factor,
                    scale_min=config["scale_min"], scale_max=config["scale_max"])
                config["first_sparse_initial_seeds"].extend(
                    (item["position"], item["scale"]) for item in geometric_candidates)
        matched = match_single_frame_cpu_pyramid(
            template=prepared.processed,
            pyramid_template=prepared.pyramid,
            observation=search_observation,
            cfg=config,
            pyramid_factor=float(config["cpu_pyramid_factor"]),
            pyramid_template_once_ms=prepared.pyramid_ms,
        )
        result = matched.result
        if result.best_position is None or result.best_scale is None:
            return RuntimeRegistrationResult(
                accepted=False,
                rejection_reasons=["结构匹配没有返回位置或尺度"],
                clean_to_screen=None,
                screenshot_to_clean=None,
                observation_bbox=bbox,
                match=matched,
                validation={},
                timings={
                    "segmentation": segmentation_ms,
                    "crop_and_evidence": crop_ms,
                    "structure_match": matched.timing.total_match_ms,
                    "total": (time.perf_counter() - total_started) * 1000.0,
                },
                observation_image=observation_image,
                observation_mask=observation.map_mask,
                alignment_preview=None,
            )

        scale = float(result.best_scale)
        screenshot_to_processed = np.float32(
            [
                [scale, 0.0, float(result.best_position[0]) - scale * x],
                [0.0, scale, float(result.best_position[1]) - scale * y],
            ]
        )
        if (config.get("sparse_pose_review_enabled", True) and prior_screenshot_to_clean is None
                and 5000 <= visible_pixels < 30000
                and (index_coarse_spatial_margin is None or index_coarse_spatial_margin < .08
                     or scale*scale < .65)):
            local_pose = screenshot_to_processed.copy()
            local_pose[:, 2] += local_pose[:, :2] @ np.float32([x, y])
            alternatives = [np.float32([[c.scale, 0, c.top_left[0]], [0, c.scale, c.top_left[1]]])
                            for c in matched.result.top_k_candidates[:12]]
            factor = float(config["cpu_pyramid_factor"])
            alternatives.extend(np.float32([[c.scale, 0, c.top_left[0]/factor],
                                              [0, c.scale, c.top_left[1]/factor]])
                                for c in matched.coarse_result.top_k_candidates)
            alternatives.extend(np.float32([[c.scale, 0, c.position[0]], [0, c.scale, c.position[1]]])
                                for c in index_candidates)
            refined, sparse_review = review_sparse_pose(
                prepared.processed, observation, local_pose, alternatives,
                scale_min=config["scale_min"], scale_max=config["scale_max"],
                viewport=screenshot.shape[:2], origin=(x, y),
            )
            proposal = refined.copy()
            proposal[:, 2] -= proposal[:, :2] @ np.float32([x, y])
            quality = _validation(proposal, segmentation, prepared.processed)
            sparse_review["refined_validation"] = quality
            sparse_review["proposed_screenshot_to_processed"] = proposal.tolist()
            if (sparse_review["position_status"] == "confirmed"
                    and sparse_review["scale"]["status"] == "confirmed"
                    and quality["mapped_structure_fraction"] >= .95
                    and quality["structure_precision"] >= .97 and quality["semantic_precision"] >= .90
                    and quality["edge_median_error_px"] is not None
                    and quality["edge_median_error_px"] <= MAX_EDGE_MEDIAN_ERROR_PX):
                screenshot_to_processed = refined_transform = proposal
                matched = replace(matched, result=replace(
                    matched.result, best_position=tuple(float(v) for v in refined[:, 2]),
                    best_scale=float(refined[0, 0])))
        selected = finish(screenshot_to_processed, matched, "global_search", strict=False)
        selected.timings["index_query"] = index_query_ms
        selected.timings["index_coarse"] = index_coarse_ms
        selected.timings["index_fine"] = index_fine_ms
        selected.timings["index_feature"] = index_feature_ms
        selected.timings["index_coarse_score"] = index_coarse_score
        selected.timings["index_coarse_spatial_margin"] = index_coarse_spatial_margin
        selected.timings["index_fine_scale_count"] = index_fine_scale_count
        selected.timings["index_refine"] = index_refine_ms
        selected.timings["index_candidate_count"] = index_candidate_count
        selected.timings["index_skip_reason"] = index_skip_reason
        if (config.get("first_sparse_candidate_pool_enabled", False)
                and prior_screenshot_to_clean is None and selected.accepted):
            center = np.float32([width / 2, height / 2])
            winner_center = np.float32(result.best_position) + float(result.best_scale) * center
            for competitor in result.top_k_candidates[1:]:
                competitor_center = np.float32(competitor.top_left) + competitor.scale * center
                if (competitor.score < result.top_k_candidates[0].score - .035
                        or np.linalg.norm(competitor_center - winner_center) <= 35):
                    continue
                alternative = np.float32([
                    [competitor.scale, 0, competitor.top_left[0] - competitor.scale * x],
                    [0, competitor.scale, competitor.top_left[1] - competitor.scale * y]])
                quality = _validation(alternative, segmentation, prepared.processed)
                edge = quality["edge_median_error_px"]
                if (quality["structure_precision"] >= .95 and quality["semantic_precision"] >= .90
                        and quality["mapped_structure_fraction"] >= .80
                        and edge is not None and edge <= 4):
                    selected.accepted = False
                    selected.validation["spatial_competitor_rejected"] = True
                    selected.rejection_reasons.append("候选池中存在不同位置的高质量配准，请继续探索")
                    return selected
        if selected.accepted:
            if (prior_screenshot_to_clean is None and visible_pixels < 50000
                    and index_reference is not None):
                alternative, alternative_quality = index_reference
                center = np.float32([x + width / 2.0, y + height / 2.0])
                distance = float(np.linalg.norm(
                    alternative[:, :2] @ center + alternative[:, 2]
                    - screenshot_to_processed[:, :2] @ center - screenshot_to_processed[:, 2]))
                selected.validation["index_global_center_distance_px"] = distance
                alt_edge = alternative_quality["edge_median_error_px"]
                if (distance > 35.0
                        and alternative_quality["structure_precision"] >= 0.95
                        and alternative_quality["semantic_precision"] >= 0.90
                        and alt_edge is not None and alt_edge <= 5.0
                        and alternative_quality["structure_precision"]
                        >= selected.validation["structure_precision"] - 0.03
                        and alternative_quality["semantic_precision"]
                        >= selected.validation["semantic_precision"] - 0.03
                        and alt_edge <= selected.validation["edge_median_error_px"] + 1.5):
                    selected.accepted = False
                    selected.validation["spatial_competitor_rejected"] = True
                    selected.rejection_reasons.append("局部索引与全局搜索给出不同的高质量位置，请继续探索")
            if selected.accepted:
                return selected
            if selected.validation.get("spatial_competitor_rejected"):
                return selected

        if selected.validation.get("sparse_scale_rejected") and selected.validation.get("credible_geometry"):
            # All coarse/fine/index proposals already received the common review.
            # A second template review cannot resolve its measured scale uncertainty.
            return selected

        # With a small revealed area, the highest correlation can be a repeated
        # room or corridor. Review the already-found alternatives against the
        # complete semantic masks and edges before rejecting the screenshot.
        candidates = result.top_k_candidates[:12]
        if not candidates or visible_pixels < 5000:
            return selected
        review_started = time.perf_counter()
        reviewed: list[tuple[object, np.ndarray, dict, bool]] = []
        for candidate in candidates:
            scale = float(candidate.scale)
            transform = np.float32([
                [scale, 0.0, float(candidate.top_left[0]) - scale * x],
                [0.0, scale, float(candidate.top_left[1]) - scale * y],
            ])
            validation = _validation(transform, segmentation, prepared.processed)
            edge = validation["edge_median_error_px"]
            normal = (validation["observed_structure_pixels"] >= 5000
                      and validation["structure_precision"] >= 0.78
                      and validation["semantic_precision"] >= 0.65
                      and edge is not None and edge <= MAX_EDGE_MEDIAN_ERROR_PX)
            sparse = (visible_edges >= 150 and candidate.score >= 0.52
                      and validation["observed_structure_pixels"] >= 1500
                      and validation["structure_precision"] >= 0.95
                      and validation["semantic_precision"] >= 0.90
                      and edge is not None and edge <= 4.0)
            if (normal or sparse) and temporal_evidence is not None and temporal_evidence.matches(prepared):
                _compared, conflict = temporal_evidence.conflict(
                    corridor=observation.corridor_mask, room=observation.room_mask,
                    background=observation.known_background_mask,
                    transform=transform, origin=(x, y),
                )
                if conflict is not None and conflict > 0.20:
                    normal = False
                    sparse = False
            if normal or sparse:
                reviewed.append((candidate, transform, validation, sparse and not normal))
        review_ms = (time.perf_counter() - review_started) * 1000.0
        selected.timings["candidate_review"] = review_ms
        selected.timings["total"] = (time.perf_counter() - total_started) * 1000.0
        if not reviewed:
            return selected
        winner, winner_transform, winner_validation, sparse_unique = reviewed[0]
        center = observation_anchor(observation)
        winner_point = np.float32(winner.top_left) + float(winner.scale) * center
        scale_only_consensus = False
        scale_spread = 0.0
        has_close_zoom_competitor = any(
            item[0].score >= winner.score - 0.035
            and abs(float(item[0].scale / winner.scale) - 1.0) > 0.04
            for item in reviewed[1:])
        if (prior_screenshot_to_clean is None and 15000 <= visible_pixels < 30000
                and has_close_zoom_competitor):
            direction = _edge_direction_validation(winner_transform, segmentation, prepared)
            scale_only_consensus, scale_spread = _same_location_scale_consensus(
                reviewed, (height, width), winner_validation, direction, visible_pixels)
        for competitor, _transform, _validation_result, _is_sparse in reviewed[1:]:
            competitor_point = np.float32(competitor.top_left) + float(competitor.scale) * center
            if not (sparse_unique or competitor.score >= winner.score - 0.035):
                continue
            center_distance = float(np.linalg.norm(competitor_point - winner_point))
            scale_difference = abs(float(competitor.scale / winner.scale) - 1.0)
            if center_distance > 35.0:
                selected.rejection_reasons.append("可见结构对应多个可能位置，请继续探索")
                return selected
            tolerated_zoom = (scale_only_consensus and center_distance <= 10.0
                              and scale_difference <= 0.08
                              and competitor.score >= winner.score - 0.035)
            if scale_difference > 0.04 and not tolerated_zoom:
                selected.rejection_reasons.append("同一位置存在不同缩放比例，请继续探索")
                return selected
        reviewed_match = replace(
            matched,
            result=replace(matched.result, best_position=winner.top_left, best_scale=winner.scale),
        )
        mode = "global_sparse_unique" if sparse_unique else "global_candidate_review"
        chosen = finish(winner_transform, reviewed_match, mode, strict=False,
                        sparse_unique=sparse_unique,
                        scale_only_consensus=scale_only_consensus)
        if scale_only_consensus:
            chosen.validation["scale_candidate_spread"] = scale_spread
        chosen.validation["reviewed_candidate_count"] = len(candidates)
        chosen.validation["eligible_candidate_count"] = len(reviewed)
        chosen.timings["candidate_review"] = review_ms
        chosen.timings["index_query"] = index_query_ms
        chosen.timings["index_coarse"] = index_coarse_ms
        chosen.timings["index_fine"] = index_fine_ms
        chosen.timings["index_feature"] = index_feature_ms
        chosen.timings["index_coarse_score"] = index_coarse_score
        chosen.timings["index_coarse_spatial_margin"] = index_coarse_spatial_margin
        chosen.timings["index_fine_scale_count"] = index_fine_scale_count
        chosen.timings["index_refine"] = index_refine_ms
        chosen.timings["index_candidate_count"] = index_candidate_count
        chosen.timings["index_skip_reason"] = index_skip_reason
        return chosen if chosen.accepted else selected
    except Exception as exc:
        return RuntimeRegistrationResult(
            accepted=False,
            rejection_reasons=[f"配准异常：{exc}"],
            clean_to_screen=None,
            screenshot_to_clean=None,
            observation_bbox=None,
            match=None,
            validation={},
            timings={"total": (time.perf_counter() - total_started) * 1000.0},
            observation_image=None,
            observation_mask=None,
            alignment_preview=None,
        )
