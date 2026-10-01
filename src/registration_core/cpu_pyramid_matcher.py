from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from time import perf_counter

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .masked_matcher import LocalizationResult, match_single_frame, review_pose_candidates
from .template_preprocessor import ProcessedTemplate
from .sparse_pose_refiner import refine_sparse_poses
from .pose_evidence import build_regional_evidence, regional_support


@dataclass
class CpuPyramidTiming:
    pyramid_template_once_ms: float
    observation_pyramid_ms: float
    coarse_match_ms: float
    fine_match_ms: float
    total_match_ms: float
    coarse_scale_count: int
    fine_scale_count: int
    sparse_refinement: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class CpuPyramidMatchResult:
    result: LocalizationResult
    coarse_result: LocalizationResult
    timing: CpuPyramidTiming


def _resize_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(mask, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def build_cpu_pyramid_template(
    template: ProcessedTemplate,
    *,
    factor: float = 0.5,
) -> tuple[ProcessedTemplate, float]:
    """Build the reusable low-resolution template used by coarse CPU search."""
    factor = max(0.2, min(1.0, float(factor)))
    started = perf_counter()
    corridor = _resize_mask(template.corridor_mask, factor)
    room = _resize_mask(template.room_mask, factor)
    map_mask = _resize_mask(template.map_mask, factor)
    edge = _resize_mask(template.edge_mask, factor)
    stairs = _resize_mask(template.stair_mask, factor)
    pyramid = ProcessedTemplate(
        map_id=f"{template.map_id}_cpu_pyramid_{factor:g}",
        image=cv2.resize(template.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA),
        corridor_mask=corridor,
        room_mask=room,
        map_mask=map_mask,
        edge_mask=edge,
        stair_mask=stairs,
        distance_transform=cv2.distanceTransform(cv2.bitwise_not(edge), cv2.DIST_L2, 3),
        crop_offset=(
            int(round(template.crop_offset[0] * factor)),
            int(round(template.crop_offset[1] * factor)),
        ),
        raw_path=template.raw_path,
        corridor_float=corridor.astype(np.float32) / 255.0,
        room_float=room.astype(np.float32) / 255.0,
    )
    return pyramid, (perf_counter() - started) * 1000.0


def _downsample_observation(
    observation: SegmentationResult,
    factor: float,
) -> SegmentationResult:
    values: dict[str, np.ndarray | None] = {}
    for item in fields(SegmentationResult):
        value = getattr(observation, item.name)
        values[item.name] = None if value is None else _resize_mask(value, factor)
    return SegmentationResult(**values)


def _regular_scales(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = max(1e-4, float(step))
    if stop < start:
        start, stop = stop, start
    return np.arange(start, stop + step * 0.25, step, dtype=np.float32).round(4).tolist()


def match_single_frame_cpu_pyramid(
    *,
    template: ProcessedTemplate,
    pyramid_template: ProcessedTemplate,
    observation: SegmentationResult,
    cfg: dict,
    pyramid_factor: float = 0.5,
    pyramid_template_once_ms: float = 0.0,
) -> CpuPyramidMatchResult:
    """Run coarse-to-fine structural registration using CPU OpenCV operations.

    The coarse stage searches scale and position on a reduced image. The fine
    stage searches full-resolution data only around the coarse transform.
    """
    total_started = perf_counter()
    pyramid_factor = max(0.2, min(1.0, float(pyramid_factor)))
    pyramid_started = perf_counter()
    pyramid_observation = _downsample_observation(observation, pyramid_factor)
    observation_pyramid_ms = (perf_counter() - pyramid_started) * 1000.0

    scale_min = float(cfg.get("scale_min", 0.85))
    scale_max = float(cfg.get("scale_max", 1.15))
    coarse_step = float(cfg.get("cpu_pyramid_coarse_scale_step", 0.05))
    coarse_scales = _regular_scales(scale_min, scale_max, coarse_step)
    coarse_cfg = dict(cfg)
    coarse_cfg.pop("fixed_scale", None)
    coarse_cfg.update(
        {
            "scale_values": coarse_scales,
            "top_k": int(cfg.get("cpu_pyramid_coarse_top_k", 4)),
            "candidate_pool_size": int(cfg.get("cpu_pyramid_coarse_candidate_pool_size", 4)),
            "geometric_review_enabled": False,
            "pre_match_search_windows_enabled": False,
        }
    )
    if cfg.get("first_sparse_candidate_pool_enabled", False):
        coarse_cfg["top_k"] = max(12, coarse_cfg["top_k"])
        coarse_cfg["candidate_pool_size"] = max(12, coarse_cfg["candidate_pool_size"])
    coarse_started = perf_counter()
    coarse_result = match_single_frame(
        template=pyramid_template,
        observation=pyramid_observation,
        cfg=coarse_cfg,
        player_roi=None,
    )
    coarse_match_ms = (perf_counter() - coarse_started) * 1000.0
    if coarse_result.best_position is None or coarse_result.best_scale is None:
        timing = CpuPyramidTiming(
            pyramid_template_once_ms=float(pyramid_template_once_ms),
            observation_pyramid_ms=observation_pyramid_ms,
            coarse_match_ms=coarse_match_ms,
            fine_match_ms=0.0,
            total_match_ms=(perf_counter() - total_started) * 1000.0,
            coarse_scale_count=len(coarse_scales),
            fine_scale_count=0,
        )
        return CpuPyramidMatchResult(coarse_result, coarse_result, timing)

    coarse_scale = float(coarse_result.best_scale)
    fine_radius = max(0.0, float(cfg.get("cpu_pyramid_fine_scale_radius", 0.05)))
    fine_step = float(cfg.get("cpu_pyramid_fine_scale_step", 0.01))
    fine_scales = _regular_scales(
        max(scale_min, coarse_scale - fine_radius),
        min(scale_max, coarse_scale + fine_radius),
        fine_step,
    )

    observation_height, observation_width = observation.map_mask.shape
    anchor_observation = (observation_width / 2.0, observation_height / 2.0)
    coarse_x = float(coarse_result.best_position[0]) / pyramid_factor
    coarse_y = float(coarse_result.best_position[1]) / pyramid_factor
    crop_x, crop_y = template.crop_offset
    anchor_template = [
        coarse_x + coarse_scale * anchor_observation[0] + float(crop_x),
        coarse_y + coarse_scale * anchor_observation[1] + float(crop_y),
    ]

    fine_cfg = dict(cfg)
    fine_cfg.pop("fixed_scale", None)
    fine_cfg.update(
        {
            "scale_values": fine_scales,
            "candidate_pool_size": int(cfg.get("cpu_pyramid_fine_candidate_pool_size", 8)),
            "geometric_review_candidate_count": int(
                cfg.get("cpu_pyramid_fine_review_candidate_count", 16)
            ),
            "pre_match_search_windows_enabled": True,
            "allowed_player_template_points": [anchor_template],
            "allowed_player_template_radius": float(cfg.get("cpu_pyramid_fine_position_radius", 100.0)),
        }
    )
    sparse_diagnostics = {}
    wide_fine_cfg = dict(fine_cfg)
    if (cfg.get("first_sparse_optimized_enabled", False)
            and cfg.get("first_sparse_point_refinement_enabled", False)):
        refined, sparse_diagnostics = refine_sparse_poses(
            template, observation, [((coarse_x, coarse_y), coarse_scale)],
            scale_min=scale_min, scale_max=scale_max,
            position_radius=32.0, scale_radius=fine_radius,
        )
        sparse_diagnostics["proposals"] = [asdict(pose) for pose in refined]
        if (refined and refined[0].semantic >= 0.70 and refined[0].edge_hit >= 0.35
                and refined[0].background_conflict <= .15):
            proposal = refined[0]
            fine_cfg["allowed_player_template_points"] = [[
                proposal.position[0] + proposal.scale * anchor_observation[0] + float(crop_x),
                proposal.position[1] + proposal.scale * anchor_observation[1] + float(crop_y),
            ]]
            fine_cfg["allowed_player_template_radius"] = 16.0
            sparse_diagnostics["used"] = True
        else:
            sparse_diagnostics["used"] = False
            if refined and refined[0].background_conflict > .15:
                # A locally attractive wall with contradictory known-empty
                # space must not exclude the wider search's competing room.
                sparse_diagnostics["skip_reason"] = "known_background_conflict"
    fine_started = perf_counter()
    result = None
    # Experimental scheduler: every coarse scale/location remains searched.
    # Only a spatially isolated coarse winner can try smaller fine tiers.
    # A boundary result, regional contradiction or close coarse competitor
    # always falls through to the original full fine grid.
    trace = []
    if cfg.get("staged_fine_search_enabled", False) and not sparse_diagnostics.get("used"):
        coarse_anchor = np.float32([coarse_x, coarse_y]) + coarse_scale*np.float32(anchor_observation)
        competition = any(
            c.score >= coarse_result.top_k_candidates[0].score-.08
            and np.linalg.norm(np.float32(c.top_left)/pyramid_factor
                               + c.scale*np.float32(anchor_observation)-coarse_anchor) > 35
            for c in coarse_result.top_k_candidates[1:])
        if not competition:
            evidence = build_regional_evidence(observation)
            for radius, zoom_radius in ((24., .02), (48., .035)):
                tier_scales = [s for s in fine_scales if abs(s-coarse_scale) <= zoom_radius]
                if len(tier_scales) < 3:
                    continue
                tier_cfg = {**fine_cfg, "scale_values": tier_scales,
                            "allowed_player_template_radius": radius}
                tier_started = perf_counter()
                trial = match_single_frame(template=template, observation=observation,
                                           cfg=tier_cfg, player_roi=anchor_observation)
                best = trial.top_k_candidates[0] if trial.top_k_candidates else None
                accepted = False
                if best is not None:
                    center = np.float32(best.top_left) + best.scale*np.float32(anchor_observation)
                    pose = np.float32([[best.scale, 0, best.top_left[0]],
                                      [0, best.scale, best.top_left[1]]])
                    support = regional_support(template, evidence, [pose])[0]
                    accepted = bool(
                        np.max(np.abs(center-coarse_anchor)) < radius-4
                        and min(tier_scales)+fine_step*.5 < best.scale < max(tier_scales)-fine_step*.5
                        and (best.semantic_consistency or 0) >= .94
                        and (best.edge_precision or 0) >= .75
                        and support["region_count"] >= 6
                        and support.get("supported_regions", 0) >= .85*support["region_count"]
                        and not support.get("contradictory_regions", 0))
                trace.append({"radius": radius, "scale_count": len(tier_scales),
                              "accepted_proposal": accepted, "elapsed_ms": (perf_counter()-tier_started)*1000})
                if accepted:
                    result = trial
                    break
    if result is None:
        result = match_single_frame(
            template=template, observation=observation, cfg=fine_cfg, player_roi=anchor_observation)
        if trace:
            trace.append({"radius": fine_cfg["allowed_player_template_radius"],
                          "scale_count": len(fine_scales), "full_fallback": True})
    if trace:
        sparse_diagnostics["fine_search_tiers"] = trace
    if sparse_diagnostics.get("used"):
        best = result.top_k_candidates[0] if result.top_k_candidates else None
        if best is not None:
            center = np.float32(best.top_left) + best.scale * np.float32(anchor_observation)
            proposed_center = np.float32(fine_cfg["allowed_player_template_points"][0]) - np.float32([crop_x, crop_y])
            touches_boundary = np.max(np.abs(center - proposed_center)) >= 13
        else:
            touches_boundary = True
        if (best is None or touches_boundary or (best.semantic_consistency or 0) < .78
                or (best.edge_precision or 0) < .15):
            result = match_single_frame(template=template, observation=observation,
                                        cfg=wide_fine_cfg, player_roi=anchor_observation)
            sparse_diagnostics["wide_search_fallback"] = True
    if (cfg.get("first_sparse_optimized_enabled", False)
            and cfg.get("first_sparse_candidate_pool_enabled", False)
            and result.best_position is not None):
        pool_started = perf_counter()
        best_center = np.float32(result.best_position) + float(result.best_scale) * np.float32(anchor_observation)
        raw_seeds = [((candidate.top_left[0] / pyramid_factor, candidate.top_left[1] / pyramid_factor),
                      candidate.scale) for candidate in coarse_result.top_k_candidates
                     if candidate.score >= coarse_result.top_k_candidates[0].score - .06]
        raw_seeds.extend(cfg.get("first_sparse_initial_seeds", []))
        seeds = []
        centers = []
        for position, scale in raw_seeds:
            center = np.float32(position) + scale * np.float32(anchor_observation)
            if np.linalg.norm(center - best_center) < 35:
                continue
            if any(np.linalg.norm(center - previous) < 28 for previous in centers):
                continue
            seeds.append((position, scale))
            centers.append(center)
            if len(seeds) >= 5:
                break
        proposals, details = refine_sparse_poses(
            template, observation, seeds, scale_min=scale_min, scale_max=scale_max,
            position_radius=24, scale_radius=max(fine_radius, .06), max_seeds=5)
        proposed = [(pose.position, pose.scale) for pose in proposals if pose.semantic >= .70]
        alternatives = review_pose_candidates(template, observation, proposed,
                                               {**cfg, "proposal_requires_strong_geometry": True})
        if alternatives:
            combined = sorted([*result.top_k_candidates, *alternatives], key=lambda item: item.score, reverse=True)
            result.top_k_candidates = combined[:max(int(cfg.get("top_k", 12)), len(alternatives) + 1)]
            winner = result.top_k_candidates[0]
            result.best_position, result.best_scale = winner.top_left, winner.scale
            result.confidence = winner.score - result.top_k_candidates[1].score if len(combined) > 1 else winner.score
            result.low_confidence = result.confidence < float(cfg.get("confidence_threshold", .035))
        sparse_diagnostics["candidate_pool"] = {
            **details, "reviewed_alternatives": len(alternatives),
            "total_ms": (perf_counter() - pool_started) * 1000,
            "poses": [{"position": item.top_left, "scale": item.scale, "score": item.score}
                      for item in alternatives],
        }
    fine_match_ms = (perf_counter() - fine_started) * 1000.0
    # The center point is an internal search anchor, not an observed player marker.
    result.player_roi = None
    result.player_position_in_template = None
    result.player_position_in_template_crop = None
    timing = CpuPyramidTiming(
        pyramid_template_once_ms=float(pyramid_template_once_ms),
        observation_pyramid_ms=observation_pyramid_ms,
        coarse_match_ms=coarse_match_ms,
        fine_match_ms=fine_match_ms,
        total_match_ms=(perf_counter() - total_started) * 1000.0,
        coarse_scale_count=len(coarse_scales),
        fine_scale_count=len(fine_scales),
        sparse_refinement=sparse_diagnostics,
    )
    return CpuPyramidMatchResult(result, coarse_result, timing)
