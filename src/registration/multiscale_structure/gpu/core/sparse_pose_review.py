"""Separate sparse location evidence from bounded wall-based zoom estimation.

Candidate scores from different search engines are never compared directly.
All locations receive the same refinement and regional scoring. Local support
and scale profiles are diagnostic evidence, not calibrated probabilities.
"""
from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .pose_evidence import build_regional_evidence, regional_support
from .sparse_pose_refiner import balanced_points, refine_sparse_poses


def observation_anchor(observation) -> np.ndarray:
    moments = cv2.moments(observation.map_mask, binaryImage=True)
    if not moments["m00"]:
        return np.float32([observation.map_mask.shape[1]/2, observation.map_mask.shape[0]/2])
    return np.float32([moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]])


def group_poses(poses: list[np.ndarray], center: np.ndarray, radius: float = 35) -> list[list[int]]:
    groups: list[list[int]] = []
    anchors = [pose[:, :2] @ center + pose[:, 2] for pose in poses]
    for index, anchor in enumerate(anchors):
        group = next((g for g in groups if np.linalg.norm(anchor-anchors[g[0]]) <= radius), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    return groups


def _straight_walls(mask: np.ndarray, minimum: int = 16) -> tuple[np.ndarray, np.ndarray]:
    edge = cv2.Canny(mask, 40, 120)
    # Long collinear support removes small icons and most curved reveal rims.
    # It does not prove a wall is physical; geometry checks remain mandatory.
    return (cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((1, minimum), np.uint8)),
            cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((minimum, 1), np.uint8)))


def refine_anchor_scale(template, observation, pose: np.ndarray, *,
                        scale_min: float, scale_max: float, viewport: tuple[int, int],
                        origin: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Profile zoom at a fixed visible anchor using horizontal/vertical walls."""
    masks = _straight_walls(observation.map_mask)
    points = [balanced_points(mask, 320)[0] for mask in masks]
    diagnostics = {"status": "insufficient_walls", "wall_points": [len(p) for p in points]}
    if (min(len(p) for p in points) < 24
            or np.ptp(points[0][:, 1]) < 35 or np.ptp(points[1][:, 0]) < 25):
        return pose, diagnostics
    cached = getattr(template, "_straight_wall_fields", None)
    if cached is None:
        walls = _straight_walls(template.map_mask, minimum=10)
        cached = np.stack([cv2.distanceTransform(255-mask, cv2.DIST_L2, 3) for mask in walls], axis=2)
        template._straight_wall_fields = cached
    center = observation_anchor(observation)
    anchor = pose[:, :2] @ center + pose[:, 2]
    initial_scale = float(np.sqrt(abs(np.linalg.det(pose[:, :2]))))
    scales = initial_scale * np.linspace(.92, 1.08, 65, dtype=np.float32)
    scales = scales[(scales >= scale_min) & (scales <= scale_max)]
    if len(scales) < 5:
        return pose, {**diagnostics, "status": "scale_range_too_small"}
    poses = np.zeros((len(scales), 2, 3), np.float32)
    poses[:, 0, 0] = poses[:, 1, 1] = scales
    poses[:, :, 2] = anchor - scales[:, None]*center
    all_distances = []
    orientation_scores = []
    for channel, selected in enumerate(points):
        xy = (selected[None]-center) * scales[:, None, None] + anchor
        values = cv2.remap(cached[:, :, channel], np.ascontiguousarray(xy[:, :, 0]),
                           np.ascontiguousarray(xy[:, :, 1]), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=1000) / scales[:, None]
        all_distances.append(values)
        orientation_scores.append(np.exp(-values/3).mean(axis=1))
    wall_score = np.mean(orientation_scores, axis=0)
    evidence = build_regional_evidence(observation)
    support = regional_support(template, evidence, list(poses))
    semantic = np.asarray([np.mean([r["semantic"] for r in item["regions"]]) if item["regions"] else 0
                           for item in support])
    scores = .85*wall_score + .15*semantic
    best = int(scores.argmax())
    plausible = np.flatnonzero(scores >= scores[best]-.012)
    distances = np.concatenate(all_distances, axis=1)[best]
    height, width = viewport
    xx, yy = np.meshgrid(np.linspace(0, width-1, 5), np.linspace(0, height-1, 5))
    screen_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32) - origin
    clean_points = screen_points @ poses[best, :, :2].T + poses[best, :, 2]
    disagreements = []
    for index in plausible:
        inverse = cv2.invertAffineTransform(poses[index])
        projected = clean_points @ inverse[:, :2].T + inverse[:, 2]
        disagreements.append(float(np.percentile(np.linalg.norm(projected-screen_points, axis=1), 95)))
    boundary = bool(0 in plausible or len(scales)-1 in plausible)
    stable = (not boundary and max(disagreements) <= 8.0
              and float(np.median(distances)) <= 2.0
              and float(np.percentile(distances, 90)) <= 5.0
              and float(np.mean(distances <= 4)) >= .80
              and semantic[best] >= .95)
    diagnostics.update(
        status="confirmed" if stable else "uncertain", selected_scale=float(scales[best]),
        anchor=center.tolist(), mapped_anchor=anchor.tolist(),
        wall_median_screen_px=float(np.median(distances)),
        wall_p90_screen_px=float(np.percentile(distances, 90)),
        wall_hit_fraction=float(np.mean(distances <= 4)),
        viewport_disagreement_p95_px=max(disagreements), boundary=boundary,
        plausible_scale_min=float(scales[plausible].min()), plausible_scale_max=float(scales[plausible].max()),
        profile=[{"scale": float(s), "score": float(v)} for s, v in zip(scales, scores)],
    )
    return poses[best], diagnostics


def review_sparse_pose(template, observation, winner: np.ndarray, alternatives: list[np.ndarray], *,
                       scale_min: float, scale_max: float, viewport: tuple[int, int],
                       origin: tuple[int, int]) -> tuple[np.ndarray, dict]:
    started = perf_counter()
    center = observation_anchor(observation)
    evidence = build_regional_evidence(observation)
    poses = [winner, *alternatives]
    groups = group_poses(poses, center)
    supports = regional_support(template, evidence, poses)
    group_results = []
    for group in groups[:12]:
        ranked = sorted(group, key=lambda i: supports[i]["balanced_score"], reverse=True)
        seeds = [ranked[0]]
        for index in ranked[1:]:
            if abs(float(poses[index][0, 0] / poses[seeds[0]][0, 0])-1) >= .025:
                seeds.append(index)
                break
        refined, _ = refine_sparse_poses(
            template, observation, [(tuple(poses[i][:, 2]), float(poses[i][0, 0])) for i in seeds],
            scale_min=scale_min, scale_max=scale_max, position_radius=8, scale_radius=.035,
            max_seeds=2,
        )
        candidates = [poses[i] for i in group]
        candidates.extend(np.float32([[c.scale, 0, c.position[0]], [0, c.scale, c.position[1]]]) for c in refined)
        quality = regional_support(template, evidence, candidates)
        best = max(range(len(candidates)), key=lambda i: quality[i]["balanced_score"])
        group_results.append((candidates[best], quality[best], len(group)))
    primary, support, _ = group_results[0]
    competing = [item for item in group_results[1:]
                 if item[1]["balanced_score"] >= support["balanced_score"]-.08]
    position_confirmed = (len(groups) <= 12 and len(groups) >= 2 and not competing
                          and support["region_count"] >= 6 and support["balanced_score"] >= .70
                          and support["contradictory_regions"] == 0
                          and support["supported_regions"] >= .5*support["region_count"])
    result = {
        "position_status": "confirmed" if position_confirmed else "ambiguous" if competing else "insufficient_evidence",
        "spatial_group_count": len(groups), "competitive_spatial_groups": len(competing),
        "search_truncated": len(groups) > 12,
        "groups": [{"anchor": (p[:, :2]@center+p[:, 2]).tolist(), "candidate_count": count,
                    "scale": float(p[0, 0]), **s} for p, s, count in group_results],
        "scale": {"status": "not_evaluated"},
    }
    if position_confirmed:
        primary, result["scale"] = refine_anchor_scale(
            template, observation, primary, scale_min=scale_min, scale_max=scale_max,
            viewport=viewport, origin=origin,
        )
    result["elapsed_ms"] = (perf_counter()-started)*1000
    return primary, result
