"""Balanced spatial evidence and pose ambiguity diagnostics.

Scores are geometric support, not calibrated probabilities. Nearby grid cells
are not independent observations, and a single high score never proves uniqueness.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .template_preprocessor import ProcessedTemplate


@dataclass
class RegionalEvidence:
    points: np.ndarray
    labels: np.ndarray
    regions: np.ndarray
    edges: np.ndarray
    edge_regions: np.ndarray
    center: np.ndarray
    region_count: int
    span: tuple[float, float]


def build_regional_evidence(observation: SegmentationResult) -> RegionalEvidence:
    """Sample equally from occupied cells of the *visible* bounding box."""
    yy, xx = np.nonzero(observation.map_mask)
    if not len(xx):
        return RegionalEvidence(np.empty((0, 2), np.float32), np.empty(0, int),
                                np.empty(0, int), np.empty((0, 2), np.float32),
                                np.empty(0, int), np.zeros(2, np.float32), 0, (0., 0.))
    left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
    width, height = right-left, bottom-top
    cells = np.minimum(3, (xx-left)*4//width) + 4*np.minimum(3, (yy-top)*4//height)
    ey, ex = np.nonzero(observation.edge_mask)
    edge_cells = (np.clip((ex-left)*4//width, 0, 3)
                  + 4*np.clip((ey-top)*4//height, 0, 3))
    points, labels, regions, edges, edge_regions = [], [], [], [], []
    for cell in np.unique(cells):
        indices = np.flatnonzero(cells == cell)
        edge_indices = np.flatnonzero(edge_cells == cell)
        if len(indices) < 80 or len(edge_indices) < 8:
            continue
        region = len(points)
        chosen = indices[np.linspace(0, len(indices)-1, min(96, len(indices)), dtype=int)]
        chosen_edge = edge_indices[np.linspace(0, len(edge_indices)-1,
                                               min(48, len(edge_indices)), dtype=int)]
        points.append(np.column_stack((xx[chosen], yy[chosen])))
        labels.append((observation.room_mask[yy[chosen], xx[chosen]] > 0).astype(int))
        regions.extend([region]*len(chosen))
        edges.append(np.column_stack((ex[chosen_edge], ey[chosen_edge])))
        edge_regions.extend([region]*len(chosen_edge))
    return RegionalEvidence(
        np.concatenate(points).astype(np.float32) if points else np.empty((0, 2), np.float32),
        np.concatenate(labels) if labels else np.empty(0, int), np.asarray(regions),
        np.concatenate(edges).astype(np.float32) if edges else np.empty((0, 2), np.float32),
        np.asarray(edge_regions), np.float32([xx.mean(), yy.mean()]), len(points),
        (float(width), float(height)),
    )


def warm_regional_template(template: ProcessedTemplate) -> np.ndarray:
    semantic = getattr(template, "_regional_semantic_fields", None)
    if semantic is None:
        kernel = np.ones((5, 5), np.uint8)
        semantic = np.stack([cv2.dilate(template.corridor_mask, kernel),
                             cv2.dilate(template.room_mask, kernel)], axis=2).astype(np.float32)/255
        template._regional_semantic_fields = semantic
    return semantic


def regional_support(template: ProcessedTemplate, evidence: RegionalEvidence,
                     transforms: list[np.ndarray]) -> list[dict]:
    """Use all cells with equal weight, including points outside the template."""
    if not evidence.region_count:
        return [{"region_count": 0, "balanced_score": 0., "regions": []} for _ in transforms]
    semantic = warm_regional_template(template)
    matrices = np.asarray(transforms, np.float32)
    def coordinates(points):
        mapped = np.einsum("nij,pj->npi", matrices[:, :, :2], points) + matrices[:, None, :, 2]
        return np.ascontiguousarray(mapped[:, :, 0]), np.ascontiguousarray(mapped[:, :, 1])
    sx, sy = coordinates(evidence.points)
    values = cv2.remap(semantic, sx, sy, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    values = values[:, np.arange(len(evidence.points)), evidence.labels]
    ex, ey = coordinates(evidence.edges)
    distances = cv2.remap(template.distance_transform, ex, ey, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1000)
    scales = np.sqrt(np.abs(np.linalg.det(matrices[:, :, :2])))
    distances /= np.maximum(scales[:, None], .01)
    # Evaluate each cell across the whole pose batch; avoid rebuilding a mask
    # and allocating a separate sample vector for every pose/cell combination.
    semantics_by_region = np.stack([
        values[:, evidence.regions == region].mean(axis=1)
        for region in range(evidence.region_count)], axis=1)
    edges_by_region = np.stack([
        np.median(distances[:, evidence.edge_regions == region], axis=1)
        for region in range(evidence.region_count)], axis=1)
    scores_by_region = .5*semantics_by_region + .5*np.exp(-edges_by_region/4.)
    results = []
    for index in range(len(transforms)):
        regions = []
        for region in range(evidence.region_count):
            sem = float(semantics_by_region[index, region])
            edge = float(edges_by_region[index, region])
            regions.append({"semantic": sem, "edge_median_px": edge,
                            "score": float(scores_by_region[index, region])})
        results.append({"region_count": len(regions), "regions": regions,
                        "balanced_score": float(np.mean([r["score"] for r in regions])),
                        "supported_regions": sum(r["semantic"] >= .85 and r["edge_median_px"] <= 4 for r in regions),
                        "contradictory_regions": sum(r["semantic"] < .55 and r["edge_median_px"] > 8 for r in regions)})
    return results


def analyze_pose(template: ProcessedTemplate, evidence: RegionalEvidence,
                 transform: np.ndarray, alternatives: list[np.ndarray]) -> dict:
    """Separate alternative locations from scale uncertainty around the visible anchor."""
    center = evidence.center
    anchor = transform[:, :2] @ center + transform[:, 2]
    poses = [transform]
    pose_key = lambda pose: tuple(np.rint(np.asarray(pose, np.float64).ravel()*10000).astype(np.int64))
    seen = {pose_key(transform)}
    for candidate in alternatives:
        key = pose_key(candidate)
        if key not in seen:
            seen.add(key)
            poses.append(candidate)
    # Profile scale at the visible anchor in the same batch as all candidates.
    # This is local sensitivity, not a confidence interval or proof of uniqueness.
    ratios = (.96, .98, .99, .995, 1., 1.005, 1.01, 1.02, 1.04)
    probes = []
    for ratio in ratios:
        probe = transform.copy()
        probe[:, :2] *= ratio
        probe[:, 2] = anchor-probe[:, :2] @ center
        probes.append(probe)
    combined_support = regional_support(template, evidence, poses+probes)
    support, profile = combined_support[:len(poses)], combined_support[len(poses):]
    winner = support[0]
    spatial, scale_only, decisive = [], [], 0
    strong_competitors = 0
    groups = [{"anchor": anchor, "indices": [0]}]
    for index, (pose, candidate) in enumerate(zip(poses[1:], support[1:]), start=1):
        distance = np.linalg.norm(pose[:, :2] @ center + pose[:, 2] - anchor)
        if candidate["balanced_score"] < winner["balanced_score"]-.06:
            continue
        candidate_anchor = pose[:, :2] @ center+pose[:, 2]
        group = next((g for g in groups if np.linalg.norm(candidate_anchor-g["anchor"]) <= 35), None)
        if group is None:
            groups.append({"anchor": candidate_anchor, "indices": [index]})
        else:
            group["indices"].append(index)
        (spatial if distance > 35 else scale_only).append(pose)
        if distance > 35:
            advantages = [a["score"]-b["score"] for a, b in zip(winner["regions"], candidate["regions"])]
            decisive = max(decisive, sum(value < -.12 for value in advantages))
            if (candidate["region_count"] >= 6
                    and candidate["balanced_score"] > winner["balanced_score"]+.06
                    and candidate.get("supported_regions", 0) >= .7*candidate["region_count"]
                    and candidate.get("contradictory_regions", 0) == 0
                    and sum(value < -.12 for value in advantages) >= 2):
                strong_competitors += 1
    plausible = [ratio for ratio, item in zip(ratios, profile)
                 if item["balanced_score"] >= max(.65, winner["balanced_score"]-.025)]
    span = float(max(plausible)-min(plausible)) if plausible else 0.
    group_details = []
    for group in groups:
        scales = [float(np.sqrt(abs(np.linalg.det(poses[i][:, :2])))) for i in group["indices"]]
        group_details.append({"anchor": group["anchor"].tolist(), "candidate_count": len(scales),
                              "scale_min": min(scales), "scale_max": max(scales),
                              "max_support": max(support[i]["balanced_score"] for i in group["indices"])})
    return {**winner, "spatial_competitors": len(groups)-1, "pose_groups": group_details,
            "same_anchor_competitors": len(scale_only), "competitor_advantage_regions": decisive,
            "strong_spatial_competitors": strong_competitors,
            "scale_profile_span": span,
            "scale_profile_boundary": bool(plausible and (min(plausible) == ratios[0] or max(plausible) == ratios[-1])),
            "scale_profile": [{"relative_scale": ratio, "score": item["balanced_score"]}
                              for ratio, item in zip(ratios, profile)],
            "scale_profile_anchor": evidence.center.tolist(),
            "plausible_relative_scales": plausible,
            "visible_span_px": list(evidence.span),
            "uncertainty_kind": "sampled_local_support_not_error"}
