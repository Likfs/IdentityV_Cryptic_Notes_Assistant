"""CPU point-to-distance-field refinement for partial semantic map observations.

Unknown fog never supplies negative evidence. A local optimum is a proposal,
not proof that another room or another scale is impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .template_preprocessor import ProcessedTemplate


@dataclass
class SparseEvidence:
    edges: np.ndarray
    directions: np.ndarray
    edge_weights: np.ndarray
    semantic: np.ndarray
    labels: np.ndarray
    background: np.ndarray
    center: np.ndarray
    cells: int


@dataclass(frozen=True)
class SparsePose:
    position: tuple[float, float]
    scale: float
    score: float
    semantic: float
    edge_hit: float
    background_conflict: float


def balanced_points(mask: np.ndarray, limit: int, divisions: int = 6) -> tuple[np.ndarray, int]:
    """Stratify across occupied cells; a long wall must not monopolize sampling."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.empty((0, 2), np.float32), 0
    cell_x = np.minimum(divisions - 1, xx * divisions // mask.shape[1])
    cell_y = np.minimum(divisions - 1, yy * divisions // mask.shape[0])
    cells = cell_y * divisions + cell_x
    occupied = np.unique(cells)
    quota = max(1, limit // len(occupied))
    indices = []
    for cell in occupied:
        group = np.flatnonzero(cells == cell)
        indices.extend(group[np.linspace(0, len(group) - 1, min(quota, len(group)), dtype=int)])
    selected = np.asarray(indices, int)
    return np.column_stack((xx[selected], yy[selected])).astype(np.float32), len(occupied)


def _directions(map_mask: np.ndarray, edge_points: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(map_mask, (3, 3), 0.7)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    xy = edge_points.astype(int)
    gx, gy = np.abs(dx[xy[:, 1], xy[:, 0]]), np.abs(dy[xy[:, 1], xy[:, 0]])
    return np.where(gy > gx * 1.5, 1, np.where(gx > gy * 1.5, 2, 0)).astype(int)


def build_evidence(observation: SegmentationResult) -> SparseEvidence:
    edges, cells = balanced_points(observation.edge_mask, 1000)
    directions = _directions(observation.map_mask, edges)
    weights = np.ones(len(edges), np.float32)
    if len(edges) and observation.known_background_mask is not None:
        # A reliable dark rim supports a physical wall. Unknown-adjacent edges
        # remain available at lower weight; fog boundaries cannot be identified
        # perfectly from a binary mask.
        known_rim = cv2.dilate(observation.known_background_mask, np.ones((5, 5), np.uint8))
        xy = edges.astype(int)
        weights = np.where(known_rim[xy[:, 1], xy[:, 0]] > 0, 1.0, 0.65).astype(np.float32)
    corridor, _ = balanced_points(observation.corridor_mask, 350)
    room, _ = balanced_points(observation.room_mask, 350)
    semantic = np.concatenate((corridor, room))
    labels = np.concatenate((np.zeros(len(corridor), int), np.ones(len(room), int)))
    background, _ = balanced_points(
        observation.known_background_mask if observation.known_background_mask is not None
        else np.zeros_like(observation.map_mask), 250)
    return SparseEvidence(edges, directions, weights, semantic, labels, background,
                          np.float32([observation.map_mask.shape[1] / 2,
                                      observation.map_mask.shape[0] / 2]), cells)


def _fields(template: ProcessedTemplate) -> tuple[np.ndarray, np.ndarray]:
    cached = getattr(template, "_sparse_distance_fields", None)
    if cached is not None:
        return cached
    yy, xx = np.nonzero(template.edge_mask)
    points = np.column_stack((xx, yy)).astype(np.float32)
    directions = _directions(template.map_mask, points)
    distances = [np.minimum(template.distance_transform, 40).astype(np.float32)]
    for direction in (1, 2):
        mask = np.zeros_like(template.edge_mask)
        selected = directions == direction
        mask[yy[selected], xx[selected]] = 255
        distances.append(np.minimum(cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3), 40))
    distance_fields = np.ascontiguousarray(np.stack(distances, axis=2))
    semantics = np.ascontiguousarray(np.stack((template.corridor_mask, template.room_mask), axis=2),
                                     dtype=np.float32) / 255.0
    cached = (distance_fields, semantics)
    template._sparse_distance_fields = cached
    return cached


def _sample(field: np.ndarray, points: np.ndarray, poses: np.ndarray,
            center: np.ndarray, border: float = 0) -> np.ndarray:
    coords = ((points[None] - center) * poses[:, None, 2:3] + poses[:, None, :2])
    return cv2.remap(field, np.ascontiguousarray(coords[:, :, 0]),
                     np.ascontiguousarray(coords[:, :, 1]), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(border, border, border, border))


def score_poses(template: ProcessedTemplate, evidence: SparseEvidence,
                poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Poses are mapped observation-center x/y and screenshot-to-template scale."""
    distances, semantics = _fields(template)
    n = len(poses)
    if len(evidence.edges) < 20 or len(evidence.semantic) < 20:
        return tuple(np.zeros(n, np.float32) for _ in range(4))
    values = _sample(distances, evidence.edges, poses, evidence.center, 40)
    edge_distance = values[:, np.arange(len(evidence.edges)), evidence.directions]
    edge_distance = edge_distance / np.maximum(poses[:, 2:3], 0.1)
    edge = np.average(np.exp(-edge_distance / 4.0), axis=1, weights=evidence.edge_weights)
    hit = np.average(edge_distance <= 4.0, axis=1, weights=evidence.edge_weights)
    sem_values = _sample(semantics, evidence.semantic, poses, evidence.center)
    semantic = sem_values[:, np.arange(len(evidence.semantic)), evidence.labels].mean(axis=1)
    conflict = np.zeros(n, np.float32)
    if len(evidence.background):
        background = _sample(semantics, evidence.background, poses, evidence.center)
        conflict = background.max(axis=2).mean(axis=1)
    score = 0.50 * semantic + 0.45 * edge - 0.05 * conflict
    return score, semantic, hit, conflict


def refine_sparse_poses(template: ProcessedTemplate, observation: SegmentationResult,
                        seeds: list[tuple[tuple[float, float], float]], *,
                        scale_min: float, scale_max: float,
                        position_radius: float = 24.0, scale_radius: float = 0.05,
                        max_seeds: int = 8) -> tuple[list[SparsePose], dict]:
    started = perf_counter()
    evidence = build_evidence(observation)
    if not seeds or len(evidence.edges) < 40 or evidence.cells < 3:
        return [], {"reason": "insufficient_point_evidence"}
    seeds = seeds[:max_seeds]
    initial = np.float32([[p[0] + s * evidence.center[0], p[1] + s * evidence.center[1], s]
                          for p, s in seeds])
    poses = initial.copy()
    lower = initial - np.float32([position_radius, position_radius, scale_radius])
    upper = initial + np.float32([position_radius, position_radius, scale_radius])
    lower[:, 2] = np.maximum(lower[:, 2], scale_min)
    upper[:, 2] = np.minimum(upper[:, 2], scale_max)
    # Batched bounded coordinate search avoids a fragile numerical solve on
    # a single straight wall and has predictable work even on flat objectives.
    offsets = np.float32([(x, y, s) for x in (-1, 0, 1)
                         for y in (-1, 0, 1) for s in (-1, 0, 1)])
    iterations = 0
    for translation_step, zoom_step in ((8, .024), (4, .012), (2, .006), (1, .003)):
        for _ in range(2):
            trials = poses[:, None] + offsets[None] * np.float32([translation_step, translation_step, zoom_step])
            trials = np.clip(trials, lower[:, None], upper[:, None])
            flat = trials.reshape(-1, 3)
            scores, _, _, _ = score_poses(template, evidence, flat)
            best = scores.reshape(len(poses), -1).argmax(axis=1)
            updated = trials[np.arange(len(poses)), best]
            iterations += 1
            if np.max(np.abs(updated - poses)) < 1e-4:
                break
            poses = updated
    scores, semantic, hit, conflict = score_poses(template, evidence, poses)
    result = []
    for index in np.argsort(-scores):
        scale = float(poses[index, 2])
        position = poses[index, :2] - scale * evidence.center
        result.append(SparsePose(tuple(float(v) for v in position), scale, float(scores[index]),
                                 float(semantic[index]), float(hit[index]), float(conflict[index])))
    return result, {"elapsed_ms": (perf_counter() - started) * 1000,
                    "edge_points": len(evidence.edges), "semantic_points": len(evidence.semantic),
                    "spatial_cells": evidence.cells, "seed_count": len(seeds), "iterations": iterations}
