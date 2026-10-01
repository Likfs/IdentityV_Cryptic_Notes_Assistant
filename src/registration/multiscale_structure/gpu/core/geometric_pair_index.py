"""Bounded landmark-pair voting for candidate generation, never acceptance."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np


def landmark_points(features: np.ndarray, factor: float, limit: int = 180) -> tuple[np.ndarray, np.ndarray]:
    points, labels = [], []
    for channel in range(2, min(11, features.shape[2])):
        count, _, stats, centers = cv2.connectedComponentsWithStats(
            np.uint8(features[:, :, channel] > .1), 8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 2:
                continue
            points.append(centers[index] / factor)
            labels.append(channel)
    points = np.asarray(points, np.float32).reshape(-1, 2)
    labels = np.asarray(labels, int)
    if len(points) > limit:
        selected = np.linspace(0, len(points) - 1, limit, dtype=int)
        points, labels = points[selected], labels[selected]
    return points, labels


def _pairs(points: np.ndarray, labels: np.ndarray, neighbors: int = 8) -> tuple[np.ndarray, ...]:
    starts, ends = [], []
    for index, point in enumerate(points):
        lengths = np.linalg.norm(points - point, axis=1)
        nearby = np.flatnonzero((lengths >= 24) & (lengths <= 450))
        nearby = nearby[np.argsort(lengths[nearby])[:neighbors]]
        for other in nearby:
            starts.append(index)
            ends.append(other)
    starts, ends = np.asarray(starts, int), np.asarray(ends, int)
    vector = points[ends] - points[starts]
    lengths = np.linalg.norm(vector, axis=1)
    return starts, ends, vector, lengths, labels[starts] * 16 + labels[ends]


@dataclass
class GeometricPairIndex:
    points: np.ndarray
    labels: np.ndarray
    pairs: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, features: np.ndarray, factor: float) -> "GeometricPairIndex":
        points, labels = landmark_points(features, factor)
        return cls(points, labels, _pairs(points, labels))

    def query(self, features: np.ndarray, factor: float, *, scale_min: float,
              scale_max: float, max_candidates: int = 6) -> list[dict]:
        observed, labels = landmark_points(features, factor, limit=32)
        starts, ends, vectors, lengths, kinds = _pairs(observed, labels, neighbors=6)
        ti, _, tv, tl, tk = self.pairs
        if len(starts) < 3 or len(ti) == 0:
            return []
        votes = defaultdict(list)
        for source, target, vector, length, kind in zip(starts, ends, vectors, lengths, kinds):
            scale = tl / length
            cosine = (tv @ vector) / np.maximum(tl * length, 1e-6)
            valid = np.flatnonzero((tk == kind) & (scale >= scale_min) & (scale <= scale_max)
                                   & (cosine >= .995))
            for index in valid[:64]:
                s = float(scale[index])
                translation = self.points[ti[index]] - s * observed[source]
                bucket = (round(float(translation[0]) / 12), round(float(translation[1]) / 12),
                          round(float(np.log(s)) / .04))
                votes[bucket].append((translation, s, int(source), int(target)))
        candidates = []
        for values in votes.values():
            supported = set(value[2] for value in values) | set(value[3] for value in values)
            if len(supported) < 3 or len(values) < 3:
                continue
            position = np.median(np.stack([value[0] for value in values]), axis=0)
            scale = float(np.median([value[1] for value in values]))
            candidates.append({"position": tuple(float(v) for v in position), "scale": scale,
                               "landmark_support": len(supported), "pair_votes": len(values)})
        candidates.sort(key=lambda value: (value["landmark_support"], value["pair_votes"]), reverse=True)
        selected = []
        for candidate in candidates:
            if any(np.linalg.norm(np.subtract(candidate["position"], item["position"])) < 24
                   and abs(candidate["scale"] / item["scale"] - 1) < .05 for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_candidates:
                break
        return selected
