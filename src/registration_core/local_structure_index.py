"""Persistent semantic and geometric index for one clean map floor.

The dense corridor/room channels provide the primary search. Confirmed
observation pixels and sparse landmarks rescue otherwise missed candidates.
Every proposed pose still needs full-resolution runtime validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .color_segmenter import SegmentationResult
from .structural_features import FEATURE_NAMES, FEATURE_VERSION, extract_structural_features
from .template_preprocessor import ProcessedTemplate


INDEX_VERSION = 3
INDEX_FACTOR = 0.25
COARSE_FACTOR = 0.125


@dataclass(frozen=True)
class LocalStructureCandidate:
    score: float
    scale: float
    position: tuple[float, float]  # Top left of the observation in the processed template crop.
    source: str = "semantic"
    feature_score: float | None = None
    background_conflict: float | None = None


@dataclass(frozen=True)
class LocalStructureQueryResult:
    candidates: list[LocalStructureCandidate]
    coarse_score: float
    coarse_spatial_margin: float | None
    fine_scale_count: int
    coarse_ms: float
    fine_ms: float
    feature_ms: float = 0.0
    feature_candidate_count: int = 0
    coarse_margin_status: str = "not_evaluated"

    @property
    def total_ms(self) -> float:
        return self.coarse_ms + self.fine_ms + self.feature_ms


@dataclass
class LocalStructureIndex:
    channels: np.ndarray
    coarse_channels: np.ndarray
    feature_channels: np.ndarray
    coarse_direction_channels: np.ndarray
    room_connections: tuple[tuple[int, int, int, int], ...]
    factor: float
    source_sha256: str
    cache_path: Path

    def _evidence_rescue(
        self, observation: SegmentationResult, scales: np.ndarray,
        *, max_candidates: int,
    ) -> tuple[list[LocalStructureCandidate], float]:
        """Search using only confirmed positive and negative screenshot pixels.

        A hidden part of the clean map must not lower a candidate's score.
        Sparse geometric landmarks add a small ranking bonus; final acceptance
        is still handled by the full-resolution runtime validation.
        """
        started = perf_counter()
        observed = np.stack((observation.corridor_mask, observation.room_mask), axis=2)
        known = observation.known_background_mask
        if known is None:
            known = np.zeros(observed.shape[:2], np.uint8)
        fine_observed = cv2.resize(observed, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                                   interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        fine_known = cv2.resize(known, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                                interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        coarse_observed = cv2.resize(fine_observed, None, fx=0.5, fy=0.5,
                                     interpolation=cv2.INTER_NEAREST)
        coarse_known = cv2.resize(fine_known, None, fx=0.5, fy=0.5,
                                  interpolation=cv2.INTER_NEAREST)
        if float(np.sum(coarse_observed)) < 20:
            return [], (perf_counter() - started) * 1000.0

        def evidence_map(channels: np.ndarray, padding: int) -> np.ndarray:
            occupied = np.clip(np.sum(channels, axis=2), 0.0, 1.0)
            combined = np.ascontiguousarray(np.dstack((channels, occupied)))
            return cv2.copyMakeBorder(combined, padding, padding, padding, padding,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

        coarse_padding = 5
        fine_padding = 10
        coarse_template = evidence_map(self.coarse_channels, coarse_padding)
        fine_template = evidence_map(self.channels, fine_padding)

        def evidence_patch(semantic: np.ndarray, background: np.ndarray,
                           scale: float) -> tuple[np.ndarray, float] | None:
            positive = cv2.resize(semantic, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
            negative = cv2.resize(background, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
            count = float(np.sum(positive))
            if count < 15:
                return None
            negative_count = float(np.sum(negative))
            weight = 0.35 * count / negative_count if negative_count >= 6 else 0.0
            patch = np.ascontiguousarray(np.dstack((positive, -weight * negative)))
            return patch, count

        coarse: list[LocalStructureCandidate] = []
        for scale in scales:
            prepared = evidence_patch(coarse_observed, coarse_known, float(scale))
            if prepared is None:
                continue
            patch, count = prepared
            if patch.shape[0] > coarse_template.shape[0] or patch.shape[1] > coarse_template.shape[1]:
                continue
            response = cv2.matchTemplate(coarse_template, patch, cv2.TM_CCORR) / count
            for _ in range(3):
                _, score, _, point = cv2.minMaxLoc(response)
                if score < 0.62 or not np.isfinite(score):
                    break
                coarse.append(LocalStructureCandidate(
                    float(score), float(scale),
                    ((point[0] - coarse_padding) / COARSE_FACTOR,
                     (point[1] - coarse_padding) / COARSE_FACTOR),
                    source="evidence_rescue",
                ))
                cv2.circle(response, point, 5, 0.0, -1)
        coarse.sort(key=lambda item: item.score, reverse=True)
        selected: list[LocalStructureCandidate] = []
        for candidate in coarse:
            if any(np.linalg.norm(np.subtract(candidate.position, previous.position)) < 28.0
                   and abs(candidate.scale - previous.scale) < 0.10 for previous in selected):
                continue
            selected.append(candidate)
            if len(selected) == 6:
                break
        if not selected:
            return [], (perf_counter() - started) * 1000.0

        observed_features = self._observation_features(observation)
        feature_template = cv2.copyMakeBorder(
            cv2.dilate(self.feature_channels, np.ones((3, 3), np.uint8)), fine_padding,
                                               fine_padding, fine_padding, fine_padding,
                                               cv2.BORDER_CONSTANT, value=0)
        fine: list[LocalStructureCandidate] = []
        scale_step = float(scales[1] - scales[0]) if len(scales) > 1 else 0.0
        patch_cache: dict[float, tuple[np.ndarray, float] | None] = {}
        feature_cache: dict[tuple[int, int], np.ndarray] = {}
        for candidate_index, candidate in enumerate(selected):
            shifts = (-1, 0, 1) if candidate_index < 2 else (0,)
            nearby_scales = sorted({
                round(float(np.clip(candidate.scale + shift * scale_step,
                                    scales[0], scales[-1])), 4)
                for shift in shifts
            })
            for scale in nearby_scales:
                if scale not in patch_cache:
                    patch_cache[scale] = evidence_patch(fine_observed, fine_known, scale)
                prepared = patch_cache[scale]
                if prepared is None:
                    continue
                patch, count = prepared
                cx = int(round(candidate.position[0] * INDEX_FACTOR)) + fine_padding
                cy = int(round(candidate.position[1] * INDEX_FACTOR)) + fine_padding
                radius = 7
                left = max(0, cx - radius)
                top = max(0, cy - radius)
                right = min(fine_template.shape[1], cx + patch.shape[1] + radius + 1)
                bottom = min(fine_template.shape[0], cy + patch.shape[0] + radius + 1)
                window = fine_template[top:bottom, left:right]
                if window.shape[0] < patch.shape[0] or window.shape[1] < patch.shape[1]:
                    continue
                response = cv2.matchTemplate(window, patch, cv2.TM_CCORR) / count
                _, score, _, point = cv2.minMaxLoc(response)
                px = left + point[0] - fine_padding
                py = top + point[1] - fine_padding
                fx, fy = px + fine_padding, py + fine_padding
                patch_shape = (patch.shape[1], patch.shape[0])
                if patch_shape not in feature_cache:
                    feature_cache[patch_shape] = cv2.resize(
                        observed_features, patch_shape, interpolation=cv2.INTER_NEAREST,
                    )
                feature_patch = feature_cache[patch_shape]
                expected = feature_template[fy:fy + patch.shape[0], fx:fx + patch.shape[1]]
                available = float(np.sum(feature_patch))
                feature_score = None
                if available >= 20:
                    feature_score = float(np.sum(expected * feature_patch) / available)
                    score += 0.04 * feature_score
                fine.append(LocalStructureCandidate(
                    score=float(score), scale=scale,
                    position=(px / INDEX_FACTOR, py / INDEX_FACTOR),
                    source="evidence_rescue", feature_score=feature_score,
                ))
        fine.sort(key=lambda item: item.score, reverse=True)
        unique: list[LocalStructureCandidate] = []
        for candidate in fine:
            if any(np.linalg.norm(np.subtract(candidate.position, previous.position)) < 20.0
                   and abs(candidate.scale - previous.scale) < 0.04 for previous in unique):
                continue
            unique.append(candidate)
            if len(unique) >= max_candidates:
                break
        return unique, (perf_counter() - started) * 1000.0

    @staticmethod
    def _observation_features(observation: SegmentationResult) -> np.ndarray:
        room = cv2.resize(observation.room_mask, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                          interpolation=cv2.INTER_NEAREST)
        corridor = cv2.resize(observation.corridor_mask, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                              interpolation=cv2.INTER_NEAREST)
        return extract_structural_features(room, corridor).channels

    def _query_channels(
        self, observation: SegmentationResult, *, channels: np.ndarray,
        factor: float, scales: np.ndarray, candidates_per_scale: int,
        max_candidates: int,
    ) -> tuple[list[LocalStructureCandidate], float]:
        started = perf_counter()
        corridor = cv2.resize(observation.corridor_mask, None, fx=factor, fy=factor,
                              interpolation=cv2.INTER_NEAREST)
        room = cv2.resize(observation.room_mask, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_NEAREST)
        query_channels = np.stack((corridor, room), axis=2)
        found: list[LocalStructureCandidate] = []
        for scale in scales:
            patch = cv2.resize(query_channels, None, fx=float(scale), fy=float(scale),
                               interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            if patch.shape[0] > channels.shape[0] or patch.shape[1] > channels.shape[1]:
                continue
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue
            response = cv2.matchTemplate(channels, patch, cv2.TM_CCORR_NORMED)
            for _ in range(candidates_per_scale):
                _, score, _, point = cv2.minMaxLoc(response)
                if score <= 0.0 or not np.isfinite(score):
                    break
                found.append(LocalStructureCandidate(
                    score=float(score), scale=float(scale),
                    position=(point[0] / factor, point[1] / factor),
                ))
                # Distinct nearby peaks at the same scale carry no extra pose evidence.
                cv2.circle(response, point, 7, 0.0, -1)
        found.sort(key=lambda item: item.score, reverse=True)
        unique: list[LocalStructureCandidate] = []
        for candidate in found:
            if any(
                np.linalg.norm(np.subtract(candidate.position, previous.position)) < 24.0
                and abs(candidate.scale - previous.scale) < 0.045
                for previous in unique
            ):
                continue
            unique.append(candidate)
            if len(unique) >= max_candidates:
                break
        return unique, (perf_counter() - started) * 1000.0

    def query(
        self,
        observation: SegmentationResult,
        *,
        scale_min: float,
        scale_max: float,
        scale_step: float,
        coarse_min_score: float = 0.80,
        fine_min_score: float = 0.78,
        candidates_per_scale: int = 4,
        max_candidates: int = 16,
    ) -> LocalStructureQueryResult:
        scales = np.arange(scale_min, scale_max + scale_step * 0.25, scale_step, dtype=np.float32)
        coarse, coarse_ms = self._query_channels(
            observation, channels=self.coarse_channels, factor=COARSE_FACTOR,
            scales=scales, candidates_per_scale=2, max_candidates=8,
        )
        coarse_score = coarse[0].score if coarse else 0.0
        if coarse_score < coarse_min_score:
            rescued, feature_ms = self._evidence_rescue(
                observation, scales, max_candidates=max_candidates,
            )
            return LocalStructureQueryResult(rescued, coarse_score, None, 0, coarse_ms, 0.0,
                                             feature_ms, len(rescued), "not_evaluated_low_score")
        first = coarse[0]
        # Preserve this coarse proposal-routing heuristic. Final pose review
        # groups visible anchors after uniform geometric refinement instead.
        spatial_competitor = next((candidate for candidate in coarse[1:]
                                   if np.linalg.norm(np.subtract(candidate.position, first.position)) > 35.0),
                                  None)
        spatial_margin = (first.score - spatial_competitor.score
                          if spatial_competitor is not None else None)
        margin_status = "measured" if spatial_competitor is not None else "no_competitor_found"
        # Search around the leading scale modes.  Running every fine scale
        # when coarse positions compete makes a large map needlessly slow;
        # the evidence rescue below still covers a missed coarse mode.
        if spatial_margin is None or spatial_margin >= 0.05:
            near = np.abs(scales - first.scale) <= (scale_step + 1e-5)
            fine_scales = scales[near]
        else:
            modes: list[float] = []
            for candidate in coarse:
                if any(abs(candidate.scale - previous) <= scale_step * 1.5
                       for previous in modes):
                    continue
                modes.append(candidate.scale)
                if len(modes) >= 3:
                    break
            fine_scales = scales[np.any(
                np.stack([np.abs(scales - mode) <= scale_step + 1e-5
                          for mode in modes]), axis=0,
            )]
        fine, fine_ms = self._query_channels(
            observation, channels=self.channels, factor=self.factor, scales=fine_scales,
            candidates_per_scale=candidates_per_scale, max_candidates=max_candidates,
        )
        if not fine or fine[0].score < fine_min_score:
            rescued, feature_ms = self._evidence_rescue(
                observation, scales, max_candidates=max_candidates,
            )
            if rescued and (not fine or rescued[0].score > fine[0].score):
                return LocalStructureQueryResult(
                    rescued, coarse_score, spatial_margin, len(fine_scales), coarse_ms,
                    fine_ms, feature_ms, len(rescued), margin_status,
                )
            return LocalStructureQueryResult(
                fine, coarse_score, spatial_margin, len(fine_scales), coarse_ms,
                fine_ms, feature_ms, len(rescued), margin_status,
            )
        return LocalStructureQueryResult(
            fine, coarse_score, spatial_margin, len(fine_scales), coarse_ms, fine_ms,
            coarse_margin_status=margin_status,
        )


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_or_load_local_structure_index(
    template: ProcessedTemplate,
    source_path: Path,
    cache_root: Path,
) -> LocalStructureIndex:
    """Load an index only when the source image and processed geometry match."""
    source_sha256 = _source_digest(source_path)
    processed_digest = hashlib.sha256()
    processed_digest.update(template.corridor_mask.tobytes())
    processed_digest.update(template.room_mask.tobytes())
    processed_sha256 = processed_digest.hexdigest()
    target = cache_root / template.map_id / f"local_structure_index_v{INDEX_VERSION}.npz"
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                channels = data["channels"]
                coarse_channels = data["coarse_channels"]
                feature_channels = data["feature_channels"]
                coarse_direction_channels = data["coarse_direction_channels"]
                room_connections = data["room_connections"]
            if (
                metadata.get("version") == INDEX_VERSION
                and metadata.get("feature_version") == FEATURE_VERSION
                and metadata.get("source_sha256") == source_sha256
                and metadata.get("processed_sha256") == processed_sha256
                and metadata.get("processed_shape") == list(template.map_mask.shape)
                and metadata.get("crop_offset") == list(template.crop_offset)
                and metadata.get("factor") == INDEX_FACTOR
                and channels.dtype == np.float32
                and channels.ndim == 3
                and channels.shape[2] == 2
                and coarse_channels.dtype == np.float32
                and coarse_channels.ndim == 3
                and coarse_channels.shape[2] == 2
                and feature_channels.dtype == np.float32
                and feature_channels.ndim == 3
                and feature_channels.shape[:2] == channels.shape[:2]
                and feature_channels.shape[2] == len(FEATURE_NAMES)
                and coarse_direction_channels.dtype == np.float32
                and coarse_direction_channels.shape == (*coarse_channels.shape[:2], 2)
                and room_connections.ndim == 2
                and room_connections.shape[1] == 4
            ):
                return LocalStructureIndex(
                    np.ascontiguousarray(channels), np.ascontiguousarray(coarse_channels),
                    np.ascontiguousarray(feature_channels),
                    np.ascontiguousarray(coarse_direction_channels),
                    tuple(tuple(int(value) for value in row) for row in room_connections),
                    INDEX_FACTOR, source_sha256, target,
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    corridor = cv2.resize(template.corridor_mask, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                          interpolation=cv2.INTER_NEAREST)
    room = cv2.resize(template.room_mask, None, fx=INDEX_FACTOR, fy=INDEX_FACTOR,
                      interpolation=cv2.INTER_NEAREST)
    channels = np.ascontiguousarray(np.stack((corridor, room), axis=2).astype(np.float32) / 255.0)
    coarse_channels = np.ascontiguousarray(cv2.resize(
        channels, None, fx=COARSE_FACTOR / INDEX_FACTOR,
        fy=COARSE_FACTOR / INDEX_FACTOR, interpolation=cv2.INTER_NEAREST,
    ))
    features = extract_structural_features(room, corridor)
    feature_channels = np.ascontiguousarray(features.channels)
    coarse_direction_channels = np.ascontiguousarray(cv2.resize(
        feature_channels[:, :, :2], None, fx=COARSE_FACTOR / INDEX_FACTOR,
        fy=COARSE_FACTOR / INDEX_FACTOR, interpolation=cv2.INTER_NEAREST,
    ))
    room_connections = np.asarray(features.room_connections, dtype=np.int16).reshape(-1, 4)
    metadata = {
        "version": INDEX_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_sha256": source_sha256,
        "processed_sha256": processed_sha256,
        "processed_shape": list(template.map_mask.shape),
        "crop_offset": list(template.crop_offset),
        "factor": INDEX_FACTOR,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, channels=channels, coarse_channels=coarse_channels,
                        feature_channels=feature_channels,
                        coarse_direction_channels=coarse_direction_channels,
                        room_connections=room_connections, metadata=json.dumps(metadata))
    temp.replace(target)
    return LocalStructureIndex(channels, coarse_channels, feature_channels,
                               coarse_direction_channels, features.room_connections,
                               INDEX_FACTOR, source_sha256, target)
