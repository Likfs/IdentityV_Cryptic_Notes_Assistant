"""Confirmed, per-session map evidence in processed clean-template coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time, monotonic
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.registration.runtime_registration import PreparedRegistrationTemplate, RuntimeRegistrationResult


@dataclass(frozen=True)
class EvidenceUpdate:
    timestamp: float
    method: str
    structure_precision: float
    semantic_precision: float
    edge_median_px: float
    new_structure_pixels: int
    source: str | None = None


@dataclass
class PendingPoseHypotheses:
    """Unconfirmed alternatives for a map view that has not moved on screen."""

    transforms: list[np.ndarray]  # Screenshot to processed-template coordinates.
    screen_mask: np.ndarray
    pan_count: int = 0
    created_at: float = field(default_factory=monotonic)


def pending_motion(old: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    """Independent screen-to-screen check; it never provides an absolute pose."""
    corners = cv2.goodFeaturesToTrack(old, 80, .02, 12, blockSize=5)
    if corners is None or len(corners) < 8:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(old, current, corners, None,
        winSize=(25, 25), maxLevel=3)
    if tracked is None or status is None:
        return None
    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, old, tracked, None,
        winSize=(25, 25), maxLevel=3)
    if back is None or backward_status is None:
        return None
    keep = ((status.ravel() > 0) & (backward_status.ravel() > 0)
            & (np.linalg.norm(back-corners, axis=2).ravel() <= 1.5))
    before, after = corners[keep, 0], tracked[keep, 0]
    if len(before) < 8 or min(np.ptp(before, axis=0)) < 80:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
                                               ransacReprojThreshold=1.5)
    if matrix is None or inliers is None or np.mean(inliers) < .75:
        return None
    return matrix


def estimate_pending_translation(old: np.ndarray, current: np.ndarray) -> tuple[float, float] | None:
    """A bounded proposal, accepted only when several old regions survive.

    Growth is allowed, disappearance and scale changes are not. This never
    confirms an absolute map pose; new evidence still has to separate candidates.
    """
    yy, xx = np.nonzero(old)
    if len(xx) < 1200 or np.ptp(xx) < 80 or np.ptp(yy) < 80:
        return None
    factor = min(1., 512/max(old.shape))
    size = (max(1, round(old.shape[1]*factor)), max(1, round(old.shape[0]*factor)))
    a = cv2.resize(old.astype(np.float32)/255, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(current.astype(np.float32)/255, size, interpolation=cv2.INTER_AREA)
    shift, response = cv2.phaseCorrelate(a, b)
    dx, dy = shift[0]/factor, shift[1]/factor
    if not np.isfinite([dx, dy, response]).all() or response < .25 or max(abs(dx), abs(dy)) > 160:
        return None
    # Pixel-level refinement on a small translation grid; no affine/zoom fitting.
    selected = np.linspace(0, len(xx)-1, min(2000, len(xx)), dtype=int)
    points = np.float32(np.column_stack((xx[selected], yy[selected])))
    distances = cv2.distanceTransform(np.uint8(current == 0), cv2.DIST_L2, 3)
    offsets = np.float32([(round(dx)+i, round(dy)+j) for i in range(-3, 4) for j in range(-3, 4)])
    mapped = points[None]+offsets[:, None]
    residuals = cv2.remap(distances, np.ascontiguousarray(mapped[:, :, 0]),
                          np.ascontiguousarray(mapped[:, :, 1]), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1000)
    best = int(np.argmin(np.minimum(residuals, 20).mean(axis=1)))
    if np.mean(residuals[best] <= 2) < .93:
        return None
    cells = (np.minimum(2, (points[:, 0]-xx.min())*3/(np.ptp(xx)+1)).astype(int)
             + 3*np.minimum(2, (points[:, 1]-yy.min())*3/(np.ptp(yy)+1)).astype(int))
    checked = [residuals[best, cells == cell] for cell in np.unique(cells)
               if np.count_nonzero(cells == cell) >= 50]
    if len(checked) < 3 or any(np.mean(part <= 2) < .85 for part in checked):
        return None
    return tuple(float(value) for value in offsets[best])


@dataclass
class TemporalEvidence:
    """Only accepted observations enter these session-local masks."""

    template_shape: tuple[int, int]
    source_sha256: str
    corridor_votes: np.ndarray
    room_votes: np.ndarray
    background_votes: np.ndarray
    last_source_index: np.ndarray
    confidence: np.ndarray
    updates: list[EvidenceUpdate] = field(default_factory=list)
    recent_transforms: list[np.ndarray] = field(default_factory=list)
    pending_poses: PendingPoseHypotheses | None = None

    def clear_pending(self) -> None:
        self.pending_poses = None

    def remember_candidates(self, transforms: list[np.ndarray], screen_mask: np.ndarray) -> None:
        """Keep only distinct, bounded candidates; none is confirmed evidence."""
        distinct: list[np.ndarray] = []
        center = np.float32([screen_mask.shape[1] / 2, screen_mask.shape[0] / 2])
        for transform in transforms:
            if transform.shape != (2, 3) or not np.isfinite(transform).all():
                continue
            point = transform[:, :2] @ center + transform[:, 2]
            if any(np.linalg.norm(point - (old[:, :2] @ center + old[:, 2])) < 35
                   and abs(float(transform[0, 0] - old[0, 0])) < 0.04
                   for old in distinct):
                continue
            distinct.append(transform.copy())
            if len(distinct) == 8:
                break
        if distinct:
            previous = self.pending_poses
            created = (previous.created_at if previous is not None
                       and np.array_equal(previous.screen_mask, screen_mask) else monotonic())
            self.pending_poses = PendingPoseHypotheses(distinct, screen_mask.copy(), created_at=created)

    def pending_new_mask(self, screen_mask: np.ndarray, *, allow_translation: bool = False) -> np.ndarray | None:
        pending = self.pending_poses
        if (pending is None or pending.screen_mask.shape != screen_mask.shape
                or monotonic()-pending.created_at > 120):
            self.clear_pending()
            return None
        if np.array_equal(pending.screen_mask, screen_mask):
            return None
        motion = pending_motion(pending.screen_mask, screen_mask) if allow_translation else None
        if motion is not None:
            scale = float(np.hypot(motion[0, 0], motion[1, 0]))
            angle = abs(float(np.arctan2(motion[1, 0], motion[0, 0])))
            if abs(scale-1) > .02 or angle > np.deg2rad(1):
                self.clear_pending()
                return None
        old = pending.screen_mask > 0
        current = screen_mask > 0
        overlap = np.count_nonzero(old & current)
        if (overlap < 0.65 * max(1, np.count_nonzero(old))
                or (motion is not None and np.linalg.norm(motion[:, 2]) > 2)):
            translation = (estimate_pending_translation(pending.screen_mask, screen_mask)
                           if allow_translation and pending.pan_count < 3 else None)
            if translation is None:
                self.clear_pending()
                return None
            shift = np.float32(translation)
            translated = cv2.warpAffine(pending.screen_mask,
                                        np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                        (screen_mask.shape[1], screen_mask.shape[0]),
                                        flags=cv2.INTER_NEAREST)
            transforms = []
            for pose in pending.transforms:
                updated = pose.copy()
                updated[:, 2] -= updated[:, :2] @ shift
                transforms.append(updated)
            pending = PendingPoseHypotheses(transforms, translated, pending.pan_count+1, pending.created_at)
            self.pending_poses = pending
            old = translated > 0
        new = current & ~cv2.dilate(old.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if np.count_nonzero(new) < 1200:
            return None
        return new.astype(np.uint8) * 255

    @classmethod
    def empty(cls, prepared: PreparedRegistrationTemplate) -> TemporalEvidence:
        shape = prepared.processed.map_mask.shape
        return cls(shape, prepared.local_structure_index.source_sha256,
                   np.zeros(shape, np.uint8), np.zeros(shape, np.uint8),
                   np.zeros(shape, np.uint8), np.zeros(shape, np.uint32),
                   np.zeros(shape, np.uint8))

    def matches(self, prepared: PreparedRegistrationTemplate) -> bool:
        return (self.template_shape == prepared.processed.map_mask.shape
                and self.source_sha256 == prepared.local_structure_index.source_sha256)

    @property
    def structure_mask(self) -> np.ndarray:
        return cv2.bitwise_or(self.corridor_votes, self.room_votes)

    def predicted_center(self, screen_point: np.ndarray) -> np.ndarray | None:
        if len(self.recent_transforms) < 2:
            return None
        previous, latest = self.recent_transforms[-2:]
        earlier = previous[:, :2] @ screen_point + previous[:, 2]
        current = latest[:, :2] @ screen_point + latest[:, 2]
        movement = current - earlier
        distance = float(np.linalg.norm(movement))
        if distance > 160.0:
            movement *= 160.0 / distance
        return current + movement

    def _warp(self, mask: np.ndarray, transform: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        matrix = transform.copy()
        matrix[:, 2] += matrix[:, :2] @ np.float32(origin)
        return cv2.warpAffine(mask, matrix,
                              (self.template_shape[1], self.template_shape[0]),
                              flags=cv2.INTER_NEAREST)

    def conflict(
        self,
        *,
        corridor: np.ndarray,
        room: np.ndarray,
        background: np.ndarray | None,
        transform: np.ndarray,
        origin: tuple[int, int],
    ) -> tuple[int, float | None]:
        details = self.conflict_details(
            corridor=corridor, room=room, background=background,
            transform=transform, origin=origin,
        )
        return details["compared_pixels"], details["conflict_rate"]

    def conflict_details(
        self,
        *,
        corridor: np.ndarray,
        room: np.ndarray,
        background: np.ndarray | None,
        transform: np.ndarray,
        origin: tuple[int, int],
    ) -> dict:
        """Compare current observation with previously confirmed local facts.

        A 3-pixel tolerance avoids treating ordinary edge jitter as a conflict.
        Unknown fog is never considered background.
        """
        if not self.updates:
            return {"compared_pixels": 0, "conflict_rate": None,
                    "hard_conflict_pixels": 0,
                    "semantic_pixels": 0, "structure_on_background_pixels": 0,
                    "background_on_structure_pixels": 0}
        current_corridor = self._warp(corridor, transform, origin) > 0
        current_room = self._warp(room, transform, origin) > 0
        current_background = (self._warp(background, transform, origin) > 0
                              if background is not None else np.zeros(self.template_shape, bool))
        current_structure = current_corridor | current_room
        kernel = np.ones((7, 7), np.uint8)
        close_corridor = cv2.dilate(self.corridor_votes, kernel) > 0
        close_room = cv2.dilate(self.room_votes, kernel) > 0
        close_structure = close_corridor | close_room
        close_background = cv2.dilate(self.background_votes, kernel) > 0
        comparable = (current_structure & (close_structure | close_background)) | (
            current_background & close_structure
        )
        compared = int(np.count_nonzero(comparable))
        if compared < 500:
            return {"compared_pixels": compared, "conflict_rate": None,
                    "hard_conflict_pixels": 0,
                    "semantic_pixels": 0, "structure_on_background_pixels": 0,
                    "background_on_structure_pixels": 0}
        wrong_semantic = ((current_corridor & close_room & ~close_corridor)
                          | (current_room & close_corridor & ~close_room))
        wrong_structure = current_structure & close_background & ~close_structure
        wrong_background = current_background & close_structure & ~close_background
        conflict = int(np.count_nonzero((wrong_semantic | wrong_structure | wrong_background)
                                        & comparable))
        hard_conflict = int(np.count_nonzero((wrong_semantic | wrong_structure) & comparable))
        rate = conflict / compared
        if rate < 0.10:
            return {"compared_pixels": compared, "conflict_rate": rate,
                    "hard_conflict_pixels": hard_conflict,
                    "semantic_pixels": None, "structure_on_background_pixels": None,
                    "background_on_structure_pixels": None}
        return {
            "compared_pixels": compared,
            "conflict_rate": rate,
            "hard_conflict_pixels": hard_conflict,
            "semantic_pixels": int(np.count_nonzero(wrong_semantic & comparable)),
            "structure_on_background_pixels": int(np.count_nonzero(wrong_structure & comparable)),
            "background_on_structure_pixels": int(np.count_nonzero(wrong_background & comparable)),
        }

    def add_registration(
        self,
        registration: RuntimeRegistrationResult,
        prepared: PreparedRegistrationTemplate,
        *,
        source: str | None = None,
    ) -> bool:
        if (not self.matches(prepared) or not registration.accepted
                or not registration.validation.get('trusted_history_eligible', True)
                or registration.validation.get('projection_scope')=='local'):
            return False
        observation = registration.observation_segmentation
        origin_box = registration.observation_bbox
        transform = registration.screenshot_to_clean
        if observation is None or origin_box is None or transform is None:
            return False
        validation = registration.validation
        edge = validation.get("edge_median_error_px")
        method = str(registration.timings.get("method", "unknown"))
        # Route-only corroboration and weak positions must not contaminate the
        # memory used for later registration decisions.
        if (validation.get("route_corroborated")
                or validation.get("scale_only_relaxed")
                or validation.get("structure_precision", 0.0) < 0.90
                or validation.get("semantic_precision", 0.0) < 0.82
                or edge is None or edge > 10.0):
            return False
        if (method == "global_candidate_review"
                and validation.get("eligible_candidate_count", 0) > 1):
            return False
        if method == "global_search" and (
                validation["structure_precision"] < 0.97
                or validation["semantic_precision"] < 0.94
                or edge > 4.0):
            return False
        processed_transform = transform.copy().astype(np.float32)
        processed_transform *= np.float32(prepared.registration_scale)
        processed_transform[:, 2] -= np.float32(prepared.processed.crop_offset)
        origin = (int(origin_box[0]), int(origin_box[1]))
        compared, conflict = self.conflict(
            corridor=observation.corridor_mask, room=observation.room_mask,
            background=observation.known_background_mask,
            transform=processed_transform, origin=origin,
        )
        if conflict is not None and conflict > 0.20:
            return False
        if self.updates and compared < 500 and method.startswith("temporal_") and (
                validation["structure_precision"] < 0.97
                or validation["semantic_precision"] < 0.94
                or edge > 3.0):
            return False
        corridor = self._warp(observation.corridor_mask, processed_transform, origin) > 0
        room = self._warp(observation.room_mask, processed_transform, origin) > 0
        background = (self._warp(observation.known_background_mask,
                                 processed_transform, origin) > 0
                      if observation.known_background_mask is not None
                      else np.zeros(self.template_shape, bool))
        current = corridor | room
        new_pixels = int(np.count_nonzero(current & (self.structure_mask == 0)))
        np.add(self.corridor_votes, corridor.astype(np.uint8),
               out=self.corridor_votes, where=self.corridor_votes < 3)
        np.add(self.room_votes, room.astype(np.uint8),
               out=self.room_votes, where=self.room_votes < 3)
        np.add(self.background_votes, background.astype(np.uint8),
               out=self.background_votes, where=self.background_votes < 3)
        self.last_source_index[current] = len(self.updates) + 1
        quality = min(float(validation["structure_precision"]),
                      float(validation["semantic_precision"]),
                      max(0.0, 1.0 - float(edge) / 20.0))
        self.confidence[current] = np.maximum(self.confidence[current],
                                              np.uint8(round(255.0 * quality)))
        self.updates.append(EvidenceUpdate(
            timestamp=time(), method=method,
            structure_precision=float(validation["structure_precision"]),
            semantic_precision=float(validation["semantic_precision"]),
            edge_median_px=float(edge), new_structure_pixels=new_pixels,
            source=source or f"observation_{len(self.updates) + 1}",
        ))
        self.recent_transforms.append(processed_transform.copy())
        self.recent_transforms = self.recent_transforms[-2:]
        return True
