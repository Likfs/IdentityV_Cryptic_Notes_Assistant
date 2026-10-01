"""Small-scale geometric landmarks shared by clean maps and foggy observations.

Landmarks are deliberately derived from the semantic masks, rather than from
colours in the original image.  A feature absent from a partial observation is
unknown, not evidence that the corresponding clean-map feature is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


FEATURE_VERSION = 1
FEATURE_NAMES = (
    "horizontal_edge", "vertical_edge", "corner_nw", "corner_ne",
    "corner_sw", "corner_se", "door_horizontal", "door_vertical",
    "corridor_turn", "corridor_t", "corridor_cross", "narrow_corridor",
)


@dataclass(frozen=True)
class StructuralFeatures:
    channels: np.ndarray  # H x W x len(FEATURE_NAMES), float32 in [0, 1]
    room_connections: tuple[tuple[int, int, int, int], ...]  # N, E, S, W per room.

    @property
    def counts(self) -> dict[str, int]:
        return {name: int(np.count_nonzero(self.channels[:, :, index] > 0.1))
                for index, name in enumerate(FEATURE_NAMES)}


def _thin(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton on the quarter-size corridor mask."""
    remaining = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(remaining)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(32):
        if cv2.countNonZero(remaining) == 0:
            break
        eroded = cv2.erode(remaining, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = eroded
    return skeleton


def _put_disk(channel: np.ndarray, x: int, y: int, radius: int = 2) -> None:
    height, width = channel.shape
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                channel[yy, xx] = 255


def _door_landmarks(room: np.ndarray, corridor: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Return horizontal/vertical contacts and their positions/directions."""
    height, width = room.shape
    interface = cv2.bitwise_and(cv2.dilate(room, np.ones((3, 3), np.uint8)),
                                cv2.dilate(corridor, np.ones((3, 3), np.uint8)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(interface, 8)
    doors = np.zeros((height, width, 2), np.uint8)
    points: list[tuple[int, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > 80:
            continue
        x, y = np.rint(centroids[label]).astype(int)
        if x < 2 or y < 2 or x >= width - 2 or y >= height - 2:
            continue
        # A left/right contact joins along the horizontal axis, and vice versa.
        horizontal = (int(room[y, x - 2] > 0) != int(room[y, x + 2] > 0))
        vertical = (int(room[y - 2, x] > 0) != int(room[y + 2, x] > 0))
        direction = 0 if horizontal and not vertical else 1 if vertical and not horizontal else -1
        if direction < 0:
            continue
        _put_disk(doors[:, :, direction], x, y)
        points.append((x, y, direction))
    return doors, points


def _room_connection_signatures(room: np.ndarray, doors: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int, int], ...]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(room, 8)
    signatures: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 16:
            continue
        cx, cy = centroids[label]
        directions = [0, 0, 0, 0]
        region = (labels == label).astype(np.uint8)
        nearby = cv2.dilate(region, np.ones((5, 5), np.uint8))
        for x, y, axis in doors:
            if not nearby[y, x]:
                continue
            direction = (1 if x >= cx else 3) if axis == 0 else (2 if y >= cy else 0)
            directions[direction] += 1
        signatures.append(tuple(directions))
    return tuple(signatures)


def extract_structural_features(room_mask: np.ndarray, corridor_mask: np.ndarray) -> StructuralFeatures:
    """Extract directional walls, oriented corners, contacts and junctions.

    Input masks should already be at index resolution (normally 0.25x).  This
    same function is used for the template and each observation.
    """
    room = ((room_mask > 0).astype(np.uint8) * 255)
    corridor = ((corridor_mask > 0).astype(np.uint8) * 255)
    combined = cv2.bitwise_or(room, corridor)
    height, width = combined.shape
    channels = np.zeros((height, width, len(FEATURE_NAMES)), np.uint8)
    if cv2.countNonZero(combined) == 0:
        return StructuralFeatures(channels.astype(np.float32), ())

    # Direction is defined by the wall tangent, hence a y gradient is a
    # horizontal wall.  A one-pixel dilation tolerates mask quantisation.
    smooth = cv2.GaussianBlur(combined, (3, 3), 0.6)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.Canny(combined, 40, 120) > 0
    channels[:, :, 0] = np.where(edge & (np.abs(dy) > np.abs(dx) * 1.35), 255, 0)
    channels[:, :, 1] = np.where(edge & (np.abs(dx) > np.abs(dy) * 1.35), 255, 0)

    # Harris corner locations are classified by the occupied quadrant.  This
    # preserves orientation while ignoring an unseen corner in a foggy crop.
    corners = cv2.goodFeaturesToTrack(combined, maxCorners=600, qualityLevel=0.08,
                                      minDistance=4, blockSize=3, useHarrisDetector=True)
    if corners is not None:
        for point in corners.reshape(-1, 2):
            x, y = np.rint(point).astype(int)
            if x < 3 or y < 3 or x >= width - 3 or y >= height - 3:
                continue
            values = (combined[y - 3, x - 3], combined[y - 3, x + 3],
                      combined[y + 3, x - 3], combined[y + 3, x + 3])
            occupied = [index for index, value in enumerate(values) if value > 0]
            if len(occupied) == 1:
                _put_disk(channels[:, :, 2 + occupied[0]], x, y)
            elif len(occupied) == 3:
                missing = next(index for index in range(4) if index not in occupied)
                _put_disk(channels[:, :, 2 + missing], x, y)

    doors, door_points = _door_landmarks(room, corridor)
    channels[:, :, 6:8] = doors

    skeleton = _thin(corridor)
    # A corridor centre is a turn/T/cross when its four directional arms
    # extend beyond a 2-pixel radius.  Merge nearby detections into blobs.
    coordinates = np.argwhere(skeleton > 0)
    for y, x in coordinates:
        if x < 3 or y < 3 or x >= width - 3 or y >= height - 3:
            continue
        arms = (
            np.any(skeleton[y - 3:y, x - 1:x + 2]),
            np.any(skeleton[y - 1:y + 2, x + 1:x + 4]),
            np.any(skeleton[y + 1:y + 4, x - 1:x + 2]),
            np.any(skeleton[y - 1:y + 2, x - 3:x]),
        )
        count = sum(bool(value) for value in arms)
        if count == 4:
            channels[y, x, 10] = 255
        elif count == 3:
            channels[y, x, 9] = 255
        elif count == 2 and not ((arms[0] and arms[2]) or (arms[1] and arms[3])):
            channels[y, x, 8] = 255

    distance = cv2.distanceTransform(corridor, cv2.DIST_L2, 3)
    channels[:, :, 11] = np.where((skeleton > 0) & (distance > 0) & (distance <= 3.5), 255, 0)
    return StructuralFeatures(channels.astype(np.float32) / 255.0,
                              _room_connection_signatures(room, door_points))
