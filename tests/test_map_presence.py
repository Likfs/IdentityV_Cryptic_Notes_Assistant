from __future__ import annotations

import cv2
import numpy as np

from src.map_presence import MAP_MARKER_ROI, classify_map_presence, load_map_marker
from src.route_projection_core import game_capture_region
from src.capture.region import CaptureRegion


def test_marker_asset_and_scaled_capture() -> None:
    marker = load_map_marker()
    assert marker.shape == (161, 171, 3)
    assert MAP_MARKER_ROI == (2340, 1240, 171, 161)
    region = game_capture_region(CaptureRegion(100, 50, 1920, 1080),
                                 list(MAP_MARKER_ROI), [2560, 1440])
    assert region.as_list() == [1855, 980, 128, 121]


def test_marker_match_accepts_scaled_image_and_rejects_unrelated_image() -> None:
    marker = load_map_marker()
    assert classify_map_presence(marker).state == "open"
    smaller = cv2.resize(marker, (128, 121), interpolation=cv2.INTER_AREA)
    assert classify_map_presence(smaller).state == "open"
    shifted = cv2.warpAffine(marker, np.float32([[1, 0, 5], [0, 1, -4]]), (171, 161))
    shifted = np.clip(shifted.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    assert classify_map_presence(shifted).state == "open"
    noise = np.random.default_rng(42).integers(0, 256, marker.shape, dtype=np.uint8)
    assert classify_map_presence(noise).state == "closed"
