"""Recorded UI crops and same-frame capture builders; no desktop input."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def recorded_control(kind='mixed'):
    return cv2.imread(str(ROOT / 'tests/fixtures/floor_ui' / f'{kind}.png'))


def template_control(index):
    """Synthetic placements exercise all seven template states, not real-game accuracy."""
    atlas = cv2.imread(str(ROOT / 'assets/app_ui/floor_controls.png'))
    height = 460 if index < 4 else 345
    image = np.full((510, 140, 3), 30, np.uint8)
    top = 25 if index < 4 else 135
    image[top:top+height, 30:118] = atlas[460-height:, index*88:(index+1)*88]
    return image


def projection_frame(region, map_image, map_origin, *, control=None, client=None):
    from src.route_projection_core import game_capture_region
    from src.capture.region import CaptureRegion
    from src.game_geometry import floor_sample_region
    left, top, width, height = region
    image = np.zeros((height, width, 3), np.uint8)
    x, y = map_origin[0]-left, map_origin[1]-top
    image[y:y+map_image.shape[0], x:x+map_image.shape[1]] = map_image
    client = client or CaptureRegion(0, 0, 2560, 1440)
    crop = floor_sample_region(client)
    x, y = crop.left-left, crop.top-top
    ui = recorded_control() if control is None else control
    image[y:y+crop.height, x:x+crop.width] = cv2.resize(ui, (crop.width, crop.height), interpolation=cv2.INTER_AREA)
    return image


def replay_control(case):
    """Place recorded pixels at their original screen coordinates before recropping.

    Original diagnostics omit a few outer background pixels of the new crop;
    padding is synthetic, while all visible button/label pixels stay unchanged.
    """
    from src.game_geometry import floor_sample_region
    from src.capture.region import CaptureRegion
    client = CaptureRegion.from_value(case['capture_context']['game_client_screen'])
    original = CaptureRegion.from_value(case['capture_context']['floor_ui_roi_screen'])
    target = floor_sample_region(client)
    image = cv2.imread(str(ROOT/'tests/fixtures/floor_ui'/case['image']))
    canvas = np.full((target.height, target.width, 3), 30, np.uint8)
    left, top = max(target.left, original.left), max(target.top, original.top)
    right = min(target.left+target.width, original.left+original.width)
    bottom = min(target.top+target.height, original.top+original.height)
    canvas[top-target.top:bottom-target.top, left-target.left:right-target.left] = image[
        top-original.top:bottom-original.top, left-original.left:right-original.left]
    return canvas
