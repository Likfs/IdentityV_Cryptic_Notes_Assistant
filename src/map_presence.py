"""Detect the game's map interface using its fixed 'my location' marker."""

from __future__ import annotations

from src.paths import DIAGNOSTIC_ROOT
import json
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_MARKER_PATH = ROOT / 'assets/game_ui/my_location.png'
# Inclusive game-client coordinates: (2340, 1240) through (2510, 1400).
MAP_MARKER_ROI = (2340, 1240, 171, 161)
MAP_MARKER_REPORT_DIR = DIAGNOSTIC_ROOT / 'map_presence'
OPEN_SCORE = 0.75
CLOSED_SCORE = 0.50
MATCH_MARGIN = 8


@dataclass(frozen=True)
class MapPresence:
    state: str  # open, closed, or uncertain
    score: float
    offset: tuple[int, int]


@lru_cache(maxsize=1)
def load_map_marker() -> np.ndarray:
    if not MAP_MARKER_PATH.is_file():
        raise FileNotFoundError(f"地图界面标识素材不存在：{MAP_MARKER_PATH}")
    encoded = np.fromfile(str(MAP_MARKER_PATH), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (MAP_MARKER_ROI[3], MAP_MARKER_ROI[2]):
        raise ValueError("地图界面标识素材必须是 171×161 像素的图片")
    return image


def classify_map_presence(image: np.ndarray, template: np.ndarray | None = None) -> MapPresence:
    """Score the marker after scaling a captured client ROI to reference size."""
    if template is None:
        template = load_map_marker()
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("地图界面标识截图无效")
    target_size = (template.shape[1], template.shape[0])
    if (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    if min(template.shape[:2]) <= 2 * MATCH_MARGIN:
        raise ValueError("地图界面标识素材尺寸过小")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    needle = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    needle = needle[MATCH_MARGIN:-MATCH_MARGIN, MATCH_MARGIN:-MATCH_MARGIN]
    scores = cv2.matchTemplate(gray, needle, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    score = float(np.clip(score, -1.0, 1.0))
    state = "open" if score >= OPEN_SCORE else "closed" if score <= CLOSED_SCORE else "uncertain"
    return MapPresence(state, score, (location[0] - MATCH_MARGIN, location[1] - MATCH_MARGIN))


def save_map_presence_diagnostic(image: np.ndarray | None, *, detection: MapPresence | None,
                                 region: list[int] | None, attempt: int,
                                 error: str | None = None, elapsed_ms: float | None = None,
                                 floor_sample=None, observed_state: str | None = None,
                                 timing: dict | None = None) -> Path:
    """Keep the most recent marker probe for troubleshooting without filling the disk."""
    io_started = time.perf_counter()
    MAP_MARKER_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if image is not None:
        cv2.imencode(".png", image)[1].tofile(str(MAP_MARKER_REPORT_DIR / "latest.png"))
    if floor_sample is not None:
        cv2.imencode(".png", floor_sample.image)[1].tofile(str(MAP_MARKER_REPORT_DIR / "latest_floor.png"))
    details = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "reference_roi": list(MAP_MARKER_ROI),
        "screen_roi": region,
        "attempt": attempt + 1,
        "state": observed_state or (detection.state if detection else "error"),
        "confirmed_by": ("floor_indicator" if floor_sample is not None and floor_sample.detection.accepted
                         else "mylocation" if detection and detection.state == "open" else None),
        "floor_detection": floor_sample.detection.diagnostics() if floor_sample is not None else None,
        "floor_screen_roi": floor_sample.region if floor_sample is not None else None,
        "floor_image_captured": floor_sample is not None,
        "score": detection.score if detection else None,
        "offset": list(detection.offset) if detection else None,
        "error": error,
        "image_captured": image is not None,
        "elapsed_ms": elapsed_ms,
        "timing": timing,
        "diagnostic_images_ms": (time.perf_counter() - io_started) * 1000,
    }
    path = MAP_MARKER_REPORT_DIR / "latest.json"
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    if timing is not None and timing.get('complete'):
        record = {"captured_at": details['captured_at'], "state": details['state'],
                  "confirmed_by": details['confirmed_by'], **timing,
                  "diagnostic_write_ms": (time.perf_counter() - io_started) * 1000}
        # Preserve completed operations; latest.json alone cannot reveal a distribution.
        with (MAP_MARKER_REPORT_DIR / 'timings.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path
