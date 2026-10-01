from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOOR_TEMPLATE_DIR = PROJECT_ROOT / 'assets/game_ui/floor_indicators'
FLOOR_INDICATOR_ROI = (1870, 70, 296, 66)  # inclusive corners: (1870, 70)–(2165, 135)
FLOOR_INDICATOR_ALTERNATE_ROI = (1964, 70, 296, 66)  # inclusive end: (2259, 135)
# Capture the union once. Template matching then locates the 296×66 control at
# either horizontal position (or anywhere between them).
FLOOR_INDICATOR_SEARCH_ROI = (1870, 70, 390, 66)
FLOOR_INDICATOR_MARGIN = 6
FLOOR_TEMPLATE_FILES = {
    "b1": "b1.png",
    "1st": "1st.png",
    "2nd": "2nd.png",
}


@dataclass(frozen=True)
class FloorDetection:
    floor: str | None
    scores: dict[str, float]
    score_margin: float
    highlight_floor: str
    highlight_margin: float
    reason: str | None
    match_position: tuple[int, int] | None = None

    @property
    def accepted(self) -> bool:
        return self.floor is not None

    def diagnostics(self) -> dict:
        return {
            "accepted": self.accepted,
            "floor": self.floor,
            "scores": self.scores,
            "score_margin": self.score_margin,
            "highlight_floor": self.highlight_floor,
            "highlight_margin": self.highlight_margin,
            "reason": self.reason,
            "match_position": list(self.match_position) if self.match_position is not None else None,
        }


@dataclass(frozen=True)
class CapturedFloorSample:
    """One pre-capture observation; never a persistent floor/pose history."""
    detection: FloorDetection
    image: np.ndarray
    region: list[int]
    client: list[int]
    reference_size: list[int]
    captured_at: float
    generation: int
    session_id: str | None

    def reusable(self, *, client, reference_size, generation, session_id, now):
        return (self.detection.accepted and self.client == client
                and self.reference_size == reference_size and self.generation == generation
                and self.session_id == session_id and 0 <= now - self.captured_at <= .100)


@lru_cache(maxsize=1)
def _templates() -> dict[str, np.ndarray]:
    result = {}
    for floor, filename in FLOOR_TEMPLATE_FILES.items():
        path = FLOOR_TEMPLATE_DIR / filename
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (66, 296):
            raise RuntimeError(f"楼层标识模板缺失或尺寸不是 296×66：{path}")
        result[floor] = image
    return result


def classify_floor_indicator(search_bgr: np.ndarray, core_size: tuple[int, int]) -> FloorDetection:
    """Match a padded game-client crop against the three selected-floor states."""
    width, height = core_size
    if width < 30 or height < 12 or search_bgr.shape[1] < width or search_bgr.shape[0] < height:
        raise ValueError("楼层标识截图尺寸无效")
    gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    scores: dict[str, float] = {}
    positions: dict[str, tuple[int, int]] = {}
    for floor, template in _templates().items():
        scaled = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA if width < 296 else cv2.INTER_LINEAR)
        response = cv2.matchTemplate(gray, cv2.GaussianBlur(scaled, (3, 3), 0), cv2.TM_CCOEFF_NORMED)
        _min_score, max_score, _min_pos, max_pos = cv2.minMaxLoc(response)
        scores[floor] = float(max_score)
        positions[floor] = max_pos

    ranking = sorted(scores, key=scores.get, reverse=True)
    best, runner_up = ranking[:2]
    score_margin = scores[best] - scores[runner_up]
    x, y = positions[best]
    aligned = gray[y:y + height, x:x + width]
    normalized = cv2.resize(aligned, (296, 66), interpolation=cv2.INTER_AREA)
    panel_means = {
        floor: float(normalized[8:58, left:right].mean())
        for floor, left, right in (("b1", 0, 98), ("1st", 98, 197), ("2nd", 197, 296))
    }
    highlights = sorted(panel_means, key=panel_means.get, reverse=True)
    highlight_floor = highlights[0]
    highlight_margin = panel_means[highlights[0]] - panel_means[highlights[1]]

    reason = None
    if scores[best] < 0.58:
        reason = f"最高模板相似度不足：{scores[best]:.2f}"
    elif score_margin < 0.10:
        reason = f"前两名模板差距不足：{score_margin:.2f}"
    elif highlight_floor != best or highlight_margin < 15.0:
        reason = f"高亮区域无法确认：{highlight_floor}，差距 {highlight_margin:.1f}"
    return FloorDetection(best if reason is None else None, scores, score_margin,
                          highlight_floor, highlight_margin, reason, positions[best])
