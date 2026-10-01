from __future__ import annotations

import cv2
import numpy as np

from src.floor_indicator import FLOOR_TEMPLATE_DIR, FLOOR_TEMPLATE_FILES, classify_floor_indicator


def test_all_three_floor_templates_are_recognized_with_small_crop_shift() -> None:
    for floor, filename in FLOOR_TEMPLATE_FILES.items():
        template = cv2.imdecode(np.fromfile(str(FLOOR_TEMPLATE_DIR / filename), np.uint8), cv2.IMREAD_COLOR)
        search = np.full((82, 312, 3), 30, np.uint8)
        search[8:74, 9:305] = template
        result = classify_floor_indicator(search, (296, 66))
        assert result.accepted and result.floor == floor
        assert result.scores[floor] > 0.95


def test_floor_indicator_finds_both_live_horizontal_positions() -> None:
    template = cv2.imdecode(
        np.fromfile(str(FLOOR_TEMPLATE_DIR / "1st.png"), np.uint8), cv2.IMREAD_COLOR
    )
    for left in (6, 100):
        search = np.full((78, 402, 3), 30, np.uint8)
        search[6:72, left:left + 296] = template
        result = classify_floor_indicator(search, (296, 66))
        assert result.accepted and result.floor == "1st"
        assert result.match_position == (left, 6)


def test_floor_indicator_rejects_unrelated_dark_crop() -> None:
    result = classify_floor_indicator(np.full((82, 312, 3), 32, np.uint8), (296, 66))
    assert not result.accepted
    assert result.reason


def test_floor_indicator_supports_scaled_game_window() -> None:
    template = cv2.imdecode(np.fromfile(str(FLOOR_TEMPLATE_DIR / "2nd.png"), np.uint8), cv2.IMREAD_COLOR)
    scaled = cv2.resize(template, (222, 50), interpolation=cv2.INTER_AREA)
    search = np.full((59, 231, 3), 30, np.uint8)
    search[4:54, 5:227] = scaled
    result = classify_floor_indicator(search, (222, 50))
    assert result.accepted and result.floor == "2nd"
