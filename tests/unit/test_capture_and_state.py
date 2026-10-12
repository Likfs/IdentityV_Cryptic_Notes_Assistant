from __future__ import annotations
import pytest
from src import route_projection_core as app_module
from src.capture.region import CaptureRegion


def test_game_map_capture_tracks_window_position_and_size() -> None:
    roi = [901, 185, 1180, 1050]
    reference = [2560, 1440]
    moved = app_module.game_capture_region(CaptureRegion(100, 50, 2560, 1440), roi, reference)
    assert moved.as_list() == [1001, 235, 1180, 1050]
    resized = app_module.game_capture_region(CaptureRegion(100, 50, 1920, 1080), roi, reference)
    assert resized.as_list() == [776, 189, 885, 787]
    with pytest.raises(ValueError, match="完全位于"):
        app_module.game_capture_region(CaptureRegion(0, 0, 1920, 1080), [2000, 0, 600, 300], reference)


def test_focus_suppression_does_not_create_late_toggle() -> None:
    state = app_module.MapToggleState()
    assert state.observe(True, enabled=False) is None
    assert state.observe(True, enabled=True) is None
    assert state.observe(False, enabled=True) is None
    assert state.observe(True, enabled=True) is None
    assert state.observe(False, enabled=True) == "opened"


def test_escape_only_closes_an_open_map() -> None:
    state = app_module.MapToggleState()
    assert not state.observe_escape(True)
    assert not state.is_open
    state.observe_escape(False)
    assert state.observe(True) is None
    assert state.observe(False) == "opened"
    assert state.observe_escape(True)
    assert not state.is_open
    assert not state.observe_escape(True)
    state.observe(False)
    assert state.observe(True) is None
    assert state.observe(False) == "opened"


@pytest.mark.parametrize('title, titles, expected', [
    ('Codex', ['第五人格', 'Identity V'], False),
    ('第五人格', ['第五人格', 'Identity V'], True),
    ('identity v - game', ['Identity V'], True),
    ('第五人格', [], False),
    ('', ['第五人格', 'Identity V'], False),
])
def test_game_focus_requires_a_matching_foreground_title(monkeypatch, title, titles, expected) -> None:
    monkeypatch.setattr(app_module, 'foreground_window_title', lambda: title)
    assert app_module.game_window_is_foreground(titles) is expected
