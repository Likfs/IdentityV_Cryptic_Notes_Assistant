from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from src.app_catalog import MapCatalog, load_settings, prediction_decision, save_settings
from src import route_projection_app as app_module
from src import projection_workflow as workflow
from tools.collector.fixed_roi_collector import CaptureRegion
from src.floor_indicator import FloorDetection
from src.projection.route_overlay import RouteOverlayAsset, warp_route_overlay


def test_route_version_is_selected_per_map_and_difficulty(tmp_path) -> None:
    profiles = []
    for map_id in ("first", "second"):
        clean = tmp_path / map_id / 'clean/layout_05'
        clean.mkdir(parents=True)
        calibration = tmp_path / map_id / "calibration"
        for variant_id in ("route_lhp_solo", "route_02"):
            directory = calibration / variant_id / "layout_05"
            directory.mkdir(parents=True)
            (directory / "calibration.json").write_text(json.dumps({
                "layout": "layout_05", "variant_id": variant_id,
                "route": f"{variant_id}.png",
                "floors": {floor: {"status": "PASS"} for floor in ("2nd", "1st", "b1")},
            }), encoding="utf-8")
        profiles.append({
            "id": map_id, "name": map_id,
            "difficulties": [{"id": "nightmare", "name": "噩梦",
                              "clean_root": str(clean.parent),
                              "calibration_root": str(calibration), "model": str(tmp_path / "model.pt")}],
        })
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "maps": profiles}), encoding="utf-8")
    catalog = MapCatalog(catalog_path)
    saved = {"first/nightmare": {"layout_05": "route_02"}}
    first = catalog.selected_variant(catalog.profile("first", "nightmare"), "layout_05", saved)
    second = catalog.selected_variant(catalog.profile("second", "nightmare"), "layout_05", saved)
    assert first is not None and first.variant_id == "route_02"
    assert second is not None and second.variant_id == "route_lhp_solo"


def test_layout_confidence_has_no_exploration_requirement() -> None:
    threshold = {"confidence_threshold": 0.70, "margin_threshold": 0.20}
    confident = prediction_decision([("layout_05", 0.85), ("layout_07", 0.10)], threshold)
    uncertain = prediction_decision([("layout_05", 0.42), ("layout_07", 0.37)], threshold)
    assert confident["accepted"] and confident["layout"] == "layout_05"
    assert not uncertain["accepted"] and uncertain["layout"] is None
    assert "explored_ratio" not in confident


@pytest.mark.parametrize('value,expected', [(None,150),('invalid',150),(True,150),(-4,0),(900,500),(0,0),(35,35),(65,65)])
def test_capture_delay_settings_normalize_and_roundtrip(tmp_path,value,expected):
    path=tmp_path/'settings.json';catalog=MapCatalog()
    path.write_text('{}',encoding='utf-8')
    assert load_settings(catalog,path)['map_capture_delay_ms']==150
    path.write_text(json.dumps({'map_capture_delay_ms':value}),encoding='utf-8')
    settings=load_settings(catalog,path)
    assert settings['map_capture_delay_ms']==expected
    save_settings(settings,path)
    assert load_settings(catalog,path)['map_capture_delay_ms']==expected


def test_game_map_capture_tracks_window_position_and_size() -> None:
    roi = [901, 185, 1180, 1050]
    reference = [2560, 1440]
    moved = app_module.game_capture_region(CaptureRegion(100, 50, 2560, 1440), roi, reference)
    assert moved.as_list() == [1001, 235, 1180, 1050]
    resized = app_module.game_capture_region(CaptureRegion(100, 50, 1920, 1080), roi, reference)
    assert resized.as_list() == [776, 189, 885, 787]
    with pytest.raises(ValueError, match="完全位于"):
        app_module.game_capture_region(CaptureRegion(0, 0, 1920, 1080), [2000, 0, 600, 300], reference)


def test_projection_settings_keep_map_roi_inside_game_client(tmp_path) -> None:
    catalog = MapCatalog()
    path = tmp_path / "settings.json"
    settings = load_settings(catalog, path)
    assert settings["capture_reference_size"] == [2560, 1440]
    settings["capture_roi"] = [2000, 0, 600, 300]
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="完全位于"):
        load_settings(catalog, path)


def test_map_key_toggles_only_on_each_new_press(monkeypatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"], "auto_project": True}
            self.settings_dialog = None
            self.map_state = app_module.MapToggleState()
            self.map_generation = 0
            self.auto_attempt = 0
            self.pending_auto = False
            self.closed = False
            self.map_probe_pending = False
            self.hidden = 0
            self.scheduled = []
            self.requests = []

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1
        def after(self, delay, callback) -> None: self.scheduled.append((delay, callback))
        def request_work(self, mode, *, auto) -> None: self.requests.append((mode, auto))

    pressed = {"value": False}
    monkeypatch.setattr(app_module.time, "perf_counter", lambda: 100.)
    monkeypatch.setattr(app_module, "map_key_down", lambda _key: pressed["value"])
    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app_module.RouteProjectionApp._handle_map_key(app)
    pressed["value"] = True
    app_module.RouteProjectionApp._handle_map_key(app)
    assert app.map_state.is_open and app.hidden == 1 and len(app.scheduled) == 1
    assert app.scheduled[0][0] == 150  # User-requested initial map rendering delay.
    assert app.requests == []  # Key press alone must never start registration.
    app_module.RouteProjectionApp._handle_map_key(app)  # key repeat / still down
    pressed["value"] = False
    app_module.RouteProjectionApp._handle_map_key(app)  # release keeps map open
    assert app.map_state.is_open and app.hidden == 1 and app.map_generation == 1
    assert app.map_probe_pending
    pressed["value"] = True
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_state.is_open and app.hidden == 2 and app.map_generation == 2
    assert len(app.scheduled) == 1 and app.requests == []
    assert not app.map_probe_pending  # Closing cancels the pending opening probe.
    app.scheduled[0][1]()
    assert app.requests == []


def test_map_presence_must_confirm_open_before_projection(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "save_map_presence_diagnostic", lambda *_args, **_kwargs: None)

    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"auto_project": True, "game_window_titles": ["第五人格"]}
            self.map_state = app_module.MapToggleState(is_open=True)
            self.map_generation = 7
            self.closed = False
            self.busy = False
            self.floor_monitor_active = False
            self.map_probe_pending = True
            self.map_presence_started_at = time.perf_counter() - 0.25
            self.hidden = 0
            self.requests = []
            self.scheduled = []

        def _set_status(self, *_args) -> None: pass
        def _refresh_status(self) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1
        def after(self, delay, callback) -> None: self.scheduled.append((delay, callback))
        def request_work(self, mode, *, auto) -> None: self.requests.append((mode, auto))

    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    from test_immediate_map_probe import sample
    open_result = app_module.MapPresence("open", 0.95, (0, 0))
    app_module.RouteProjectionApp._apply_map_presence_probe(
        app, 6, 0, True, None, None, None, None, sample())
    assert app.requests == []  # Stale result from a previous key press.
    app_module.RouteProjectionApp._apply_map_presence_probe(
        app, 7, 0, True, None, None, None, None, sample())
    assert app.map_state.is_open and app.requests == [("project", True)]
    assert app.map_presence_elapsed_ms is not None and app.map_presence_elapsed_ms >= 250

    app.map_generation = 8
    app.map_state.is_open = True
    app.requests.clear()
    app_module.RouteProjectionApp._apply_map_presence_probe(
        app, 8, 0, True, open_result, None, None, None)
    assert app.requests == [] and len(app.scheduled) == 1  # A marker alone cannot authorize opening.
    app.map_presence_started_at = time.perf_counter() - .501
    app_module.RouteProjectionApp._apply_map_presence_probe(
        app, 8, 1, True, open_result, None, None, None)
    assert not app.map_state.is_open and app.requests == [] and app.hidden == 1
    assert not app.map_probe_pending


def test_uncertain_map_presence_stops_projection(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "save_map_presence_diagnostic", lambda *_args, **_kwargs: None)

    class FakeApp:
        map_generation = 2
        closed = False
        map_state = app_module.MapToggleState(is_open=True)
        floor_monitor_active = False
        map_probe_pending = True
        map_presence_text = ""

        def __init__(self) -> None:
            self.settings = {}
            self.hidden = 0
            self.scheduled = []

        def _refresh_status(self) -> None: pass
        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1
        def after(self, delay, callback) -> None: self.scheduled.append((delay, callback))

    app = FakeApp()
    uncertain = app_module.MapPresence("uncertain", 0.61, (1, 0))
    app.map_presence_started_at = 100.0
    # Retry count is now deadline-based: prove the same stop behavior at 500 ms.
    for attempt, elapsed in enumerate((.020, .250, .501)):
        monkeypatch.setattr(app_module.time, "perf_counter", lambda elapsed=elapsed: 100.0 + elapsed)
        app_module.RouteProjectionApp._apply_map_presence_probe(
            app, 2, attempt, True, uncertain, None, None, None)
    assert not app.map_state.is_open and app.hidden == 1
    assert len(app.scheduled) == 2


def test_escape_invalidates_pending_map_probe_after_close_press(monkeypatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"]}
            self.settings_dialog = None
            self.map_state = app_module.MapToggleState(is_open=False)
            self.map_generation = 4
            self.map_probe_pending = True
            self.hidden = 0

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1

    monkeypatch.setattr(app_module, "map_key_down", lambda _key: False)
    monkeypatch.setattr(app_module, "escape_key_down", lambda: True)
    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_probe_pending and app.map_generation == 5
    assert not app.map_state.is_open and app.hidden == 1


def test_focus_suppression_does_not_create_late_toggle() -> None:
    state = app_module.MapToggleState()
    assert state.observe(True, enabled=False) is None
    assert state.observe(True, enabled=True) is None
    assert state.observe(False, enabled=True) is None
    assert state.observe(True, enabled=True) == "opened"


def test_escape_only_closes_an_open_map() -> None:
    state = app_module.MapToggleState()
    assert not state.observe_escape(True)
    assert not state.is_open
    state.observe_escape(False)
    assert state.observe(True) == "opened"
    assert state.observe_escape(True)
    assert not state.is_open
    assert not state.observe_escape(True)
    state.observe(False)
    assert state.observe(True) == "opened"


def test_escape_hides_overlay_and_invalidates_pending_projection(monkeypatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"], "auto_project": True}
            self.settings_dialog = None
            self.map_state = app_module.MapToggleState(is_open=True)
            self.map_generation = 3
            self.map_probe_pending = False
            self.hidden = 0

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1

    monkeypatch.setattr(app_module, "map_key_down", lambda _key: False)
    monkeypatch.setattr(app_module, "escape_key_down", lambda: True)
    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_state.is_open
    assert app.hidden == 1
    assert app.map_generation == 4
    app_module.RouteProjectionApp._handle_map_key(app)
    assert app.hidden == 1


def test_floor_change_hides_old_overlay_before_reprojecting() -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.map_generation = 4
            self.map_state = app_module.MapToggleState(is_open=True)
            self.busy = False
            self.overlay = SimpleNamespace(visible=True)
            self.floor_var = SimpleNamespace(get=lambda: "1st")
            self.floor_probe_busy = True
            self.floor_probe_candidate = None
            self.floor_probe_count = 0
            self.floor_probe_unknown_count = 0
            self.floor_probe_needs_projection = False
            self.floor_monitor_active = True
            self.hidden = 0
            self.requests = []

        def hide_overlay(self) -> None:
            self.overlay.visible = False
            self.hidden += 1

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def request_work(self, mode, *, auto) -> None: self.requests.append((mode, auto))

    app = FakeApp()
    detection = FloorDetection("2nd", {"b1": 0.1, "1st": 0.2, "2nd": 0.9}, 0.7, "2nd", 40, None)
    app_module.RouteProjectionApp._apply_floor_probe(app, 4, detection, None)
    assert app.hidden == 1 and app.requests == []
    app_module.RouteProjectionApp._apply_floor_probe(app, 4, detection, None)
    assert app.requests == [("project", True)]


@pytest.mark.parametrize("scores", [(0.95, 0.03), (0.42, 0.37), (0.05, 0.05), (0.0, 0.0), (0.75, 0.70)])
def test_recognition_requires_confirmation_before_lock(monkeypatch, scores) -> None:
    class Variable:
        def __init__(self) -> None: self.value = None
        def set(self, value) -> None: self.value = value

    class Label:
        def configure(self, **_kwargs) -> None: pass

    class FakeApp:
        def __init__(self) -> None:
            self.map_generation = 0
            self.locked_layout = "layout_03"
            self.registration_priors = {("map", "layout_03", "b1"): object()}
            self.registration_soft_priors = {("map", "layout_03", "b1"): object()}
            self.registration_evidence = {("map", "layout_03", "b1"): object()}
            self.layout_var = Variable()
            self.decision_label = Label()
            self.roi_label = Label()
            self.path_var = Variable()
            self.scheduled = []
            self.toasts = []
            self.toast = SimpleNamespace(show_message=lambda *args, **kwargs:
                                         self.toasts.append((args, kwargs)))

        def _set_preview(self, *_args) -> None: pass
        def _update_ranking(self, *_args) -> None: pass
        def _update_assets(self) -> None: pass
        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def _preload_locked_layout(self, *_args) -> None: pass
        def after(self, delay, callback) -> None: self.scheduled.append((delay, callback))

    monkeypatch.setattr(workflow, "predict_image", lambda *_args:
                        [("layout_05", scores[0]), ("layout_07", scores[1])])
    result = workflow.process_screenshot(
        np.zeros((8, 8, 3), np.uint8), catalog=None, profile=SimpleNamespace(key=None),
        model_bundle=(None, {}, None), floor="2nd", layout_override=None,
        route_versions={}, opacity=1.0, prepared_cache={}, asset_cache={}, recognize_only=True,
    )
    assert result.accepted and result.selected_layout == "layout_05"
    assert result.reasons == []
    assert result.decision["accepted"] == (scores[0] >= 0.70 and scores[0] - scores[1] >= 0.20)
    app = FakeApp()
    monkeypatch.setattr(app_module.messagebox, "confirm_layout", lambda *_args, **_kwargs: (None, False))
    app_module.RouteProjectionApp._apply_result(app, "recognize", False, 0, [0, 0, 8, 8], result, None)
    assert app.scheduled[-1][0] == 0  # Confirm immediately; the toast keeps its original wording.
    assert app.toasts[-1][0][0] == "布局识别成功"
    app.scheduled[-1][1]()
    assert app.locked_layout == "layout_03"
    assert app.layout_var.value == "layout_03"
    monkeypatch.setattr(app_module.messagebox, "confirm_layout", lambda *_args, **_kwargs: ("layout_05", False))
    app_module.RouteProjectionApp._apply_result(app, "recognize", False, 0, [0, 0, 8, 8], result, None)
    assert app.locked_layout == "layout_03"
    app.scheduled[-1][1]()
    assert app.locked_layout == "layout_05"
    assert app.registration_priors == {}
    assert app.registration_evidence == {}

    # Skip-confirmation preference applies only to confident results. Weak
    # candidates must still permit choosing the runner-up, even at zero confidence.
    skip_app = FakeApp()
    skip_app.settings = {"skip_layout_confirmation": True}
    choices = []
    monkeypatch.setattr(app_module.messagebox, "confirm_layout", lambda candidates, *_args, **_kwargs:
                        choices.append(candidates) or ("layout_07", False))
    app_module.RouteProjectionApp._apply_result(skip_app, "recognize", False, 0, [0, 0, 8, 8], result, None)
    skip_app.scheduled[-1][1]()
    if result.decision["accepted"]:
        assert skip_app.locked_layout == "layout_05" and not choices
    else:
        assert skip_app.locked_layout == "layout_07"
        assert choices == [result.decision["ranking"]]
        assert skip_app.registration_priors == {} and skip_app.registration_evidence == {}

    failed = SimpleNamespace(**{**vars(result), "accepted": False,
                                "reasons": ["画面结构过少"]})
    count = len(app.scheduled)
    app_module.RouteProjectionApp._apply_result(app, "recognize", False, 0, [0, 0, 8, 8], failed, None)
    assert len(app.scheduled) == count
    assert app.toasts[-1][0][0] == "布局识别失败"
    assert "画面结构过少" in app.toasts[-1][0][1]

    # The captured result remains confirmable when the player closes the map
    # while the success toast is fading.
    closed_map_app = FakeApp()
    closed_map_app.map_state = SimpleNamespace(is_open=True)
    calls = []
    monkeypatch.setattr(app_module.messagebox, "confirm_layout", lambda *_args, **kwargs:
                        calls.append(kwargs) or ("layout_05", False))
    app_module.RouteProjectionApp._apply_result(
        closed_map_app, "recognize", False, 0, [0, 0, 8, 8], result, None)
    closed_map_app.map_generation = 1
    closed_map_app.map_state.is_open = False
    closed_map_app.scheduled[-1][1]()
    assert closed_map_app.locked_layout == "layout_05"
    assert calls[-1]["screen_roi"] == [0, 0, 8, 8]

    # Closing the map just after capture, before the worker result arrives,
    # must also retain the captured layout for confirmation.
    closed_before_result = FakeApp()
    closed_before_result.map_generation = 1
    closed_before_result.map_state = SimpleNamespace(is_open=False)
    app_module.RouteProjectionApp._apply_result(
        closed_before_result, "recognize", False, 0, [0, 0, 8, 8], result, None)
    closed_before_result.scheduled[-1][1]()
    assert closed_before_result.locked_layout == "layout_05"
    confirmed_count = len(calls)

    # A second map transition invalidates an older recognition result.
    reopened_app = FakeApp()
    reopened_app.map_state = SimpleNamespace(is_open=True)
    app_module.RouteProjectionApp._apply_result(
        reopened_app, "recognize", False, 0, [0, 0, 8, 8], result, None)
    reopened_app.map_generation = 2
    reopened_app.scheduled[-1][1]()
    assert reopened_app.locked_layout == "layout_03"
    assert len(calls) == confirmed_count


def test_recognition_without_candidates_cannot_be_confirmed(monkeypatch) -> None:
    kwargs = dict(catalog=None, profile=SimpleNamespace(key=None), floor="2nd", layout_override=None,
                  route_versions={}, opacity=1.0, prepared_cache={}, asset_cache={}, recognize_only=True)
    screenshot = np.zeros((8, 8, 3), np.uint8)
    result = workflow.process_screenshot(screenshot, model_bundle=None, **kwargs)
    assert not result.accepted and result.selected_layout is None
    monkeypatch.setattr(workflow, "predict_image", lambda *_args: [])
    with pytest.raises(ValueError, match="模型未返回布局排名"):
        workflow.process_screenshot(screenshot, model_bundle=(None, {}, None), **kwargs)


def test_low_confidence_candidate_is_not_displayed_as_structure_confirmed() -> None:
    labels = []
    decision = prediction_decision([("layout_05", 0.05), ("layout_07", 0.05)], {})
    app = SimpleNamespace(last_decision=None, locked_layout=None, rank_widgets=[],
                          decision_label=SimpleNamespace(configure=lambda **kwargs: labels.append(kwargs)))
    result = SimpleNamespace(decision=decision, selected_layout="layout_05", accepted=True, registration=None)
    app_module.RouteProjectionApp._update_ranking(app, result)
    assert "仅有候选：layout_05" in labels[-1]["text"]
    assert "5.0%" in labels[-1]["text"] and labels[-1]["fg"] == app_module.AMBER


def test_non_game_window_does_not_count_as_game_focus(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "foreground_window_title", lambda: "Codex")
    assert not app_module.game_window_is_foreground(["第五人格", "Identity V"])
    monkeypatch.setattr(app_module, "foreground_window_title", lambda: "第五人格")
    assert app_module.game_window_is_foreground(["第五人格", "Identity V"])


def test_route_coverage_is_uncomputed_until_registration_passes() -> None:
    registration = SimpleNamespace(accepted=False, validation={
        "structure_precision": 0.862, "semantic_precision": 0.704,
        "edge_median_error_px": 10.5,
    })
    result = SimpleNamespace(registration=registration, overlay_validation=None, timings_ms={})
    text = app_module.format_projection_metrics(result)
    assert "结构 86.2%" in text and "语义 70.4%" in text
    assert "路线覆盖：未计算" in text
    assert "路线覆盖：0.0%" not in text


def test_opacity_percent_and_projected_alpha_agree() -> None:
    assert app_module.opacity_to_percent(0.72) == 72
    assert app_module.percent_to_opacity(72) == 0.72
    assert app_module.percent_to_opacity(0) == 0.0
    assert app_module.percent_to_opacity(100) == 1.0
    rgba = np.full((16, 16, 4), 200, dtype=np.uint8)
    asset = RouteOverlayAsset("layout_05", "2nd", "route_lhp_solo", rgba, np.eye(3),
                              Path("calibration.json"), Path("asset.png"))
    opaque, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 1.0)
    half, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 0.5)
    invisible, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 0.0)
    assert opaque[8, 8, 3] == 200
    assert half[8, 8, 3] == 100
    assert invisible[8, 8, 3] == 0
    assert np.array_equal(opaque[8, 8, :3], half[8, 8, :3])


def test_locked_layout_skips_classifier(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "predict_image", lambda *_args: (_ for _ in ()).throw(
        AssertionError("locked layout must not run classification")))
    catalog = SimpleNamespace(layouts=lambda _profile: [])
    profile = SimpleNamespace(key="test", map_id="test", difficulty_id="test")
    result = workflow.process_screenshot(
        np.zeros((16, 16, 3), np.uint8), catalog=catalog, profile=profile,
        model_bundle=(object(), {}, "cpu"), floor="2nd", layout_override="layout_05",
        route_versions={}, opacity=0.72, prepared_cache={}, asset_cache={},
    )
    assert result.decision is None
    assert "classification" not in result.timings_ms
    assert result.selected_layout == "layout_05"


def test_all_settings_are_persisted_without_save_button(tmp_path) -> None:
    catalog = MapCatalog()
    path = tmp_path / "settings.json"
    initial = load_settings(catalog, path)
    initial['route_versions'] = {"a_bringer_of_doom/nightmare": {
        "layout_01": "route_lhp_solo", "layout_02": "route_clean"}}
    app = SimpleNamespace(catalog=catalog, settings_path=path, settings=initial)

    def apply_settings(updated) -> None:
        app.settings = updated
        save_settings(updated, path)

    app.apply_settings = apply_settings
    status = SimpleNamespace(value="", set=lambda value: setattr(status, "value", value))
    opacity_status = SimpleNamespace(value="", set=lambda value: setattr(opacity_status, "value", value))
    values = {
        "map_key": "J", "window_titles": "第五人格, Identity V",
        "recognize_hotkey": "F7", "project_hotkey": "F9",
        "auto_project": False, "auto_floor": True, "opacity": 83.0,
        "roi": "900, 180, 1180, 1050", "reference_size": "2560, 1440",
    }
    dialog = SimpleNamespace(
        app=app,
        vars={name: SimpleNamespace(get=lambda value=value: value) for name, value in values.items()},
        save_status=status,
        opacity_text=opacity_status,
        _settings_save_job=None,
        _settings_dirty=True,
    )
    dialog._collect_settings = lambda: app_module.SettingsDialog._collect_settings(dialog)
    assert app_module.SettingsDialog._persist_settings(dialog)
    persisted = load_settings(catalog, path)
    assert persisted["map_key"] == "J"
    assert persisted["recognize_hotkey"] == "F7"
    assert persisted["project_hotkey"] == "F9"
    assert persisted["auto_project"] is False
    assert persisted["auto_floor"] is True
    assert persisted["capture_roi"] == [900, 180, 1180, 1050]
    assert persisted["game_window_titles"] == ["第五人格", "Identity V"]
    assert persisted["route_versions"]["a_bringer_of_doom/nightmare"]["layout_01"] == "route_lhp_solo"
    assert persisted["route_versions"] == initial["route_versions"]
    assert persisted["opacity"] == 0.83
    assert status.value == "已自动保存并立即生效"
    assert opacity_status.value == "83% 已保存"
