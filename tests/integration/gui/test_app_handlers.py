from __future__ import annotations
from types import SimpleNamespace
from src.app_catalog import MapCatalog, load_settings, save_settings
from src import route_projection_app as app_module
from tests.support.key_events import record


def test_map_key_opens_on_release_and_closes_early_on_next_press(monkeypatch, source) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"], "auto_project": True}
            self.settings_page = None
            self.map_state = app_module.MapToggleState()
            self.map_generation = 0
            self.auto_attempt = 0
            self.pending_auto = False
            self.closed = False
            self.projection_delay_pending = False
            self.hidden = 0
            self.scheduled = []
            self.requests = []

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1
        def after(self, delay, callback) -> None: self.scheduled.append((delay, callback))
        def request_work(self, mode, *, auto) -> None: self.requests.append((mode, auto))
        def _handle_map_key(self) -> None: app_module.RouteProjectionApp._handle_map_key(self)

    monkeypatch.setattr(app_module.time, "perf_counter", lambda: 100.)
    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app.map_keys = source
    app_module.RouteProjectionApp._handle_map_key(app)
    record(source, True)
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_state.is_open and app.hidden == 0 and app.scheduled == []
    assert app.requests == []  # Key press alone must never start registration.
    record(source, True)
    app_module.RouteProjectionApp._handle_map_key(app)  # key repeat / still down
    record(source, False)
    app_module.RouteProjectionApp._handle_map_key(app)  # release starts opening delay
    assert app.map_state.is_open and app.hidden == 1 and app.map_generation == 1
    assert app.scheduled[0][0] == 150  # Delay starts only after release.
    assert app.projection_delay_pending
    record(source, True)
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_state.is_open and app.hidden == 2 and app.map_generation == 2
    assert len(app.scheduled) == 1 and app.requests == []
    assert not app.projection_delay_pending  # Closing cancels the pending opening probe.
    app.scheduled[0][1]()
    assert app.requests == []


def test_escape_invalidates_pending_projection_after_close_press(monkeypatch, source) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"]}
            self.settings_page = None
            self.map_state = app_module.MapToggleState(is_open=False)
            self.map_generation = 4
            self.projection_delay_pending = True
            self.hidden = 0

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1

    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app.map_keys = source
    record(source, True, vk=0x1B)
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.projection_delay_pending and app.map_generation == 5
    assert not app.map_state.is_open and app.hidden == 1


def test_escape_hides_overlay_and_invalidates_pending_projection(monkeypatch, source) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.settings = {"map_key": "G", "game_window_titles": ["第五人格"], "auto_project": True}
            self.settings_page = None
            self.map_state = app_module.MapToggleState(is_open=True)
            self.map_generation = 3
            self.projection_delay_pending = False
            self.hidden = 0

        def _set_status(self, *_args) -> None: pass
        def _log(self, *_args) -> None: pass
        def hide_overlay(self) -> None: self.hidden += 1

    monkeypatch.setattr(app_module, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
    app = FakeApp()
    app.map_keys = source
    record(source, True, vk=0x1B)
    app_module.RouteProjectionApp._handle_map_key(app)
    assert not app.map_state.is_open
    assert app.hidden == 1
    assert app.map_generation == 4
    app_module.RouteProjectionApp._handle_map_key(app)
    assert app.hidden == 1


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
        "reference_floor_hotkey": "K",
        "chest_gold_add_hotkey": "U", "chest_gold_subtract_hotkey": "I",
        "chest_purple_add_hotkey": "O", "chest_purple_subtract_hotkey": "P",
        "chest_counter_visible": False, "capture_exclusion_enabled": False,
        "auto_project": False, "auto_floor": True, "opacity": 83.0,
        "layout_recognition_backend": "pure_structure",
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
    dialog._collect_settings = lambda: app_module.SettingsPage._collect_settings(dialog)
    assert app_module.SettingsPage._persist_settings(dialog)
    persisted = load_settings(catalog, path)
    assert persisted["map_key"] == "J"
    assert persisted["recognize_hotkey"] == "F7"
    assert persisted["project_hotkey"] == "F9"
    assert persisted['chest_gold_add_hotkey'] == 'U'
    assert persisted['chest_purple_subtract_hotkey'] == 'P'
    assert not persisted['chest_counter_visible'] and not persisted['capture_exclusion_enabled']
    assert persisted["auto_project"] is False
    assert persisted["auto_floor"] is False
    assert persisted["capture_roi"] == [900, 180, 1180, 1050]
    assert persisted["game_window_titles"] == ["第五人格", "Identity V"]
    assert persisted["route_versions"]["a_bringer_of_doom/nightmare"]["layout_01"] == "route_lhp_solo"
    assert persisted["route_versions"] == initial["route_versions"]
    assert persisted["opacity"] == 0.83
    assert status.value == "已自动保存并立即生效"
    assert opacity_status.value == "83% 已保存"
