from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox

from src import route_projection_app as app
from src.app_catalog import load_settings, prediction_decision
from src.route_projection_qt_ui import LayoutConfirmationDialog


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(app, "exclude_window_from_capture", lambda _: False)
    monkeypatch.setattr(app, "game_window_is_foreground", lambda _: True)
    monkeypatch.setattr(app, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app, "map_key_down", lambda _: False)
    monkeypatch.setattr(app.RouteProjectionApp, "after", lambda *_: None)
    monkeypatch.setattr(app, "GlobalActionHotkeys", lambda *_: SimpleNamespace(start=lambda: None, stop=lambda: None))
    qt = QApplication.instance() or QApplication([])
    # Floor-specific reference maps belong to the structure backend.
    (tmp_path / "settings.json").write_text('{"registration_backend": "cpu"}', encoding="utf-8")
    instance = app.RouteProjectionApp(settings_path=tmp_path / "settings.json")
    instance.locked_layout = "layout_11"
    instance.layout_var.set("layout_11")
    instance.map_state.is_open = True
    instance._update_assets()
    yield instance
    instance.close()
    qt.processEvents()


def ready(window, floor="2nd", generation=None):
    window._apply_reference_ready(window.map_generation if generation is None else generation,
                                  [100, 100, 1180, 1050], floor, window.profile.key, window.locked_layout)


def feedback(window, *, accepted=False, floor="2nd"):
    result = SimpleNamespace(accepted=accepted, selected_layout=window.locked_layout,
                             profile_key=window.profile.key, floor=floor, registration=None,
                             reasons=[] if accepted else ["可见结构不足"])
    window._finish_map_feedback(window.map_generation, [100, 100, 1180, 1050], result)


def press_h(window, monkeypatch):
    monkeypatch.setattr(app, "map_key_down", lambda key: key == "H")
    window._handle_reference_key()
    monkeypatch.setattr(app, "map_key_down", lambda _: False)
    window._handle_reference_key()


def test_reference_lifecycle_and_user_hide_survives_result(window, monkeypatch):
    ready(window)
    assert window.reference_overlay.visible
    assert "2 楼" in window.reference_overlay.caption.text()
    press_h(window, monkeypatch)
    assert not window.reference_overlay.visible
    feedback(window)
    assert not window.reference_overlay.visible  # A late failure must respect H.
    assert window.toast.title.text() == "地图暂未配准"
    press_h(window, monkeypatch)
    assert window.reference_overlay.visible
    window.reference_override = None
    feedback(window, accepted=True)
    assert not window.reference_overlay.visible
    assert window.toast.title.text() == "地图配准成功"
    press_h(window, monkeypatch)
    assert window.reference_overlay.visible  # H remains useful after successful registration.


def test_closed_map_drops_late_reference_and_projection_results(window, monkeypatch):
    generation = window.map_generation
    ready(window)
    assert window.reference_overlay.visible
    monkeypatch.setattr(app, "map_key_down", lambda key: key == window.settings["map_key"])
    monkeypatch.setattr(app, "escape_key_down", lambda: False)
    window._handle_map_key()
    assert not window.map_state.is_open
    assert window.map_generation == generation + 1
    assert not window.reference_overlay.visible
    ready(window, generation=generation)
    assert window.reference_context is None
    assert not window.reference_overlay.visible
    window._apply_result("project", False, generation, [100, 100, 1180, 1050],
                         SimpleNamespace(accepted=True), None)
    assert window.last_result is None
    assert not window.toast.isVisible()
    window.map_state.is_open = True
    ready(window, generation=generation)
    assert not window.reference_overlay.visible  # Reopening doesn't revive the old result.


def test_unknown_floor_hides_reference_and_new_floor_changes_image(window):
    ready(window)
    original = window.reference_overlay.label.pixmap().toImage()
    feedback(window, floor="unknown")
    assert not window.reference_overlay.visible and window.reference_context is None
    ready(window, floor="b1")
    assert window.reference_overlay.visible
    assert "地下室" in window.reference_overlay.caption.text()
    assert original != window.reference_overlay.label.pixmap().toImage()


def test_reference_versions_opacity_and_focus(window, monkeypatch):
    ready(window)
    images = []
    for version in ("route_clean", "route_lhp_solo", "route_zhanshi_fast"):
        window.route_combo.setCurrentIndex(window.route_combo.findData(version))
        assert window.reference_overlay.visible
        images.append(window.reference_overlay.label.pixmap().toImage())
    assert images[0] != images[1] and images[1] != images[2]
    window.reference_opacity_slider.setValue(43)
    assert abs(window.reference_overlay.windowOpacity() - .43) < .01
    assert load_settings(window.catalog, window.settings_path)["reference_opacity"] == .43
    assert window.reference_overlay.visible
    monkeypatch.setattr(app, "game_window_is_foreground", lambda _: False)
    window._handle_reference_key()
    assert not window.reference_overlay.visible
    monkeypatch.setattr(app, "game_window_is_foreground", lambda _: True)
    window._handle_reference_key()
    assert window.reference_overlay.visible


def test_confirmation_contains_the_matched_entrance_image(window, monkeypatch):
    observed = []

    def inspect(dialog):
        preview = dialog.findChild(QLabel, "entrance_preview")
        assert preview is not None and not preview.pixmap().isNull()
        assert dialog.width() <= QApplication.primaryScreen().availableGeometry().width()
        observed.append(True)
        dialog.close()
        return LayoutConfirmationDialog.DialogCode.Accepted

    monkeypatch.setattr(LayoutConfirmationDialog, "exec", inspect)
    root = window.profile.entrance_root
    assert app.messagebox.confirm_layout([{"layout": "layout_11", "confidence": .95}], root,
                                        screen_roi=[0, 0, 800, 600]) == ("layout_11", False)
    assert observed


def test_confirmation_locks_chosen_candidate_and_remembers_preference(window, monkeypatch):
    result = SimpleNamespace(selected_layout="layout_13", decision=prediction_decision(
        [("layout_13", .765), ("layout_07", .234)], {}))
    monkeypatch.setattr(window, "_preload_locked_layout", lambda *_: None)
    monkeypatch.setattr(app.messagebox, "confirm_layout", lambda *_args, **_kwargs: ("layout_07", True))
    window.registration_priors["old"] = object()
    window._pending_confirmation = (window.map_generation, result, window.session_id, window.profile.key)
    window._confirm_recognition(window.map_generation, result)
    assert window.locked_layout == "layout_07"
    assert window.layout_var.get() == "layout_07"
    assert not window.registration_priors
    assert load_settings(window.catalog, window.settings_path)["skip_layout_confirmation"] is True

    def unexpected_dialog(*_args, **_kwargs):
        raise AssertionError("Automatic locking should not open a dialog")
    monkeypatch.setattr(app.messagebox, "confirm_layout", unexpected_dialog)
    window._pending_confirmation = (window.map_generation, result, window.session_id, window.profile.key)
    window._confirm_recognition(window.map_generation, result)
    assert window.locked_layout == "layout_13"

    window.open_settings()
    window.settings_dialog.vars["skip_layout_confirmation"].set(False)
    assert window.settings_dialog._persist_settings()
    assert not load_settings(window.catalog, window.settings_path)["skip_layout_confirmation"]


def test_stale_confirmation_does_not_lock_or_save_preference(window, monkeypatch):
    result = SimpleNamespace(selected_layout="layout_13", decision={"confidence": .95})
    def stale_dialog(*_args, **_kwargs):
        window.map_generation += 2
        return "layout_13", True
    monkeypatch.setattr(app.messagebox, "confirm_layout", stale_dialog)
    window._pending_confirmation = (window.map_generation, result, window.session_id, window.profile.key)
    window._confirm_recognition(window.map_generation, result)
    assert window.locked_layout == "layout_11"
    assert not load_settings(window.catalog, window.settings_path)["skip_layout_confirmation"]


def test_map_toasts_do_not_appear_after_close(window):
    window.map_state.is_open = False
    window._map_toast("地图暂未配准", "保持地图打开", success=False)
    assert not window.toast.isVisible()
