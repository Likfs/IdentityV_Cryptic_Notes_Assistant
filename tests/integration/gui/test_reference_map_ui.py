from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication, QLabel
from src import route_projection_app as app
from src.app_catalog import load_settings
from src.route_projection_qt_ui import LayoutConfirmationDialog


def ready(window, floor="all", generation=None):
    window._apply_reference_ready(window.map_generation if generation is None else generation,
                                  [100, 100, 1180, 1050], floor, window.profile.key, window.locked_layout)


def feedback(window, *, accepted=False, floor="all"):
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
    assert "三层完整原图" in window.reference_overlay.caption.text()
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
    assert window.toast.title.text() == "模板匹配候选已投影"
    press_h(window, monkeypatch)
    assert window.reference_overlay.visible  # H remains useful after successful registration.


def test_closed_map_drops_late_reference_and_projection_results(window, source, monkeypatch):
    generation = window.map_generation
    ready(window)
    assert window.reference_overlay.visible
    from tests.support.key_events import install, record
    install(window, source)
    record(source, True)
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


def test_reference_versions_opacity_and_focus(window, monkeypatch):
    requests = []
    monkeypatch.setattr(window, 'request_work',
                        lambda mode, **kwargs: requests.append((mode, kwargs)))
    ready(window)
    images = []
    for version in ("route_clean", "route_lhp_solo", "route_zhanshi_fast"):
        window.route_combo.setCurrentIndex(window.route_combo.findData(version))
        ready(window, floor="all")  # A source switch invalidates the previous context.
        assert window.reference_overlay.visible
        images.append(window.reference_overlay.label.pixmap().toImage())
    assert requests == [('project', {'auto': False})] * 3
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
    assert app.messagebox.confirm_layout([{"layout": "layout_11", "score": .95}], root,
                                        screen_roi=[0, 0, 800, 600]) == "layout_11"
    assert observed


def test_stale_confirmation_does_not_lock_or_save_preference(window, monkeypatch):
    result = SimpleNamespace(selected_layout="layout_13", decision={
        "ranking": [{"layout": "layout_13", "score": .95}], "score_is_probability": False})
    def stale_dialog(*_args, **_kwargs):
        window.map_generation += 2
        return "layout_13"
    monkeypatch.setattr(app.messagebox, "confirm_layout", stale_dialog)
    window._pending_confirmation = (window.map_generation, result, window.session_id, window.profile.key)
    window._confirm_recognition(window.map_generation, result)
    assert window.locked_layout == "layout_11"
    assert not load_settings(window.catalog, window.settings_path)["skip_layout_confirmation"]


def test_map_toasts_do_not_appear_after_close(window):
    window.map_state.is_open = False
    window._map_toast("地图暂未配准", "保持地图打开", success=False)
    assert not window.toast.isVisible()
