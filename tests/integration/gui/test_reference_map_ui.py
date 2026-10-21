from tests.support.qt_wait import wait_for
from types import SimpleNamespace
from dataclasses import replace
import threading
import time
from PyQt6.QtWidgets import QApplication, QLabel
from src import route_projection_app as app
from src.app_catalog import load_settings
from src.route_projection_qt_ui import LayoutConfirmationDialog
import numpy as np
import pytest


def ready(window, floor="b1"):
    window.floor_combo.setCurrentIndex(window.floor_combo.findData(floor))
    if not window.reference_visible:
        window._toggle_reference_map()
    window._refresh_reference_map()
    wait_for(lambda: window.reference_overlay.visible)


def feedback(window, *, accepted=False, floor="b1"):
    result = SimpleNamespace(accepted=accepted, selected_layout=window.locked_layout,
                             profile_key=window.profile.key, floor=floor, registration=None,
                             variant_id=window.route_combo.currentData(),
                             layer=np.zeros((4, 4, 4), np.uint8) if accepted else None,
                             reasons=[] if accepted else ["可见结构不足"])
    window._finish_map_feedback(window.map_generation, [100, 100, 1180, 1050], result)


def press_h(window, monkeypatch):
    # Button-level tests use the same action as H; Raw Input delivery has dedicated cases below.
    window._toggle_reference_map()


def test_reference_lifecycle_and_user_hide_survives_result(window, monkeypatch):
    ready(window)
    assert window.reference_overlay.visible
    assert "地下室" in window.reference_overlay.caption.text()
    press_h(window, monkeypatch)
    assert not window.reference_overlay.visible
    feedback(window)
    assert not window.reference_overlay.visible  # A late failure must respect H.
    assert window.toast.title.text() == "地图暂未配准"
    press_h(window, monkeypatch)
    assert window.reference_overlay.visible
    feedback(window, accepted=True)
    assert window.reference_overlay.visible
    assert window.toast.title.text() == "模板匹配候选已投影"
    press_h(window, monkeypatch)
    assert not window.reference_overlay.visible  # Successful matches do not override manual H.


def test_closed_map_keeps_reference_but_drops_late_projection_results(window, source, monkeypatch):
    generation = window.map_generation
    ready(window)
    assert window.reference_overlay.visible
    from tests.support.key_events import install, record
    install(window, source)
    record(source, True)
    window._handle_map_key()
    assert not window.map_state.is_open
    assert window.map_generation == generation + 1
    assert window.reference_overlay.visible
    assert window.settings['reference_floor'] == 'b1'
    window._apply_result("project", False, generation, [100, 100, 1180, 1050],
                         SimpleNamespace(accepted=True), None)
    assert window.last_result is None
    assert not window.toast.isVisible()
    window.map_state.is_open = True
    window._refresh_reference_map()
    assert window.reference_overlay.visible  # Reference has its own lifetime and revision.


def test_reference_versions_opacity_and_focus(window, monkeypatch):
    requests = []
    monkeypatch.setattr(window, 'request_work',
                        lambda mode, **kwargs: requests.append((mode, kwargs)))
    ready(window)
    images = []
    for version in ("route_clean", "route_lhp_solo", "route_zhanshi_fast"):
        window.route_combo.setCurrentIndex(window.route_combo.findData(version))
        ready(window, floor="b1")  # A source switch invalidates the previous context.
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


def test_late_single_floor_reference_cannot_restore_previous_floor(window, monkeypatch):
    window.settings['reference_floor'] = 'b1'
    key = window._reference_image_key()
    window.settings['reference_floor'] = '2nd'
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: pytest.fail('old floor restored'))
    window._receive_work_event(('floor_reference', key, np.ones((10, 10, 3), np.uint8), None))
    assert getattr(window, '_floor_reference_image', None) is None


def test_delayed_single_floor_reference_respects_manual_hide(window, monkeypatch):
    key = window._reference_image_key()
    window.reference_visible = False
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: pytest.fail('manual hide ignored'))
    window._receive_work_event(('floor_reference', key, np.ones((10, 10, 3), np.uint8), None))
    assert not window.reference_overlay.visible


def test_hidden_by_default_and_hotkeys_work_with_map_closed_without_capture(window, monkeypatch):
    window.map_state.is_open = False
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Reference requested a screenshot'))
    assert not window.reference_visible and not window.reference_overlay.visible
    press_h(window, monkeypatch)
    wait_for(lambda: window.reference_overlay.visible)
    assert '当前参考楼层' in window.reference_overlay.caption.text()
    assert '手动' in window.reference_overlay.caption.text()
    window._next_reference_floor()
    wait_for(lambda: '1 楼' in window.reference_overlay.caption.text())
    assert window.settings['reference_floor'] == '1st'
    assert load_settings(window.catalog, window.settings_path)['reference_floor'] == '1st'
    press_h(window, monkeypatch)
    window.map_state.is_open = True
    window._refresh_reference_map()
    assert not window.reference_overlay.visible


def test_successful_projection_syncs_reference_floor_and_j_can_override(window):
    ready(window, floor='1st')
    feedback(window, accepted=True, floor='b1')
    wait_for(lambda: '地下室' in window.reference_overlay.caption.text())
    assert window.settings['reference_floor'] == 'b1'
    assert window.floor_combo.currentData() == 'b1'
    assert '最近配准' in window.reference_overlay.caption.text()
    window._next_reference_floor()
    wait_for(lambda: '2 楼' in window.reference_overlay.caption.text())
    assert window.settings['reference_floor'] == '2nd'
    assert '手动' in window.reference_overlay.caption.text()


def test_successful_projection_updates_hidden_reference_without_showing_it(window, monkeypatch):
    feedback(window, accepted=True, floor='1st')
    assert window.settings['reference_floor'] == '1st' and window.floor_combo.currentData() == '1st'
    assert load_settings(window.catalog, window.settings_path)['reference_floor'] == '1st'
    assert not window.reference_visible and not window.reference_overlay.visible
    press_h(window, monkeypatch)
    wait_for(lambda: window.reference_overlay.visible)
    assert '1 楼' in window.reference_overlay.caption.text()
    assert '最近配准' in window.reference_overlay.caption.text()


@pytest.mark.parametrize('change', ['failed', 'closed', 'generation', 'layout', 'profile', 'variant', 'floor'])
def test_invalid_projection_feedback_cannot_change_reference_floor(window, change):
    result = SimpleNamespace(accepted=True, floor='1st', layer=np.zeros((4, 4, 4), np.uint8),
                             profile_key=window.profile.key, selected_layout=window.locked_layout,
                             variant_id=window.route_combo.currentData(), reasons=[])
    generation = window.map_generation
    if change == 'failed':
        result.accepted = False
    elif change == 'closed':
        window.map_state.is_open = False
    elif change == 'generation':
        generation -= 1
    elif change == 'layout':
        result.selected_layout = 'layout_12'
    elif change == 'profile':
        result.profile_key = 'old/profile'
    elif change == 'variant':
        result.variant_id = 'old_version'
    else:
        result.floor = 'unknown'
    window._finish_map_feedback(generation, [0, 0, 40, 40], result)
    assert window.settings['reference_floor'] == '2nd'
    assert window.reference_floor_source == '手动'


def test_fast_g_result_syncs_once_and_late_preview_preserves_manual_switch(window, direct_inputs, monkeypatch):
    from tests.support.projection_results import result_for
    from src.projection_workflow import complete_projection_preview
    monkeypatch.setattr(app, 'floor_control_mouse_down', lambda *_: False)
    result = replace(result_for(window, direct_inputs), floor='1st')
    generation, session = window.map_generation, window.session_id
    now = time.perf_counter()
    meta = dict(posted_at=now, requested_at=now, session=session, ack=threading.Event())
    window._receive_work_event(('ready', ('work', 'project', True, generation, [0, 0, 40, 40], result, None), meta))
    assert meta['ack'].is_set() and window.settings['reference_floor'] == '1st'
    window._next_reference_floor()
    assert window.settings['reference_floor'] == 'b1'
    window._receive_work_event(('preview', generation, session, result, complete_projection_preview(result)))
    assert window.settings['reference_floor'] == 'b1' and window.reference_floor_source == '手动'


def test_manual_floor_selection_leaves_projection_and_generation_unchanged(window, monkeypatch):
    projection = object()
    window.live_projection = projection
    generation = window.map_generation
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Manual reference floor rematched projection'))
    window.floor_combo.setCurrentIndex(window.floor_combo.findData('1st'))
    assert window.live_projection is projection and window.map_generation == generation
    assert window.settings['floor'] == '2nd'


def test_reference_survives_settings_and_resets_with_new_round(window):
    ready(window)
    window.open_settings()
    assert not window.reference_overlay.visible and window.reference_visible
    window.settings_page.close()
    wait_for(lambda: window.reference_overlay.visible)
    window.reset_map_state()
    assert not window.reference_visible and not window.reference_overlay.visible
    assert window.reference_context is None


def test_raw_input_reference_keys_deduplicate_and_ignore_other_apps(window, source, monkeypatch):
    from tests.support.key_events import install, record
    window.map_state.is_open = False
    source.configure('G', ['IdentityV'], action_keys={'reference_toggle': 'H', 'reference_next_floor': 'J'})
    install(window, source)
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Reference requested projection'))
    generation = window.map_generation
    for down in (True, True, False):
        record(source, down, vk=ord('H'))
    record(source, True, vk=ord('J'))
    record(source, False, vk=ord('J'))
    window._handle_map_key()
    wait_for(lambda: window.reference_overlay.visible)
    assert window.settings['reference_floor'] == '1st' and window.map_generation == generation
    record(source, True, vk=ord('J'), title='Other')
    record(source, False, vk=ord('J'))
    window._handle_map_key()
    assert window.settings['reference_floor'] == '1st'


def test_hard_reference_cycles_only_two_floors(window):
    window.difficulty_var.set('困难')
    window._profile_changed()
    window.locked_layout = 'layout_05'
    window._update_assets()
    window._next_reference_floor()
    assert window.settings['reference_floor'] == '1st'
    window._next_reference_floor()
    assert window.settings['reference_floor'] == '2nd'
    assert window.floor_combo.findData('b1') == -1


def test_late_source_after_round_reset_or_version_change_is_discarded(window, monkeypatch):
    key = window._reference_image_key()
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: pytest.fail('Old reference restored'))
    window.settings['template_route_versions'][window.profile.key] = 'route_clean'
    window._receive_work_event(('floor_reference', key, np.ones((10, 10, 3), np.uint8), None))
    assert getattr(window, '_floor_reference_image', None) is None
    key = window._reference_image_key()
    window.reset_map_state()
    window._receive_work_event(('floor_reference', key, np.ones((10, 10, 3), np.uint8), None))
    assert not window.reference_visible and window.reference_context is None
