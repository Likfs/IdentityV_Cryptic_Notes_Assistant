from copy import deepcopy
from threading import Event
import time

import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src import projection_workflow as workflow
from src import route_projection_app as app
from src.route_projection_core import AUTO
from src.route_projection_qt_ui import LayoutConfirmationDialog
from tests.support.layout_candidates import candidates
from tests.support.projection_results import result_for


@pytest.mark.parametrize('counts', [(0, 0), (3, 2)])
def test_next_round_clears_session_even_with_zero_counts_and_undo_only_restores_counts(window, counts):
    for _ in range(counts[0]): window._change_chest_count('chest_gold_add')
    for _ in range(counts[1]): window._change_chest_count('chest_purple_add')
    old_session, old_generation = window.session_id, window.map_generation
    settings = deepcopy(window.settings)
    index = object()
    window.layout_retrievers = {'cached': index}
    window.last_result = object()
    window.last_decision = {'score': .9}
    window._set_preview(np.zeros((4, 4, 3), np.uint8), 'old')
    window.reference_visible = True
    window._pending_confirmation = (old_generation, object(), old_session, window.profile.key)
    window._pending_auto_projection = (old_generation, old_session, 'template', window.locked_layout)
    window._pending_template_version_projection = ('old',)
    window.projection_delay_pending = True
    window.projection_triggered_at = time.perf_counter()
    token = Event()
    window._active_work = dict(id='old', mode='recognize', cancel=token)
    window.busy = True
    window.hotkeys.events.put('recognize')
    window.chest_reset_button.click()
    assert window.chest_reset_button.text() == '下一局'
    assert (window.chest_counter.gold, window.chest_counter.purple) == (0, 0)
    assert token.is_set() and not window.busy and window._active_work is None
    assert window.session_id != old_session and window.map_generation > old_generation
    assert window.locked_layout is None and window.layout_var.get() == AUTO
    assert window.last_result is None and window.last_decision is None
    assert window.capture_panel._pixmap is None
    assert window._pending_confirmation is None and window._pending_auto_projection is None
    assert window._pending_template_version_projection is None and not window.projection_delay_pending
    assert window.projection_triggered_at is None and window.projection_wait_elapsed_ms is None
    assert not window.map_state.is_open and window.live_projection is None
    assert not window.overlay.visible and not window.reference_visible and not window.reference_overlay.visible
    assert window.reference_context is None and window.hotkeys.events.empty()
    assert all(name.text() == score.text() == '—' for name, score, _ in window.rank_widgets)
    assert window.settings == settings and window.layout_retrievers['cached'] is index
    if counts != (0, 0):
        window.chest_undo_button.click()
        assert (window.chest_counter.gold, window.chest_counter.purple) == counts
        assert window.locked_layout is None and window.last_result is None
    # The sentinel represents an already prepared shared index, not a real retriever.
    window.layout_retrievers = {}


@pytest.mark.parametrize('automatic', [False, True])
def test_queued_old_results_cannot_change_new_round_or_clear_new_busy(window, direct_inputs, monkeypatch, automatic):
    old = result_for(window, direct_inputs)
    old.session_id = window.session_id
    generation, session = window.map_generation, window.session_id
    reference_key = window._reference_image_key()
    window.chest_reset_button.click()
    window.settings['auto_select_layout'] = automatic
    window.busy = True
    active = dict(id='new', mode='recognize', cancel=Event())
    window._active_work = active
    status = window.status_var.get()
    monkeypatch.setattr(window.overlay, 'show', lambda *a: pytest.fail('Old layer restored'))
    monkeypatch.setattr(window, '_set_preview', lambda *a: pytest.fail('Old screenshot restored'))
    for mode, event_name in [('recognize', 'recognition_work'), ('project', 'work_result')]:
        for result, error in [(old, None), (None, 'old error')]:
            window.results.put((event_name, 'old', ('work', mode, False, generation, [0, 0, 40, 40], result, error)))
    window.results.put(('saved', old, 'old_directory', None))
    meta = dict(work_id='old', session=session, posted_at=time.perf_counter(),
                requested_at=time.perf_counter(), ack=Event())
    window._receive_work_event(('ready', ('work', 'project', True, generation, [0, 0, 40, 40], old, None), meta))
    window._receive_work_event(('preview', generation, session, old, old))
    window._receive_work_event(('preview_error', generation, session, old, 'old preview error'))
    window._receive_work_event(('work_progress', 'old', session, generation, 'old progress'))
    window._receive_work_event(('floor_reference', reference_key, np.zeros((4, 4, 3), np.uint8), None))
    # Also reject the legacy unwrapped completion path, including errors with no result.
    window._apply_result('project', False, generation, [0, 0, 40, 40], None, 'old error')
    window._tick()
    assert meta['ack'].is_set() and window.busy and window._active_work is active
    assert window.locked_layout is None and window.last_result is None and window.live_projection is None
    assert window._pending_confirmation is None and window.capture_panel._pixmap is None
    assert window._floor_reference_image is None and window.status_var.get() == status


def test_next_round_discards_scheduled_capture_and_projection(window, monkeypatch):
    scheduled = []
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append(callback))
    window._schedule_button_action('recognize')
    generation, session = window.map_generation, window.session_id
    scheduled.append(lambda: window._project_after_delay(generation, session))
    window.chest_reset_button.click()
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Old timer started a capture'))
    for callback in scheduled: callback()
    assert not window.busy and window.locked_layout is None


def test_next_round_closes_open_confirmation_and_cannot_commit_old_choice(window, monkeypatch):
    from types import SimpleNamespace
    window.settings['layout_recognition_backend'] = 'icon_structure'
    window.clear_layout_lock()
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=window.catalog,
        profile=window.profile, recognize_only=True, opacity=.5,
        layout_retriever=SimpleNamespace(recognize=lambda *a: candidates()))
    result.session_id = window.session_id
    generation = window.map_generation
    window._apply_result('recognize', False, generation, [0, 0, 80, 80], result, None)
    seen = []
    def next_round_while_modal():
        seen.extend(widget for widget in QApplication.topLevelWidgets()
                    if isinstance(widget, LayoutConfirmationDialog) and widget.isVisible())
        window.start_next_round()
    QTimer.singleShot(0, next_round_while_modal)
    window._confirm_recognition(generation, result)
    assert seen and all(not widget.isVisible() for widget in seen)
    assert window.locked_layout is None and window.last_result is None
    assert window._pending_confirmation is None and '下一局' in window.status_var.get()


def test_auto_select_setting_is_default_off_and_persists_from_ui(window):
    from src.app_catalog import load_settings
    window.open_settings()
    page = window.settings_page
    assert not page.auto_select_layout_check.isChecked()
    page.auto_select_layout_check.setChecked(True)
    assert page._persist_settings()
    page.close()
    assert window.settings['auto_select_layout'] is True
    assert load_settings(window.catalog, window.settings_path)['auto_select_layout'] is True
    window.open_settings()
    assert window.settings_page.auto_select_layout_check.isChecked()
    window.settings_page.auto_select_layout_check.setChecked(False)
    window.settings_page.close()
    assert load_settings(window.catalog, window.settings_path)['auto_select_layout'] is False
