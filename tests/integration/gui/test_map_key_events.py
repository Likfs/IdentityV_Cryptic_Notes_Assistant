from dataclasses import replace
import os
import threading
import time
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from src import route_projection_app as app
from src.capture.map_keys import GlobalMapKeys
from tests.support.key_events import record, install
from tests.support.projection_results import result_for


def test_complete_press_release_is_buffered_without_gui_polling(source):
    record(source, True)
    record(source, False)
    assert [(event.action, event.down, event.enabled) for event in list(source.events.queue)] == [
        ('map', True, True), ('map', False, True)]


def test_repeat_is_deduplicated_and_release_outside_game_is_kept(source):
    record(source, True)
    record(source, True)
    record(source, False, title='Other window')
    record(source, True)
    events = list(source.events.queue)
    assert [event.down for event in events] == [True, False, True]
    assert [event.sequence for event in events] == [1, 2, 3]
    assert [event.enabled for event in events] == [True, False, True]


def test_focus_and_suspension_are_recorded_at_capture(source):
    record(source, True, pid=os.getpid())
    assert not source.events.get_nowait().enabled
    source.configure('G', ['IdentityV'], suspended=True)
    record(source, True)
    assert not source.events.get_nowait().enabled


def test_reconfigure_discards_old_key_events_and_held_startup(monkeypatch, source):
    record(source, True)
    revision = source.revision
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _keys: {ord('J')}))
    source.configure('J', ['IdentityV'])
    assert source.revision > revision and source.events.empty()
    record(source, True, vk=ord('G'))
    record(source, True, vk=ord('J'))
    assert source.events.empty()
    record(source, False, vk=ord('J'))
    record(source, True, vk=ord('J'))
    assert [event.down for event in list(source.events.queue)] == [False, True]


def test_short_close_during_gui_stall_hides_projection(window, source):
    install(window, source)
    window.overlay.show(np.zeros((4, 4, 4), np.uint8))
    generation = window.map_generation
    record(source, True)
    record(source, False)
    # Neither the timer nor Qt signal has run during the entire key cycle.
    assert window.overlay.visible and window.map_state.is_open
    QApplication.instance().processEvents()
    assert not window.overlay.visible and not window.map_state.is_open
    assert window.map_generation == generation + 1


def test_focus_change_after_captured_close_cannot_discard_it(window, source, monkeypatch):
    install(window, source)
    record(source, True)
    record(source, False)
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: False)
    monkeypatch.setattr(app, 'own_window_is_foreground', lambda: True)
    window._handle_map_key()
    assert not window.map_state.is_open


def test_close_hides_before_expensive_log_rendering(window, source, monkeypatch):
    install(window, source)
    source.ready = True
    window.overlay.show(np.zeros((4, 4, 4), np.uint8))
    refreshes = []
    monkeypatch.setattr(window, '_refresh_logs', lambda: refreshes.append(
        (window.map_state.is_open, window.overlay.visible)))
    record(source, True)
    record(source, False)
    window._handle_map_key()
    assert refreshes == [(False, False)]


def test_ignored_press_does_not_stick_after_release(window, source):
    install(window, source)
    record(source, True, title='Other window')
    record(source, False, title='Other window')
    record(source, True)
    window._handle_map_key()
    assert not window.map_state.is_open
    assert window.map_generation == 1


def test_escape_only_closes_and_cancels_pending_open(window, source):
    install(window, source)
    record(source, True, vk=0x1B)
    record(source, False, vk=0x1B)
    record(source, True, vk=0x1B)
    window._handle_map_key()
    assert not window.map_state.is_open and window.map_generation == 1
    record(source, False, vk=0x1B)
    window.projection_delay_pending = True
    record(source, True, vk=0x1B)
    window._handle_map_key()
    assert not window.projection_delay_pending and window.map_generation == 2


def test_close_open_while_busy_starts_latest_request_after_old_result(window, source, monkeypatch):
    install(window, source, template=True)
    requests, scheduled = [], []
    monkeypatch.setattr(window, 'request_work', lambda mode, *, auto: requests.append((mode, auto)))
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append((delay, callback)))
    old_generation, session = window.map_generation, window.session_id
    window.busy = True
    window._pending_auto_projection = (old_generation, session, 'template', window.locked_layout)
    record(source, True, age=.2)
    record(source, False, age=.2)
    record(source, True, age=.2)
    record(source, False, age=.2)
    window._handle_map_key()
    assert window.map_state.is_open and window.map_generation == old_generation + 2
    assert window._pending_auto_projection is None
    assert len(scheduled) == 1 and scheduled[0][0] == 0
    scheduled[0][1]()
    assert not requests and window._pending_auto_projection[0] == window.map_generation
    meta = dict(posted_at=time.perf_counter(), requested_at=time.perf_counter(),
                session=session, ack=threading.Event())
    window._receive_work_event(('ready', ('work', 'project', True, old_generation, [0, 0, 4, 4], None, None), meta))
    assert requests == [('project', True)]
    assert window._pending_auto_projection is None and meta['ack'].is_set()


def test_queued_close_wins_before_ready_layer_delivery(window, source, direct_inputs, monkeypatch):
    install(window, source, template=True)
    ready = result_for(window, direct_inputs)
    generation = window.map_generation
    record(source, True)
    record(source, False)
    monkeypatch.setattr(window.overlay, 'show', lambda *a: pytest.fail('late layer restored'))
    meta = dict(posted_at=time.perf_counter(), requested_at=time.perf_counter(),
                session=window.session_id, ack=threading.Event())
    window.busy = True
    window._receive_work_event(('ready', ('work', 'project', True, generation, [0, 0, 40, 40], ready, None), meta))
    assert not window.map_state.is_open and not window.busy and meta['ack'].is_set()
    assert window.last_result is None


def test_queued_close_wins_before_preview_and_reference_delivery(window, source, direct_inputs, monkeypatch):
    install(window, source, template=True)
    ready = result_for(window, direct_inputs)
    generation, session = window.map_generation, window.session_id
    window.last_result = ready
    record(source, True)
    record(source, False)
    monkeypatch.setattr(window, '_set_preview', lambda *a: pytest.fail('late screenshot restored'))
    window._receive_work_event(('preview', generation, session, ready, replace(ready)))
    assert window.last_result is ready and not window.map_state.is_open
    key = window._reference_image_key()
    window._receive_work_event(('floor_reference', key, direct_inputs, None))
    assert not window.reference_visible and not window.reference_overlay.visible


def test_close_arriving_during_result_ui_updates_cannot_restore_overlay(window, source, direct_inputs, monkeypatch):
    install(window, source, template=True)
    result = result_for(window, direct_inputs, deferred=False)
    queued = []
    def update_preview(*args):
        if not queued:
            queued.append(True)
            record(source, True)
            record(source, False)
    monkeypatch.setattr(window, '_set_preview', update_preview)
    monkeypatch.setattr(window.overlay, 'show', lambda *a: pytest.fail('close during UI update was missed'))
    window._apply_result('project', True, window.map_generation, [0, 0, 40, 40], result, None)
    assert not window.map_state.is_open and not window.overlay.visible


def test_settings_and_reset_discard_prior_events(window, source, monkeypatch):
    install(window, source)
    window.apply_settings({**window.settings, 'map_key': 'J'})
    revision = source.revision
    record(source, True, vk=ord('J'))
    window.reset_map_state()
    assert source.revision > revision and source.events.empty()
    assert not window.map_state.is_open
    record(source, True, vk=ord('G'))
    window._handle_map_key()
    assert not window.map_state.is_open


def test_listener_failure_hides_once_and_is_visible_to_user(window, source):
    install(window, source)
    source.error = '地图按键事件监听失败：测试错误'
    window._handle_map_key()
    generation = window.map_generation
    assert not window.map_state.is_open and not window.overlay.visible
    assert source.error in window.status_var.get()
    window._handle_map_key()
    assert window.map_generation == generation


def test_shutdown_stops_input_source(window, source, monkeypatch):
    install(window, source)
    stopped = []
    monkeypatch.setattr(source, 'stop', lambda: stopped.append(True))
    window._shutdown()
    assert stopped == [True]
