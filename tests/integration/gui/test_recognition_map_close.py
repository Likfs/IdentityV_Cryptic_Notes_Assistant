from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src import route_projection_app as app
from src.capture.region import CaptureRegion
from src.route_projection_qt_ui import LayoutConfirmationDialog
from tests.support.layout_candidates import candidates
from tests.support.key_events import install, record
from tests.support.qt_wait import wait_for


class Replay:
    """Real worker/workflow and Raw Input queue, gated at the capture/retrieval seams."""

    def __init__(self, window, source, monkeypatch, tmp_path, *, block_capture=False):
        self.window, self.source = window, source
        self.capture_entered, self.release_capture = Event(), Event()
        self.matching, self.release_result, self.returned = Event(), Event(), Event()
        self.release_new = Event()
        self.calls = 0
        self.backend = window.settings['layout_recognition_backend']
        if not block_capture:
            self.release_capture.set()
        def capture(_roi):
            self.capture_entered.set()
            assert self.release_capture.wait(3)
            return np.full((80, 80, 3), 100, np.uint8)
        def retrieve(*_args, **_kwargs):
            gate = self.release_result if self.calls == 0 else self.release_new
            self.calls += 1
            self.matching.set()
            try:
                assert gate.wait(3)
                result = candidates()
                result.backend = self.backend
                return result
            finally:
                self.returned.set()
        monkeypatch.setattr(app, 'foreground_game_client', lambda _: CaptureRegion(0, 0, 2560, 1440))
        monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _: None)
        monkeypatch.setattr(app, 'recognition_window_identity', lambda _: ['same-game'])
        monkeypatch.setattr(app, 'capture_bgr', capture)
        monkeypatch.setattr(app, 'save_workflow_result', lambda _: tmp_path / 'diagnostic')
        monkeypatch.setattr(window, '_layout_retriever', lambda: SimpleNamespace(recognize=retrieve))
        install(window, source)
        window.clear_layout_lock()
        window.apply_settings(window.settings, save=False)
        window.map_state.reset(False)
        self.key('G')

    def key(self, name, *, drain=True, age=0):
        vk = {'G': ord('G'), 'Esc': 0x1B, 'F8': 0x77, 'F12': 0x7B}[name]
        for down in (True, False):
            record(self.source, down, vk, age=age)
        if drain:
            self.window._handle_map_key()

    def start(self):
        self.key('F8')

    def complete(self):
        self.release_result.set()
        wait_for(lambda: not self.window.results.empty())
        self.window._tick()

    def release(self):
        self.release_capture.set()
        self.release_result.set()
        self.release_new.set()


@pytest.mark.parametrize('backend', ['pure_structure', 'icon_structure'])
@pytest.mark.parametrize('close_key', ['G', 'Esc'])
@pytest.mark.parametrize('confirmation', ['reopen', 'proactive'])
def test_closed_map_recognition_finishes_and_confirms_only_on_user_action(
        window, source, monkeypatch, tmp_path, backend, close_key, confirmation):
    window.settings['layout_recognition_backend'] = backend
    replay = Replay(window, source, monkeypatch, tmp_path)
    scheduled, prompts = [], []
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append((delay, callback)) if delay != 25 else None)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: prompts.append(1) or 'layout_27')
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        scheduled.clear()
        replay.key(close_key)
        replay.complete()
        assert not window.map_state.is_open and not prompts and not scheduled
        assert window.last_result is not None and window._pending_confirmation is not None
        assert window.confirm_layout_button.isVisible()
        assert not window.busy and not window.overlay.visible and window.live_projection is None
        if confirmation == 'reopen':
            replay.key('G')
            assert len(scheduled) == 1 and scheduled[0][0] == 0
            scheduled[0][1]()
        else:
            window.confirm_layout_button.click()
        assert prompts == [1] and window.locked_layout == 'layout_27'
        assert window._pending_confirmation is None and not window.overlay.visible
        assert window.last_result.registration is None and window.live_projection is None
    finally:
        replay.release()


@pytest.mark.parametrize('queued_close', [False, True])
def test_close_before_snapshot_completion_is_rejected_even_with_late_ui_delivery(
        window, source, monkeypatch, tmp_path, queued_close):
    replay = Replay(window, source, monkeypatch, tmp_path, block_capture=True)
    try:
        replay.start()
        wait_for(replay.capture_entered.is_set)
        request = window._recognition_request
        replay.key('Esc', drain=not queued_close)
        replay.release_capture.set()
        replay.release_result.set()
        wait_for(lambda: request.capture_invalid)
        wait_for(lambda: not window.busy)
        window._tick()
        assert request.cancel.is_set() and window.locked_layout is None
        assert window.last_result is None and window._pending_confirmation is None
        assert not window.confirm_layout_button.isVisible() and not window.overlay.visible
    finally:
        replay.release()


def test_multiple_toggles_after_snapshot_keep_candidates_but_never_auto_project(window, source, monkeypatch, tmp_path):
    replay = Replay(window, source, monkeypatch, tmp_path)
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        for key in ('G', 'G', 'Esc', 'G', 'G'):
            replay.key(key)
        assert not window.map_state.is_open
        replay.complete()
        assert window._pending_confirmation is not None and window.locked_layout is None
        assert not window.overlay.visible and window.live_projection is None
    finally:
        replay.release()


def test_auto_selection_while_closed_only_locks_layout(window, source, monkeypatch, tmp_path):
    window.settings['auto_select_layout'] = True
    replay = Replay(window, source, monkeypatch, tmp_path)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Auto choice prompted'))
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.key('Esc')
        replay.complete()
        assert window.locked_layout == 'layout_22' and not window.map_state.is_open
        assert window._pending_confirmation is None and not window.overlay.visible
        assert window.live_projection is None and window.last_result.registration is None
    finally:
        replay.release()


@pytest.mark.parametrize('change', ['next_round', 'restart', 'profile', 'version', 'settings', 'layout'])
def test_close_does_not_weaken_context_invalidation_or_new_busy(window, source, monkeypatch, tmp_path, change):
    replay = Replay(window, source, monkeypatch, tmp_path)
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        request = window._recognition_request
        replay.key('G')
        if change == 'next_round':
            replay.key('F12')
        elif change == 'restart':
            replay.key('F8')
        elif change == 'profile':
            window.difficulty_var.set('困难')
            window._profile_changed()
        elif change == 'version':
            window.route_combo.setCurrentIndex(window.route_combo.findData('route_clean'))
        elif change == 'settings':
            window.open_settings()
        else:
            monkeypatch.setattr(app.messagebox, 'askyesno', lambda *a, **kw: True)
            window._manual_layout_changed('layout_01')
        assert request.cancel.is_set()
        active = window._active_work
        replay.release_result.set()
        wait_for(replay.returned.is_set)
        QApplication.processEvents()
        window._tick()
        assert window._pending_confirmation is None and not window.confirm_layout_button.isVisible()
        if change == 'restart':
            assert window._active_work is active and window.busy
        elif change != 'layout':
            assert window.locked_layout is None
        assert not window.overlay.visible
    finally:
        replay.release()


def test_next_round_discards_deferred_confirmation_and_scheduled_reopen(window, source, monkeypatch, tmp_path):
    replay = Replay(window, source, monkeypatch, tmp_path)
    callbacks = []
    monkeypatch.setattr(window, 'after', lambda delay, callback: callbacks.append(callback) if delay != 25 else None)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Stale dialog opened'))
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.key('Esc')
        replay.complete()
        replay.key('G')
        assert callbacks
        window.start_next_round()
        for callback in list(callbacks):
            callback()
        assert window._pending_confirmation is None and window.locked_layout is None
    finally:
        replay.release()


def test_closing_before_scheduled_dialog_defers_until_reopen(window, source, monkeypatch, tmp_path):
    replay = Replay(window, source, monkeypatch, tmp_path)
    callbacks, prompts = [], []
    monkeypatch.setattr(window, 'after', lambda delay, callback: callbacks.append(callback) if delay != 25 else None)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: prompts.append(1) or 'layout_27')
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.complete()
        assert len(callbacks) == 1
        replay.key('G')
        callbacks.pop()()
        assert not prompts and window._pending_confirmation is not None
        replay.key('G')
        callbacks.pop()()
        assert prompts == [1] and window.locked_layout == 'layout_27'
    finally:
        replay.release()


@pytest.mark.parametrize('transition', ['close', 'next_round'])
def test_real_modal_watcher_closes_without_locking_and_only_close_keeps_candidates(
        window, source, monkeypatch, tmp_path, transition):
    replay = Replay(window, source, monkeypatch, tmp_path)
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.key('Esc')
        replay.complete()
        pending = window._pending_confirmation
        replay.key('G')
        QTimer.singleShot(10, lambda: replay.key('Esc' if transition == 'close' else 'F12'))
        # Bound the real modal loop if the watcher regresses.
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(lambda: QApplication.activeModalWidget().reject()
                              if QApplication.activeModalWidget() else None)
        guard.start(1500)
        try:
            window._confirm_recognition(pending[0], pending[1], window._pending_confirmation_roi)
            assert guard.remainingTime() > 500
        finally:
            guard.stop()
        assert window.locked_layout is None and window._confirmation_in_progress is None
        assert not window.overlay.visible
        if transition == 'close':
            assert window._pending_confirmation is pending and window.confirm_layout_button.isVisible()
        else:
            assert window._pending_confirmation is None and not window.confirm_layout_button.isVisible()
    finally:
        replay.release()


def test_changed_game_window_discards_deferred_candidates(window, source, monkeypatch, tmp_path):
    replay = Replay(window, source, monkeypatch, tmp_path)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Changed game prompted'))
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.key('G')
        replay.complete()
        monkeypatch.setattr(app, 'recognition_window_identity', lambda _: ['other-game'])
        window.confirm_layout_button.click()
        assert window._pending_confirmation is None and window._recognition_request is None
        assert window.locked_layout is None and not window.confirm_layout_button.isVisible()
    finally:
        replay.release()


def test_proactive_modal_cannot_confirm_after_same_session_settings_invalidation(
        window, source, monkeypatch, tmp_path):
    replay = Replay(window, source, monkeypatch, tmp_path)
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        replay.key('G')
        replay.complete()
        session, generation = window.session_id, window.map_generation
        def confirm(*_args, **kwargs):
            window.apply_settings({**window.settings, 'game_window_titles': ['Different game']}, save=False)
            assert window.session_id == session and window.map_generation == generation
            assert not kwargs['is_current']()
            return 'layout_27'  # Even a dialog acceptance already queued must be ignored.
        monkeypatch.setattr(app.messagebox, 'confirm_layout', confirm)
        window._confirm_pending_layout()
        assert window.locked_layout is None and window._recognition_request is None
        assert window._pending_confirmation is None and window._confirmation_in_progress is None
    finally:
        replay.release()


@pytest.mark.parametrize('automatic', [False, True])
def test_recognition_never_drains_queued_projection_for_previously_locked_layout(
        window, source, monkeypatch, tmp_path, automatic):
    window.settings['auto_select_layout'] = automatic
    replay = Replay(window, source, monkeypatch, tmp_path)
    try:
        replay.start()
        wait_for(replay.matching.is_set)
        # Same winning layout as the previous lock must still need a fresh user map action.
        window.locked_layout = 'layout_22'
        window._auto_if_open(window.map_generation)
        assert window._pending_auto_projection is not None
        monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Old pending projection started'))
        replay.complete()
        assert window._pending_auto_projection is None and window.live_projection is None
        if not automatic:
            app.RouteProjectionApp.request_work(window, 'project', auto=False)
            assert '请先确认' in window.status_var.get()
    finally:
        replay.release()
