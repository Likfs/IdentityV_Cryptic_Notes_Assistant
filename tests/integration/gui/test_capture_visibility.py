from threading import Event
from types import SimpleNamespace

import pytest
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog
from src import route_projection_app as app
from src.capture import visibility
from src.capture.region import CaptureRegion
from tests.support.qt_wait import wait_for


def test_windows_affinity_values(monkeypatch):
    calls = []
    class Setter:
        def __call__(self, hwnd, flag):
            calls.append((hwnd, flag))
            return 1
    setter = Setter()
    monkeypatch.setattr(visibility, 'ctypes', SimpleNamespace(
        windll=SimpleNamespace(user32=SimpleNamespace(SetWindowDisplayAffinity=setter))))
    assert visibility.set_capture_excluded(123, True)
    assert visibility.set_capture_excluded(123, False)
    assert calls == [(123, 0x11), (123, 0)]


def test_setting_applies_to_existing_new_and_reshown_windows(window, monkeypatch):
    calls = []
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda hwnd, excluded: calls.append((hwnd, excluded)) or True)
    window.open_settings()
    page = window.settings_page
    page.capture_exclusion_check.setChecked(False)
    assert page._persist_settings()
    assert not window.capture_policy.excluded and not window.capture_excluded
    assert not window.settings['capture_exclusion_enabled']
    assert '允许' in page.capture_status_label.text()
    dialog = QDialog(window)
    try:
        visibility.apply_window_capture_policy(dialog)
        dialog.show()
        assert (int(dialog.winId()), False) in calls
        window.overlay.show(np.zeros((8, 8, 4), np.uint8))
        assert not window.overlay.capture_excluded
        window.overlay.hide()
        window.overlay.show(np.zeros((8, 8, 4), np.uint8))
        assert not window.overlay.capture_excluded
        page.capture_exclusion_check.setChecked(True)
        assert page._persist_settings()
        assert window.capture_excluded and dialog.capture_excluded
    finally:
        dialog.close()
        page.close()


def test_capture_guards_overlap_and_settings_changes_do_not_restore_old_policy(window, monkeypatch):
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *_: True)
    policy = window.capture_policy
    policy.set_excluded(False)
    policy.begin_capture('first')
    policy.begin_capture('second')
    assert policy.effective_excluded and window.capture_excluded
    policy.end_capture('first')
    assert policy.effective_excluded
    policy.set_excluded(True)
    policy.end_capture('second')
    assert policy.effective_excluded and window.capture_excluded
    policy.set_excluded(False)
    assert not window.capture_excluded


def test_internal_screenshot_is_clean_and_restores_before_matching_finishes(window, monkeypatch):
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *_: True)
    window.apply_settings({**window.settings, 'capture_exclusion_enabled': False})
    monkeypatch.setattr(app, 'foreground_game_client', lambda _: CaptureRegion(0, 0, 2560, 1440))
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _: None)
    monkeypatch.setattr(app, 'recognition_window_identity', lambda _: ['game'])
    monkeypatch.setattr(window, '_layout_retriever', lambda: SimpleNamespace())
    flushed = []
    monkeypatch.setattr(app, 'flush_desktop', lambda: flushed.append(True))
    captured = []
    def capture(_):
        captured.append(window.capture_policy.effective_excluded)
        return np.zeros((20, 20, 3), np.uint8)
    monkeypatch.setattr(app, 'capture_bgr', capture)
    entered, release = Event(), Event()
    def process(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('test matching failure')
    monkeypatch.setattr(app, 'process_screenshot', process)
    try:
        window.request_work('recognize', auto=False)
        wait_for(entered.is_set)
        wait_for(lambda: not window.capture_policy.effective_excluded)
        assert captured == [True] and flushed == [True]
        assert not window.capture_excluded
    finally:
        release.set()
        wait_for(lambda: not window.results.empty())
        window._tick()


def test_failed_affinity_rejects_dirty_capture_and_reports_failure(window, monkeypatch):
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *_: False)
    policy = window.capture_policy
    policy.set_excluded(False)
    with pytest.raises(RuntimeError, match='内部截图'):
        policy.begin_capture('failed')
    assert not policy.effective_excluded
    assert '失败' in window.capture_status_label.text()


@pytest.mark.parametrize('failure', ['window', 'screenshot', 'cancelled'])
def test_capture_error_or_cancellation_restores_user_policy(window, monkeypatch, failure):
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *_: True)
    window.apply_settings({**window.settings, 'capture_exclusion_enabled': False})
    monkeypatch.setattr(window, '_layout_retriever', lambda: SimpleNamespace())
    monkeypatch.setattr(app, 'foreground_game_client', lambda _: CaptureRegion(0, 0, 2560, 1440))
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _: None)
    monkeypatch.setattr(app, 'recognition_window_identity', lambda _: ['game'])
    monkeypatch.setattr(app, 'flush_desktop', lambda: None)
    def fail(*_):
        raise RuntimeError('test capture failure')
    if failure == 'window':
        monkeypatch.setattr(app, 'foreground_game_client', fail)
    else:
        monkeypatch.setattr(app, 'capture_bgr', fail)
    queued = []
    if failure == 'cancelled':
        class DeferredThread:
            def __init__(self, *, target, **kwargs):
                self.target = target
            def start(self):
                queued.append(self.target)
        monkeypatch.setattr(app.threading, 'Thread', DeferredThread)
    window.request_work('recognize', auto=False)
    if failure == 'cancelled':
        window._active_work['cancel'].set()
        queued[0]()
    wait_for(lambda: not window.capture_policy.effective_excluded)
    assert not window.capture_excluded and not window.settings['capture_exclusion_enabled']


def test_destroying_native_window_does_not_recreate_handle(window, monkeypatch):
    calls = []
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *args: calls.append(args) or True)
    dialog = QDialog(window)
    visibility.apply_window_capture_policy(dialog)
    dialog.show()
    assert dialog.effectiveWinId()
    calls.clear()
    dialog.destroy()
    assert not dialog.effectiveWinId()
    assert calls == []


def test_confirmation_stops_context_watcher_before_deferred_deletion(qt_app, monkeypatch):
    dialogs = []
    monkeypatch.setattr(app.LayoutConfirmationDialog, 'deleteLater', lambda dialog: dialogs.append(dialog))
    try:
        assert app.messagebox.confirm_layout([dict(layout='layout_12', score=.9)], None,
                                            is_current=lambda: False) is None
        assert dialogs and not any(timer.isActive() for timer in dialogs[0].findChildren(QTimer))
    finally:
        for dialog in dialogs:
            for timer in dialog.findChildren(QTimer):
                timer.stop()
            dialog.close()
