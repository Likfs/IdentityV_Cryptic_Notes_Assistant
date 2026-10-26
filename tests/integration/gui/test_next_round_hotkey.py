from threading import Event

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from src import route_projection_app as app
from src.app_catalog import load_settings
from tests.support.key_events import install, record


def test_raw_next_round_key_uses_reset_cancels_work_and_discards_queued_actions(window, source, monkeypatch):
    install(window, source)
    window.apply_settings(window.settings, save=False)
    window._change_chest_count('chest_gold_add')
    window._set_preview(np.zeros((10, 10, 3), np.uint8), 'old capture')
    old_session, old_generation = window.session_id, window.map_generation
    token = Event()
    window._active_work = dict(id='old', mode='recognize', cancel=token)
    window.busy = True
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Pre-reset action replayed'))
    record(source, True, 0x7B)
    record(source, True, 0x7B)  # OS repeat is de-duplicated before delivery.
    record(source, False, 0x7B)
    record(source, True, 0x74)  # Old queued counter action must also be discarded.
    record(source, False, 0x74)
    window._handle_map_key()
    assert token.is_set() and not window.busy
    assert window.locked_layout is None and window.capture_panel._pixmap is None
    assert window.session_id != old_session and window.map_generation > old_generation
    assert window.chest_counter.gold == 0 and source.events.empty()
    assert not window.map_state.is_open
    newer = dict(id='new', mode='recognize', cancel=Event())
    window._active_work, window.busy = newer, True
    window._receive_work_event(('preview', old_generation, old_session, object(), object()))
    window._apply_result('recognize', False, old_generation, [0, 0, 10, 10], None, 'old failure')
    assert window._active_work is newer and window.busy
    assert window.locked_layout is None and window.capture_panel._pixmap is None
    window._active_work, window.busy = None, False
    window.chest_undo_button.click()
    assert window.chest_counter.gold == 1 and window.locked_layout is None


@pytest.mark.parametrize('blocked', ['other_window', 'settings', 'changed_foreground'])
def test_next_round_hotkey_respects_game_foreground_and_settings(window, source, monkeypatch, blocked):
    install(window, source)
    window.apply_settings(window.settings, save=False)
    window._change_chest_count('chest_gold_add')
    if blocked == 'settings':
        window.open_settings()
    generation, session = window.map_generation, window.session_id
    if blocked == 'changed_foreground':
        monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: False)
    for down in (True, False):
        record(source, down, 0x7B, title='Other' if blocked == 'other_window' else 'IdentityV')
    window._handle_map_key()
    assert window.session_id == session and window.map_generation == generation
    assert window.chest_counter.gold == 1
    if blocked == 'settings':
        window.settings_page.close()


def test_next_round_recording_conflicts_persistence_and_custom_raw_key(window, source):
    install(window, source)
    window.open_settings()
    page = window.settings_page
    page.tabs.setCurrentIndex(1)
    assert not page._capture_key('next_round_hotkey', 'F8')
    assert not page._capture_key('next_round_hotkey', 'G')
    assert not page._capture_key('recognize_hotkey', 'F12')
    button = page.hotkey_buttons['next_round_hotkey']
    button.click()
    QApplication.sendEvent(button, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Enter,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.KeypadModifier, 0, 0x0D, 0))
    assert button.value == 'Ctrl+NumEnter' and not button.capturing
    assert page._persist_settings()
    page.close()
    assert load_settings(window.catalog, window.settings_path)['next_round_hotkey'] == 'Ctrl+NumEnter'
    window._change_chest_count('chest_purple_add')
    session = window.session_id
    for vk, down in ((0x7B, True), (0x7B, False), (0x0D, True), (0x0D, False)):
        record(source, down, vk)
    window._handle_map_key()
    assert window.session_id == session  # Old F12 and main Enter no longer reset.
    for vk, down in ((0xA2, True), (0x10D, True), (0x10D, False), (0xA2, False)):
        record(source, down, vk)
    window._handle_map_key()
    assert window.session_id != session and window.chest_counter.purple == 0


@pytest.mark.parametrize('size', [(1280, 800), (1080, 680)])
def test_hotkey_settings_are_two_columns_with_all_ten_fields_visible(window, tmp_path, size):
    window.resize(*size)
    window.open_settings()
    page = window.settings_page
    page.tabs.setCurrentIndex(1)
    QApplication.processEvents()
    left = page.hotkey_buttons['map_key']
    right = page.hotkey_buttons['next_round_hotkey']
    assert right.mapTo(page, right.rect().topLeft()).x() > left.mapTo(page, left.rect().topLeft()).x() + left.width()
    assert abs(right.mapTo(page, right.rect().topLeft()).y() - left.mapTo(page, left.rect().topLeft()).y()) <= 2
    assert len(page.hotkey_buttons) == 10
    for button in page.hotkey_buttons.values():
        point = button.mapTo(page, button.rect().topLeft())
        assert button.isVisible() and page.rect().contains(point + button.rect().bottomRight())
        assert button.width() >= 100
    assert window.grab().save(str(tmp_path / 'hotkey-columns.png'))
    page.close()
