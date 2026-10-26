import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication
from src.app_catalog import load_settings
from tests.support.key_events import install, record


@pytest.mark.parametrize('key,native,label', [
    (Qt.Key.Key_1, 0x61, 'Num1'), (Qt.Key.Key_0, 0x60, 'Num0'),
    (Qt.Key.Key_Plus, 0x6B, 'NumAdd'), (Qt.Key.Key_Minus, 0x6D, 'NumSubtract'),
    (Qt.Key.Key_Asterisk, 0x6A, 'NumMultiply'), (Qt.Key.Key_Slash, 0x6F, 'NumDivide'),
    (Qt.Key.Key_Period, 0x6E, 'NumDecimal'), (Qt.Key.Key_Enter, 0x0D, 'NumEnter'),
    (Qt.Key.Key_End, 0x23, 'Num1'), (Qt.Key.Key_Delete, 0x2E, 'NumDecimal'),
])
def test_numpad_recording_and_settings_roundtrip(window, key, native, label):
    window.open_settings()
    page = window.settings_page
    button = page.hotkey_buttons['recognize_hotkey']
    button.click()
    QApplication.sendEvent(button, QKeyEvent(QEvent.Type.KeyPress, key,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.KeypadModifier, 0, native, 0))
    assert not button.capturing and button.value == 'Ctrl+' + label
    assert page._persist_settings()
    page.close()
    assert load_settings(window.catalog, window.settings_path)['recognize_hotkey'] == 'Ctrl+' + label


def test_numpad_actions_use_same_raw_queue_and_never_toggle_map(window, source, monkeypatch):
    install(window, source)
    calls = []
    monkeypatch.setattr(window, 'request_work', lambda mode, **kw: calls.append(mode))
    window.apply_settings({**window.settings, 'recognize_hotkey': 'Num1', 'project_hotkey': 'Ctrl+NumEnter',
                           'chest_gold_add_hotkey': 'NumAdd'})
    window.map_state.is_open = False
    record(source, True, 0x61)
    record(source, True, 0x61)
    record(source, False, 0x61)
    record(source, True, 0x6B)
    record(source, False, 0x6B)
    record(source, True, 0xA2)
    record(source, True, 0x10D)
    record(source, False, 0xA2)
    record(source, False, 0x10D)
    window._handle_map_key()
    assert calls == ['recognize', 'project']
    assert window.chest_counter.gold == 1 and not window.map_state.is_open
    assert window.hotkeys is None  # No second RegisterHotKey path can fire the same action.


def test_main_keyboard_and_numpad_can_be_bound_independently(window):
    window.open_settings()
    page = window.settings_page
    assert page._capture_key('recognize_hotkey', 'Num1')
    assert page._capture_key('project_hotkey', '1')
    assert page._capture_key('reference_hotkey', 'NumEnter')
    assert page._capture_key('reference_floor_hotkey', 'Enter')
    assert page._persist_settings()
    page.close()


def test_numpad_map_key_opens_on_release_and_escape_still_only_closes(window, source):
    install(window, source)
    window.apply_settings({**window.settings, 'map_key': 'NumEnter'})
    window.map_state.is_open = False
    record(source, True, 0x0D)  # Main keyboard Enter must not toggle the map.
    record(source, False, 0x0D)
    window._handle_map_key()
    assert not window.map_state.is_open
    record(source, True, 0x10D)
    window._handle_map_key()
    assert not window.map_state.is_open
    record(source, False, 0x10D)
    window._handle_map_key()
    assert window.map_state.is_open and window.projection_delay_pending
    record(source, True, 0x1B)
    record(source, False, 0x1B)
    window._handle_map_key()
    assert not window.map_state.is_open and not window.projection_delay_pending


@pytest.mark.parametrize('condition', ['other_window', 'settings'])
def test_numpad_recognition_and_projection_respect_frontmost_game_and_settings(window, source, monkeypatch, condition):
    install(window, source)
    window.apply_settings({**window.settings, 'recognize_hotkey': 'Num1', 'project_hotkey': 'Num2'})
    calls = []
    monkeypatch.setattr(window, 'request_work', lambda mode, **kw: calls.append(mode))
    if condition == 'settings': window.open_settings()
    for vk in (0x61, 0x62):
        record(source, True, vk, title='Other' if condition == 'other_window' else 'IdentityV')
        record(source, False, vk)
    window._handle_map_key()
    assert not calls
    if condition == 'settings': window.settings_page.close()
