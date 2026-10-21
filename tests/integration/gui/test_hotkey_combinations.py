from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.app_catalog import load_settings
from tests.support.key_events import install, record


def test_settings_record_and_save_combinations_for_all_actions(window):
    window.open_settings()
    page = window.settings_page
    assignments = [
        ('recognize_hotkey', Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier, 'Ctrl+1'),
        ('project_hotkey', Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, 'Ctrl+Shift+1'),
        ('reference_hotkey', Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, '1'),
        ('reference_floor_hotkey', Qt.Key.Key_J, Qt.KeyboardModifier.AltModifier, 'Alt+J'),
        ('chest_gold_add_hotkey', Qt.Key.Key_F5, Qt.KeyboardModifier.ControlModifier, 'Ctrl+F5'),
    ]
    for field, key, modifiers, label in assignments:
        button = page.hotkey_buttons[field]
        button.click()
        QTest.keyClick(button, key, modifiers)
        assert button.value == label and page.vars[field].get() == label
        assert not button.capturing
    # Canonical duplicates are rejected, while a different modifier combination is legal.
    assert not page._capture_key('chest_purple_add_hotkey', 'ctrl+1')
    assert not page._capture_key('chest_purple_add_hotkey', 'Ctrl+G')
    assert page._persist_settings()
    page.close()
    saved = load_settings(window.catalog, window.settings_path)
    for field, _, _, label in assignments:
        assert saved[field] == label == window.settings[field]
    assert saved['chest_gold_subtract_hotkey'] == 'F6'


def test_capture_waits_for_primary_and_cancellation_keeps_old_value(window):
    window.open_settings()
    button = window.settings_page.hotkey_buttons['project_hotkey']
    button.click()
    QTest.keyPress(button, Qt.Key.Key_Control)
    QTest.keyPress(button, Qt.Key.Key_Shift, Qt.KeyboardModifier.ControlModifier)
    assert button.capturing and button.value == 'F10'
    QTest.keyClick(button, Qt.Key.Key_Escape)
    QTest.keyRelease(button, Qt.Key.Key_Shift)
    QTest.keyRelease(button, Qt.Key.Key_Control)
    assert not button.capturing and button.value == 'F10'
    assert window.settings_page.vars['project_hotkey'].get() == 'F10'


def test_shifted_digit_is_recorded_as_physical_number_key(window):
    window.open_settings()
    button = window.settings_page.hotkey_buttons['reference_hotkey']
    button.click()
    # Windows can report Ctrl+Shift+1 as '!'; Qt's native VK retains the number row key.
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Exclam,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, 0x02, 0x31, 0, '!')
    QApplication.sendEvent(button, event)
    assert button.value == 'Ctrl+Shift+1' and not button.capturing


def test_map_capture_accepts_single_key_and_rejects_chord(window):
    window.open_settings()
    button = window.settings_page.hotkey_buttons['map_key']
    button.click()
    # Send a combined event directly: a single modifier press is itself a legal map key.
    QApplication.sendEvent(button, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier))
    assert button.capturing and button.value == 'G'
    QTest.keyClick(button, Qt.Key.Key_T)
    assert not button.capturing and button.value == 'T'


def test_configured_combinations_reach_counter_reference_and_floor_actions(window, source):
    install(window, source)
    window.apply_settings({**window.settings, 'reference_hotkey': 'Ctrl+1',
        'reference_floor_hotkey': 'Ctrl+Shift+1', 'chest_gold_add_hotkey': '1'})
    window.map_state.is_open = False
    generation = window.map_generation
    first_floor = window.settings['reference_floor']
    record(source, True, ord('1'))
    record(source, False, ord('1'))
    window._handle_map_key()
    assert window.chest_counter.gold == 1 and not window.reference_visible
    for vk, down in [(0xA2, True), (ord('1'), True), (ord('1'), True), (0xA2, False), (ord('1'), False)]:
        record(source, down, vk)
    window._handle_map_key()
    assert window.reference_visible and window.chest_counter.gold == 1
    for vk, down in [(0xA3, True), (0xA1, True), (ord('1'), True), (ord('1'), False), (0xA1, False), (0xA3, False)]:
        record(source, down, vk)
    window._handle_map_key()
    assert window.settings['reference_floor'] != first_floor
    assert window.reference_visible and window.chest_counter.gold == 1
    assert not window.map_state.is_open and window.map_generation == generation
