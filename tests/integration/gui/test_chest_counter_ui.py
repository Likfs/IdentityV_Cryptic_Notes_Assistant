from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from src import route_projection_app as app
from src.capture.region import CaptureRegion
from src.chest_counter import CHEST_ACTIONS
from tests.support.key_events import install, record


def test_counter_buttons_and_short_key_taps_survive_map_close(window, source, monkeypatch):
    monkeypatch.setattr(app, 'foreground_game_client', lambda _: CaptureRegion(0, 0, 2560, 1440))
    source.configure('G', ['IdentityV'], action_keys={
        action: window.settings[f'{action}_hotkey'] for action in CHEST_ACTIONS})
    install(window, source)
    for down in (True, True, False, True, False):
        record(source, down, vk=0x74)
    window._handle_map_key()
    assert window.chest_counter.gold == 2 and window.chest_counter.purple == 0
    window.chest_action_buttons['chest_purple_add'].click()
    assert window.chest_counter.purple == 1
    record(source, True)  # Close the game map; counter remains visible.
    record(source, False)
    window._handle_map_key()
    window._refresh_chest_overlay()
    assert not window.map_state.is_open and window.chest_overlay.visible
    assert '金箱 2' in window.chest_overlay.label.text()
    window.reset_map_state()
    assert (window.chest_counter.gold, window.chest_counter.purple) == (2, 1)
    window.chest_reset_button.click()
    assert (window.chest_counter.gold, window.chest_counter.purple) == (0, 0)
    window.chest_undo_button.click()
    assert (window.chest_counter.gold, window.chest_counter.purple) == (2, 1)
    assert window.chest_overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert window.chest_overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: False)
    window._refresh_chest_overlay()
    assert not window.chest_overlay.visible


def test_foreground_and_settings_filter_counter_keys_without_toggling_map(window, source):
    source.configure('G', ['IdentityV'], action_keys={'chest_gold_add': 'F5'})
    install(window, source)
    generation = window.map_generation
    record(source, True, vk=0x74, title='Other')
    record(source, False, vk=0x74)
    window._handle_map_key()
    assert window.chest_counter.gold == 0 and window.map_generation == generation
    window.open_settings()
    record(source, True, vk=0x74)
    record(source, False, vk=0x74)
    window._handle_map_key()
    assert window.chest_counter.gold == 0
    window.settings_page.close()
    record(source, True, vk=0x74)
    window._handle_map_key()
    assert window.chest_counter.gold == 1


def test_counter_keys_conflict_with_all_existing_keys_and_save(window):
    window.open_settings()
    page = window.settings_page
    button = page.hotkey_buttons['chest_gold_add_hotkey']
    button.click()
    QTest.keyClick(button, Qt.Key.Key_F8)
    assert page.vars['chest_gold_add_hotkey'].get() == 'F5'
    QTest.keyClick(button, Qt.Key.Key_U)
    assert page.vars['chest_gold_add_hotkey'].get() == 'U'
    assert not page._capture_key('project_hotkey', 'U')
    assert page._persist_settings()
    page.close()
    assert window.settings['chest_gold_add_hotkey'] == 'U'
