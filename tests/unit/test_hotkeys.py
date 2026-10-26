from types import SimpleNamespace
import ctypes
from ctypes import wintypes

import pytest

from src.hotkeys import (normalize_hotkey, parse_hotkey, validate_action_hotkeys,
                         physical_modifier_vk, physical_key_vk, KEYPAD_SCAN_VKS, NUM_ENTER)
from src.capture.map_keys import GlobalMapKeys
from src.capture.runtime import GlobalActionHotkeys
from tests.support.key_events import record


@pytest.mark.parametrize('value,label,vk,flags', [
    (' shift + CTRL + 1 ', 'Ctrl+Shift+1', ord('1'), 6),
    ('alt+ctrl+shift+f12', 'Ctrl+Shift+Alt+F12', 0x7B, 7),
    ('ctrl+space', 'Ctrl+Space', 0x20, 2),
    ('h', 'H', ord('H'), 0), ('F8', 'F8', 0x77, 0),
])
def test_shortcuts_normalize_without_losing_modifiers(value, label, vk, flags):
    shortcut = parse_hotkey(value)
    assert (shortcut.label, shortcut.vk, shortcut.modifiers) == (label, vk, flags)
    assert normalize_hotkey(value) == label


@pytest.mark.parametrize('value', [None, [], '', 'Ctrl', 'Ctrl+', 'Ctrl+Ctrl+1',
    'Win+H', 'Ctrl+X1', 'Ctrl+F13', '1+Ctrl', 'A+B', 'Ctrl++1'])
def test_unsupported_shortcuts_are_rejected(value):
    with pytest.raises(ValueError):
        parse_hotkey(value)


def test_same_primary_can_have_distinct_combinations_but_map_conflicts_are_rejected():
    assert validate_action_hotkeys('G', {'one': '1', 'two': 'ctrl+1', 'three': 'shift+ctrl+1'}) == {
        'one': '1', 'two': 'Ctrl+1', 'three': 'Ctrl+Shift+1'}
    for map_key, shortcuts in [('G', {'a': 'Ctrl+G'}), ('Ctrl', {'a': 'Ctrl+1'}),
                                ('G', {'a': 'Ctrl+1', 'b': 'ctrl+1'})]:
        with pytest.raises(ValueError):
            validate_action_hotkeys(map_key, shortcuts)


def test_native_registration_passes_modifiers_and_norepeat_to_windows(monkeypatch):
    registrations, removed = [], []
    messages = iter([GlobalActionHotkeys.IDS['project'], GlobalActionHotkeys.IDS['recognize']])

    def get_message(pointer, *_args):
        identity = next(messages, None)
        if identity is None:
            return 0
        message = ctypes.cast(pointer, ctypes.POINTER(wintypes.MSG)).contents
        message.message, message.wParam = 0x0312, identity
        return 1

    user32 = SimpleNamespace(RegisterHotKey=lambda *args: registrations.append(args) or True,
        UnregisterHotKey=lambda *args: removed.append(args), GetMessageW=get_message)
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=user32,
        kernel32=SimpleNamespace(GetCurrentThreadId=lambda: 42)), raising=False)
    source = GlobalActionHotkeys('Ctrl+1', 'Ctrl+Shift+1')
    source._run()
    assert source.error == '' and not source.ready and source.thread_id is None
    assert registrations == [(None, 0x6401, 0x4002, ord('1')), (None, 0x6402, 0x4006, ord('1'))]
    assert removed == [(None, 0x6401), (None, 0x6402)]
    assert list(source.events.queue) == ['project', 'recognize']


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _: set()))
    result = GlobalMapKeys('G', ['IdentityV'], lambda: None)
    result.configure('G', ['IdentityV'], action_keys={
        'bare': '1', 'ctrl': 'Ctrl+1', 'ctrl_shift': 'Ctrl+Shift+1', 'alt': 'Alt+1'})
    return result


@pytest.mark.parametrize('modifiers,action', [([], 'bare'), ([0xA2], 'ctrl'),
    ([0xA3], 'ctrl'), ([0xA0, 0xA2], 'ctrl_shift'), ([0xA3, 0xA1], 'ctrl_shift'),
    ([0xA4], 'alt'), ([0xA0], None), ([0xA2, 0xA4], None), ([0x5B], None)])
def test_raw_input_matches_exact_modifiers_and_keeps_action_until_primary_release(source, modifiers, action):
    for vk in modifiers:
        record(source, True, vk)
    record(source, True, ord('1'))
    record(source, True, ord('1'))  # Windows repeat never adds a second press.
    for vk in modifiers:
        record(source, False, vk)  # Releasing modifiers first must not turn it into a bare action.
    record(source, False, ord('1'))
    record(source, False, ord('1'))
    assert [(event.action, event.down) for event in source.events.queue] == (
        [(action, True), (action, False)] if action else [])


def test_modifier_pressed_after_primary_does_not_synthesize_combination(source):
    record(source, True, ord('1'))
    record(source, True, 0xA2)
    record(source, False, ord('1'))
    assert [(event.action, event.down) for event in source.events.queue] == [('bare', True), ('bare', False)]


def test_reconfigure_requires_held_chord_release_and_discards_old_queue(source, monkeypatch):
    record(source, True, ord('1'))
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _: {0xA2, ord('1')}))
    source.configure('G', ['IdentityV'], action_keys={'new': 'Ctrl+1'})
    record(source, True, ord('1'))
    assert source.events.empty()
    record(source, False, ord('1'))
    record(source, True, ord('1'))
    assert [(event.action, event.down) for event in source.events.queue] == [('new', False), ('new', True)]


def test_modifier_map_key_stays_down_until_both_sides_release(source):
    source.configure('Ctrl', ['IdentityV'], action_keys={'bare': '1'})
    for vk, down in [(0xA2, True), (0xA3, True), (0xA2, False), (0xA3, False)]:
        record(source, down, vk)
    assert [(event.action, event.down) for event in source.events.queue] == [('map', True), ('map', False)]


@pytest.mark.parametrize('vk,scan,extended,physical', [
    (0x10, 0x2A, False, 0xA0), (0x10, 0x36, False, 0xA1),
    (0x11, 0x1D, False, 0xA2), (0x11, 0x1D, True, 0xA3),
    (0x12, 0x38, False, 0xA4), (0x12, 0x38, True, 0xA5),
])
def test_raw_keyboard_side_information_is_preserved(vk, scan, extended, physical):
    assert physical_modifier_vk(vk, scan, extended) == physical


@pytest.mark.parametrize('label,vk', [('Num0', 0x60), ('Num1', 0x61), ('Num9', 0x69),
    ('NumAdd', 0x6B), ('NumSubtract', 0x6D), ('NumMultiply', 0x6A), ('NumDivide', 0x6F),
    ('NumDecimal', 0x6E), ('NumSeparator', 0x6C), ('NumEnter', NUM_ENTER), ('Enter', 0x0D)])
def test_numpad_shortcuts_parse_with_modifiers_and_distinct_identities(label, vk):
    shortcut = parse_hotkey('ctrl+shift+' + label.lower())
    assert shortcut.label == 'Ctrl+Shift+' + label and shortcut.vk == vk and shortcut.modifiers == 6
    assert validate_action_hotkeys('G', {'a': label, 'b': 'Ctrl+' + label})


@pytest.mark.parametrize('scan,expected', list(KEYPAD_SCAN_VKS.items()))
def test_raw_numpad_identity_survives_numlock_change_and_does_not_match_navigation(scan, expected):
    nav = {0x52: 0x2D, 0x4F: 0x23, 0x50: 0x28, 0x51: 0x22, 0x4B: 0x25,
           0x4C: 0x0C, 0x4D: 0x27, 0x47: 0x24, 0x48: 0x26, 0x49: 0x21, 0x53: 0x2E}[scan]
    assert physical_key_vk(nav, scan, False) == physical_key_vk(expected, scan, False) == expected
    assert physical_key_vk(nav, scan, True) == nav


def test_raw_enter_keys_are_distinct_and_internal_token_is_not_registered():
    assert physical_key_vk(0x0D, 0x1C, False) == 0x0D
    assert physical_key_vk(0x0D, 0x1C, True) == NUM_ENTER
    with pytest.raises(ValueError, match='Raw Input'):
        GlobalActionHotkeys.registration('NumEnter')


@pytest.mark.parametrize('map_key,held_vk,release_vk', [('Num1', 0x23, 0x23),
    ('NumEnter', 0x0D, 0x0D), ('Enter', 0x0D, NUM_ENTER)])
def test_held_keypad_alias_at_reconfigure_is_released_before_next_press(monkeypatch, map_key, held_vk, release_vk):
    # Model GetAsyncKeyState's ambiguous navigation/Enter state without native listeners.
    from src.hotkeys import polling_vks, key_to_vk
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(
        lambda keys: {key for key in keys if held_vk in polling_vks(key)}))
    source = GlobalMapKeys(map_key, ['IdentityV'], lambda: None)
    actual = key_to_vk(map_key)
    record(source, True, actual)
    assert source.events.empty()
    record(source, False, release_vk)
    record(source, True, actual)
    assert [(event.action, event.down) for event in source.events.queue] == [('map', False), ('map', True)]
