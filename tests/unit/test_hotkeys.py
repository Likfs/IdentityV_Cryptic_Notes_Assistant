from types import SimpleNamespace
import ctypes
from ctypes import wintypes

import pytest

from src.hotkeys import (normalize_hotkey, parse_hotkey, validate_action_hotkeys,
                         physical_modifier_vk)
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
