"""Canonical keyboard shortcuts shared by settings, Qt and Windows listeners."""
from __future__ import annotations

from dataclasses import dataclass
import re

HOTKEYS = tuple(f'F{i}' for i in range(1, 13))
MAP_KEYS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') + (
    'Tab', 'Space', 'Shift', 'Ctrl', 'Alt', 'CapsLock', 'X1', 'X2') + HOTKEYS
ACTION_KEYS = tuple(key for key in MAP_KEYS if key not in ('Shift', 'Ctrl', 'Alt', 'X1', 'X2'))
MODIFIERS = (('Ctrl', 2), ('Shift', 4), ('Alt', 1))
# Include Win in observed state so Win+H never fires a plain H action.
MODIFIER_GROUPS = {2: (0x11, 0xA2, 0xA3), 4: (0x10, 0xA0, 0xA1),
                   1: (0x12, 0xA4, 0xA5), 8: (0x5B, 0x5C)}
MODIFIER_VKS = frozenset(vk for group in MODIFIER_GROUPS.values() for vk in group)
PHYSICAL_MODIFIER_VKS = frozenset((0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C))


def key_to_vk(key: str) -> int:
    if len(key) == 1 and key.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
        return ord(key.upper())
    if re.fullmatch(r'F([1-9]|1[0-2])', key):
        return 0x70 + int(key[1:]) - 1
    special = {'Tab': 0x09, 'Space': 0x20, 'Shift': 0x10, 'Ctrl': 0x11,
               'Alt': 0x12, 'CapsLock': 0x14, 'X1': 0x05, 'X2': 0x06}
    if key not in special:
        raise ValueError(f'不支持的地图按键：{key}')
    return special[key]


@dataclass(frozen=True)
class Hotkey:
    key: str
    modifiers: int = 0

    @property
    def vk(self):
        return key_to_vk(self.key)

    @property
    def label(self):
        return '+'.join([name for name, flag in MODIFIERS if self.modifiers & flag] + [self.key])


def parse_hotkey(value) -> Hotkey:
    if not isinstance(value, str):
        raise ValueError('快捷键必须是按键或组合键文本')
    parts = [part.strip().casefold() for part in value.split('+')]
    keys = {key.casefold(): key for key in ACTION_KEYS}
    modifiers = {name.casefold(): flag for name, flag in MODIFIERS}
    if not parts or parts[-1] not in keys:
        raise ValueError('快捷键须为字母、数字、F1～F12 或常用键，可搭配 Ctrl／Shift／Alt')
    flags = 0
    for part in parts[:-1]:
        if part not in modifiers or flags & modifiers[part]:
            raise ValueError('组合键只支持 Ctrl、Shift、Alt，修饰键不能重复')
        flags |= modifiers[part]
    return Hotkey(keys[parts[-1]], flags)


def normalize_hotkey(value, default=None):
    try:
        return parse_hotkey(value).label
    except ValueError:
        if default is None:
            raise
        return default


def conflicts_with_map(hotkey, map_key):
    shortcut = parse_hotkey(hotkey)
    map_vk = key_to_vk(map_key)
    if shortcut.vk == map_vk:
        return True
    return any(map_vk == group[0] and shortcut.modifiers & flag
               for flag, group in MODIFIER_GROUPS.items())


def validate_action_hotkeys(map_key, shortcuts):
    key_to_vk(map_key)
    normalized = {name: normalize_hotkey(value) for name, value in shortcuts.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError('快捷键已被其他操作使用，请选择其他按键或组合')
    if any(conflicts_with_map(value, map_key) for value in normalized.values()):
        raise ValueError('程序快捷键不能使用游戏地图键作为主键或修饰键')
    return normalized


def modifier_flags(held):
    return sum(flag for flag, keys in MODIFIER_GROUPS.items() if any(key in held for key in keys))


def physical_modifier_vk(vk, scan_code, extended):
    if vk == 0x10:
        return 0xA1 if scan_code == 0x36 else 0xA0
    if vk == 0x11:
        return 0xA3 if extended else 0xA2
    if vk == 0x12:
        return 0xA5 if extended else 0xA4
    return vk
