"""Windows capture exclusion for tool-owned top-level native windows."""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

WDA_EXCLUDEFROMCAPTURE = 0x00000011


def exclude_native_window(hwnd: int) -> bool:
    """Require real exclusion, never downgrade to a black capture rectangle."""
    if sys.platform != "win32" or sys.getwindowsversion().build < 19041 or not hwnd:
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        setter = user32.SetWindowDisplayAffinity
        setter.argtypes = (wintypes.HWND, wintypes.DWORD)
        setter.restype = wintypes.BOOL
        getter = user32.GetWindowDisplayAffinity
        getter.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        getter.restype = wintypes.BOOL
        if not setter(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return False
        actual = wintypes.DWORD()
        return bool(getter(hwnd, ctypes.byref(actual))) and actual.value == WDA_EXCLUDEFROMCAPTURE
    except (AttributeError, OSError):
        return False
