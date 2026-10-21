"""Non-consuming Windows Raw Input for map keys, independent of Qt polling."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
import queue
import sys
import threading
import time
from typing import Callable

from src.hotkeys import (key_to_vk, parse_hotkey, modifier_flags, physical_modifier_vk,
                         MODIFIER_GROUPS, MODIFIER_VKS, PHYSICAL_MODIFIER_VKS)


@dataclass(frozen=True)
class MapKeyEvent:
    sequence: int
    revision: int
    action: str
    down: bool
    detected_at: float
    enabled: bool
    foreground_title: str
    foreground_hwnd: int


class GlobalMapKeys:
    """Queue every press/release; only the consumer may touch application state."""

    def __init__(self, map_key: str, titles: list[str], notify: Callable[[], None]):
        self.events: queue.Queue[MapKeyEvent] = queue.Queue()
        self.ready = False
        self.error = ""
        self.thread: threading.Thread | None = None
        self._notify = notify
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hwnd = None
        self._revision = 0
        self._sequence = 0
        self._configuration = None
        self._held: set[int] = set()
        self._pressed_actions: dict[int, str] = {}
        self.configure(map_key, titles)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def configure(self, map_key: str, titles: list[str], *, suspended=False, discard=False, action_keys=None):
        with self._lock:
            actions = (self._configuration[3] if self._configuration else ()) if action_keys is None else tuple(
                (parse_hotkey(key), action) for action, key in action_keys.items())
            configuration = (key_to_vk(map_key), tuple(title.casefold() for title in titles), bool(suspended), actions)
            if configuration == self._configuration and not discard:
                return
            self._configuration = configuration
            self._revision += 1
            # A held key at startup/reconfiguration must first be released.
            self._held = self._read_held(self._watched_keys(configuration))
            flags = modifier_flags(self._held)
            self._pressed_actions = {shortcut.vk: action for shortcut, action in actions
                                     if shortcut.vk in self._held and shortcut.modifiers == flags}
            while True:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    break

    @staticmethod
    def _watched_keys(configuration):
        map_vk = configuration[0]
        keys = {0x1B, *PHYSICAL_MODIFIER_VKS, *(shortcut.vk for shortcut, _ in configuration[3])}
        if map_vk not in MODIFIER_VKS:
            keys.add(map_vk)
        return keys

    @staticmethod
    def _map_down(held, vk):
        group = next((group for group in MODIFIER_GROUPS.values() if group[0] == vk), (vk,))
        return any(key in held for key in group)

    @staticmethod
    def _read_held(keys: set[int]) -> set[int]:
        if sys.platform != "win32":
            return set()
        return {key for key in keys if ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000}

    def _wake(self):
        try:
            self._notify()
        except RuntimeError:
            pass  # The Qt receiver was destroyed during shutdown.

    def _record(self, vk: int, down: bool, foreground: tuple[int, str, int], *, detected_at=None):
        hwnd, title, process_id = foreground
        with self._lock:
            map_vk, titles, suspended, actions = self._configuration
            if vk not in self._watched_keys(self._configuration) and vk not in MODIFIER_VKS:
                return
            if down == (vk in self._held):
                return  # Auto-repeat and duplicate releases do not toggle.
            map_was_down = self._map_down(self._held, map_vk)
            if down:
                self._held.add(vk)
            else:
                self._held.discard(vk)
            if vk in MODIFIER_VKS:
                map_is_down = self._map_down(self._held, map_vk)
                if map_was_down == map_is_down:
                    return  # Modifier edges update state, never synthesize an action press.
                action, down = 'map', map_is_down
            elif vk == map_vk:
                action = 'map'
            elif vk == 0x1B:
                action = 'escape'
            elif down:
                flags = modifier_flags(self._held)
                action = next((action for shortcut, action in actions
                               if shortcut.vk == vk and shortcut.modifiers == flags), None)
                if action is not None:
                    self._pressed_actions[vk] = action
            else:
                action = self._pressed_actions.pop(vk, None)
            if action is None:
                return
            enabled = (not suspended and process_id != os.getpid()
                       and any(token in title.casefold() for token in titles))
            self._sequence += 1
            self.events.put(MapKeyEvent(self._sequence, self._revision,
                action, down,
                time.perf_counter() if detected_at is None else detected_at,
                enabled, title, hwnd))
        self._wake()

    def start(self):
        if self.thread is not None:
            return
        if sys.platform != "win32":
            self.error = "地图按键事件监听仅支持 Windows"
            self._wake()
            return
        self.thread = threading.Thread(target=self._run, name="map-key-raw-input", daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self._hwnd is not None:
            post = ctypes.windll.user32.PostMessageW
            post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            post(self._hwnd, 0x0010, 0, 0)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.)

    def _run(self):
        try:
            self._run_windows()
        except Exception as exc:
            self.error = f"地图按键事件监听失败：{exc}"
            self.ready = False
            self._wake()

    def _run_windows(self):
        user32, kernel32 = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
        wndproc_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                        wintypes.WPARAM, wintypes.LPARAM)

        class WindowClass(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", wndproc_type),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        class RawDevice(ctypes.Structure):
            _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                        ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]

        class RawHeader(ctypes.Structure):
            _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                        ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]

        class RawKeyboard(ctypes.Structure):
            _fields_ = [("MakeCode", wintypes.USHORT), ("Flags", wintypes.USHORT),
                        ("Reserved", wintypes.USHORT), ("VKey", wintypes.USHORT),
                        ("Message", wintypes.UINT), ("ExtraInformation", wintypes.ULONG)]

        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
        user32.RegisterClassW.restype = wintypes.WORD
        user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
            wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = ctypes.c_ssize_t
        user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RawDevice), wintypes.UINT, wintypes.UINT]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
                                         ctypes.POINTER(wintypes.UINT), wintypes.UINT]
        user32.GetRawInputData.restype = wintypes.UINT
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.restype = ctypes.c_ssize_t
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        instance = kernel32.GetModuleHandleW(None)
        class_name = f"IdentityVMapInput_{os.getpid()}_{id(self)}"
        registered_class = registered_input = False

        def read_input(handle):
            size = wintypes.UINT()
            header_size = ctypes.sizeof(RawHeader)
            if user32.GetRawInputData(handle, 0x10000003, None, ctypes.byref(size), header_size) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            data = ctypes.create_string_buffer(size.value)
            if user32.GetRawInputData(handle, 0x10000003, data, ctypes.byref(size), header_size) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if size.value < header_size + ctypes.sizeof(RawKeyboard):
                return
            if RawHeader.from_buffer_copy(data).dwType != 1:
                return
            keyboard = RawKeyboard.from_buffer_copy(data, header_size)
            vk = physical_modifier_vk(keyboard.VKey, keyboard.MakeCode, bool(keyboard.Flags & 2))
            with self._lock:
                if vk not in self._watched_keys(self._configuration):
                    return
            detected_at = time.perf_counter()
            hwnd = user32.GetForegroundWindow()
            title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
            user32.GetWindowTextW(hwnd, title, len(title))
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            self._record(vk, not bool(keyboard.Flags & 1),
                         (hwnd or 0, title.value, pid.value), detected_at=detected_at)

        @wndproc_type
        def window_proc(hwnd, message, wparam, lparam):
            if message == 0x00FF:
                try:
                    read_input(lparam)
                except Exception as exc:
                    self.error = f"地图按键事件读取失败：{exc}"
                    self._wake()
                    user32.PostMessageW(hwnd, 0x0010, 0, 0)
            elif message == 0x0010:
                user32.DestroyWindow(hwnd)
                return 0
            elif message == 0x0002:
                self._hwnd = None
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        try:
            window_class = WindowClass(lpfnWndProc=window_proc, hInstance=instance, lpszClassName=class_name)
            if not user32.RegisterClassW(ctypes.byref(window_class)):
                raise ctypes.WinError(ctypes.get_last_error())
            registered_class = True
            self._hwnd = user32.CreateWindowExW(0, class_name, "", 0, 0, 0, 0, 0,
                                               wintypes.HWND(-3), None, instance, None)
            if not self._hwnd:
                raise ctypes.WinError(ctypes.get_last_error())
            # INPUTSINK receives keys in the background; no NOLEGACY/suppression flags.
            device = RawDevice(1, 6, 0x00000100, self._hwnd)
            if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
                raise ctypes.WinError(ctypes.get_last_error())
            registered_input = self.ready = True
            self._wake()
            message = wintypes.MSG()
            while not self._stop.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == -1:
                    raise ctypes.WinError(ctypes.get_last_error())
                if result == 0:
                    break
                user32.DispatchMessageW(ctypes.byref(message))
        except Exception as exc:
            self.error = f"地图按键事件监听失败：{exc}"
            self._wake()
        finally:
            self.ready = False
            if registered_input:
                device = RawDevice(1, 6, 0x00000001, None)
                user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device))
            if self._hwnd:
                user32.DestroyWindow(self._hwnd)
            self._hwnd = None
            if registered_class:
                user32.UnregisterClassW(class_name, instance)
