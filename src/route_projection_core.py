from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from ctypes import wintypes

import numpy as np

from src.app_catalog import key_to_vk
from src.hotkeys import polling_vks
from src.capture.region import CaptureRegion
from src.capture.runtime import capture_region, pil_to_bgr
FLOOR_CONTROL_ROI = (1870, 70, 390, 66)
from src.projection_workflow import WorkflowResult


AUTO = "自动判断"


def capture_bgr(roi: list[int]) -> np.ndarray:
    return pil_to_bgr(capture_region(CaptureRegion.from_value(roi)))


def floor_control_mouse_down(titles: list[str], reference_size: list[int]) -> bool:
    """Observe a click in the floor control, without capturing or classifying it."""
    if sys.platform != 'win32' or not (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000):
        return False
    try:
        client = foreground_game_client(titles)
        region = game_capture_region(client, list(FLOOR_CONTROL_ROI), reference_size)
        point = wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return False
        return region.left <= point.x < region.left+region.width and region.top <= point.y < region.top+region.height
    except (OSError, ValueError, RuntimeError):
        return False


def game_capture_region(client: CaptureRegion, roi: list[int], reference_size: list[int]) -> CaptureRegion:
    """Map a client-local reference ROI to physical screen pixels."""
    reference_width, reference_height = reference_size
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("游戏参考分辨率必须大于 0")
    local = CaptureRegion.from_value(roi)
    if local.left < 0 or local.top < 0 or local.left + local.width > reference_width or local.top + local.height > reference_height:
        raise ValueError("地图区域必须完全位于游戏参考画面内")
    x0 = round(local.left * client.width / reference_width)
    y0 = round(local.top * client.height / reference_height)
    x1 = round((local.left + local.width) * client.width / reference_width)
    y1 = round((local.top + local.height) * client.height / reference_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("游戏窗口太小，地图截图区域无效")
    return CaptureRegion(client.left + x0, client.top + y0, x1 - x0, y1 - y0)


def foreground_game_client(titles: list[str]) -> CaptureRegion:
    """Locate the visible game client area; never substitute a desktop ROI."""
    if sys.platform != "win32":
        raise RuntimeError("仅支持在 Windows 游戏窗口中截图")
    if not titles or not game_window_is_foreground(titles):
        raise RuntimeError("游戏窗口不在前台，请切回游戏后按 F8 或地图键")
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if user32.IsIconic(hwnd):
        raise RuntimeError("游戏窗口已最小化，无法截取地图区域")
    user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.ClientToScreen.restype = wintypes.BOOL
    client_rect = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError()
    width = int(client_rect.right - client_rect.left)
    height = int(client_rect.bottom - client_rect.top)
    if width <= 0 or height <= 0:
        raise RuntimeError("游戏窗口已最小化或画面区域无效")
    return CaptureRegion(int(origin.x), int(origin.y), width, height)


def ensure_region_on_screen(region: CaptureRegion) -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    left = user32.GetSystemMetrics(76)
    top = user32.GetSystemMetrics(77)
    right = left + user32.GetSystemMetrics(78)
    bottom = top + user32.GetSystemMetrics(79)
    if (region.left < left or region.top < top or
            region.left + region.width > right or region.top + region.height > bottom):
        raise RuntimeError("地图截图区域超出可见屏幕，请移动或调整游戏窗口")


def map_key_down(key: str) -> bool:
    if sys.platform != "win32":
        return False
    return any(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000 for vk in polling_vks(key_to_vk(key)))


def escape_key_down() -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)


def own_window_is_foreground() -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    hwnd = user32.GetForegroundWindow()
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return process_id.value == os.getpid()


def foreground_window_title() -> str:
    if sys.platform != "win32":
        return ""
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowTextW.restype = ctypes.c_int
    buffer = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def game_window_is_foreground(titles: list[str]) -> bool:
    """Shared UI/capture focus check; foreground title lookup remains injectable."""
    if not titles:
        return False
    current = foreground_window_title().casefold()
    return any(title.casefold() in current for title in titles)


def opacity_to_percent(opacity: float) -> int:
    return round(min(1.0, max(0.0, opacity)) * 100)


def percent_to_opacity(percent: float) -> float:
    return round(min(100.0, max(0.0, percent)) / 100.0, 4)


@dataclass
class MapToggleState:
    is_open: bool = False
    key_was_down: bool = False
    escape_was_down: bool = False
    open_on_release: bool = False

    def observe(self, key_down: bool, *, enabled: bool = True) -> str | None:
        pressed_now = key_down and not self.key_was_down
        released_now = not key_down and self.key_was_down
        self.key_was_down = key_down
        if not enabled:
            self.open_on_release = False
            return None
        if pressed_now:
            if self.is_open:
                # Hide early when closing; this release must not reopen the map.
                self.is_open = False
                self.open_on_release = False
                return 'closed'
            self.open_on_release = True
        elif released_now:
            opening = self.open_on_release
            self.open_on_release = False
            if opening:
                self.is_open = True
                return 'opened'
        return None

    def reset(self, key_down: bool = False) -> None:
        self.is_open = False
        self.key_was_down = key_down
        self.escape_was_down = False
        self.open_on_release = False

    def observe_escape(self, key_down: bool, *, enabled: bool = True) -> bool:
        pressed_now = key_down and not self.escape_was_down
        self.escape_was_down = key_down
        if not enabled or not pressed_now or not (self.is_open or self.open_on_release):
            return False
        self.is_open = False
        self.open_on_release = False
        return True


def format_projection_metrics(result: WorkflowResult) -> str:
    validation = result.registration.validation if result.registration else {}
    timing = result.timings_ms
    template = validation.get('template', {})
    preview = ('后台处理中' if getattr(result, 'projection_preview_pending', False)
               else f"{timing.get('projection_composite', 0):.0f} ms")
    submit = f"{timing['overlay_submit']:.1f} ms" if 'overlay_submit' in timing else '未提交'
    delivery = (f" · 回传 {timing['result_delivery']:.1f} ms · 覆盖提交 {submit}"
                if 'result_delivery' in timing else '')
    end_to_end = (f" · 按键检测至覆盖提交 {timing['key_to_overlay_submit']:.0f} ms"
                  if 'key_to_overlay_submit' in timing else '')
    scheduling = (f" · 按键至请求 {timing['key_to_request']:.0f} ms"
                  if 'key_to_request' in timing else '')
    if 'request_setup' in timing:
        scheduling += (f" · 请求准备 {timing['request_setup']:.1f} ms"
                       f" · 线程启动 {timing.get('worker_start_delay', 0):.1f} ms")
    return (f"完整原图配准：{'候选已生成' if result.accepted else '未生成候选'} · 定尺度模板匹配不分楼层\n"
            f"尺度 {template.get('scale', 0):.4f} · 原图位置 {template.get('location', '—')} · 请核对投影位置\n"
            f"耗时：截图 {timing.get('capture', 0):.0f} ms · "
            f"配准 {timing.get('registration', 0):.0f} ms · 投影图层 {timing.get('projection_layer', timing.get('projection', 0)):.0f} ms · "
            f"预览合成 {preview}{delivery}{scheduling}{end_to_end}")
