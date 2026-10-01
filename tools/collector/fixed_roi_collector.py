from __future__ import annotations

import ctypes
import queue
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any, Optional

import mss
from PIL import Image

from src.capture.region import CaptureRegion, enable_physical_pixel_coordinates
from tools.common.capture_exclusion import exclude_native_window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Compatibility exports for existing integrations; new tools import the services directly.
from tools.common.capture_config import DEFAULT_CONFIG_PATH, MIN_INTERVAL_SECONDS, load_capture_config
from tools.common.sample_store import (
    DEFAULT_LABELS_PATH, DEFAULT_ROI_ROOT, PENDING_LAYOUT_ID, PENDING_LAYOUT_NAME,
    Category, LayoutCategory, SampleWriter, load_category_config, safe_label, image_sha256,
)
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

VOICE_PROMPTS = {
    "app_started": "采集程序已启动",
    "preview_success": "预览截图成功",
    "sampling_started": "开始连续采集",
    "sampling_paused": "采集已暂停",
    "sampling_resumed": "继续采集",
    "sampling_stopped": "采集已停止",
    "single_saved": "单张采集完成",
    "duplicate_skipped": "重复画面，已跳过",
    "layout_confirmed": "布局归档完成",
    "layout_supplemented": "样本已补充到已有布局",
    "layout_empty": "本轮没有可归档图片",
    "sample_deleted": "样本已删除",
    "jsonl_rebuilt": "索引重建完成",
    "no_pending": "没有待确认样本",
    "operation_failed": "操作失败",
}

VOICE_FALLBACK_TONES = {
    "app_started": ((820, 90), (1080, 120)),
    "preview_success": ((980, 100),),
    "sampling_started": ((900, 180),),
    "sampling_paused": ((560, 180),),
    "sampling_resumed": ((720, 90), (920, 120)),
    "sampling_stopped": ((460, 220),),
    "single_saved": ((1040, 90),),
    "duplicate_skipped": ((420, 70), (420, 70)),
    "layout_confirmed": ((800, 90), (1050, 90), (1250, 130)),
    "layout_supplemented": ((740, 80), (940, 100), (1140, 80), (1320, 130)),
    "layout_empty": ((380, 110), (320, 150)),
    "sample_deleted": ((520, 100), (360, 150)),
    "jsonl_rebuilt": ((680, 80), (880, 80), (1080, 120)),
    "no_pending": ((400, 100), (400, 100)),
    "operation_failed": ((280, 180), (220, 260)),
}


class ChineseVoiceNotifier:
    """Play serialized Chinese prompts through Windows SAPI without blocking the UI."""

    def __init__(self) -> None:
        self.messages: queue.Queue[Optional[str]] = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.available = sys.platform == "win32"
        self.tts_available = self.available

    def start(self) -> None:
        if not self.available or (self.thread is not None and self.thread.is_alive()):
            return
        self.thread = threading.Thread(target=self._run, name="collector-chinese-voice", daemon=True)
        self.thread.start()

    def speak(self, text: str) -> None:
        if self.available and text.strip():
            self.messages.put(text.strip().replace("\r", " ").replace("\n", " "))

    def stop(self) -> None:
        if self.thread is not None:
            self.messages.put(None)

    @staticmethod
    def _play_fallback_tone(text: str) -> None:
        try:
            import winsound

            prompt = next((key for key, value in VOICE_PROMPTS.items() if value == text), "operation_failed")
            for frequency, duration in VOICE_FALLBACK_TONES[prompt]:
                winsound.Beep(frequency, duration)
        except (ImportError, RuntimeError, OSError):
            pass

    def _run(self) -> None:
        comtypes_module = None
        voice = None
        try:
            import comtypes
            from comtypes.client import CreateObject

            comtypes_module = comtypes
            comtypes.CoInitialize()
            voice = CreateObject("SAPI.SpVoice")
            chinese_voices = voice.GetVoices("Language=804")
            if chinese_voices.Count > 0:
                voice.Voice = chinese_voices.Item(0)
        except Exception:
            self.tts_available = False

        try:
            while True:
                message = self.messages.get()
                if message is None:
                    break
                if voice is not None and self.tts_available:
                    try:
                        voice.Speak(message)
                        continue
                    except Exception:
                        self.tts_available = False
                self._play_fallback_tone(message)
        finally:
            if comtypes_module is not None:
                try:
                    comtypes_module.CoUninitialize()
                except Exception:
                    pass


class GlobalHotkeyManager:
    HOTKEYS = {
        0x6101: ("start", MOD_NOREPEAT, 0x75, "F6"),
        0x6102: ("pause", MOD_NOREPEAT, 0x76, "F7"),
        0x6103: ("stop", MOD_NOREPEAT, 0x77, "F8"),
        0x6104: ("capture", MOD_NOREPEAT, 0x79, "F10"),
        0x6105: ("preview", MOD_NOREPEAT, 0x74, "F5"),
        0x6106: ("confirm", MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x4E, "Ctrl+Alt+N"),
        0x6107: ("rebuild", MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x52, "Ctrl+Alt+R"),
        0x6108: ("supplement", MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x45, "Ctrl+Alt+E"),
    }

    def __init__(self) -> None:
        self.events: queue.Queue[str] = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.thread_id: Optional[int] = None
        self.ready = False
        self.errors: list[str] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        if sys.platform != "win32":
            self.errors.append("全局快捷键仅支持 Windows")
            return
        self.thread = threading.Thread(target=self._run, name="collector-global-hotkeys", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.thread_id is not None:
            try:
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
            except Exception:
                pass

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self.thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        registered: list[int] = []
        for hotkey_id, (_action, modifiers, virtual_key, label) in self.HOTKEYS.items():
            if user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
                registered.append(hotkey_id)
            else:
                self.errors.append(f"{label} 注册失败，可能被其他程序占用")
        self.ready = bool(registered)
        message = wintypes.MSG()
        try:
            while not self._stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                if message.message == WM_HOTKEY:
                    hotkey = self.HOTKEYS.get(int(message.wParam))
                    if hotkey is not None:
                        self.events.put(hotkey[0])
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)
            self.thread_id = None


def border_rectangles(region: CaptureRegion, thickness: int) -> tuple[tuple[int, int, int, int], ...]:
    """Four strips outside the capture rectangle, so the border is never saved in a PNG."""
    if thickness <= 0:
        raise ValueError("边框宽度必须大于 0")
    left, top, width, height = region.as_list()
    return (
        (left - thickness, top - thickness, width + 2 * thickness, thickness),
        (left - thickness, top + height, width + 2 * thickness, thickness),
        (left - thickness, top, thickness, height),
        (left + width, top, thickness, height),
    )


class ScreenRegionBorder:
    """Independent, topmost Windows strips that do not take focus or intercept clicks."""

    WS_EX_TOPMOST = 0x00000008
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_LAYERED = 0x00080000
    WS_EX_NOACTIVATE = 0x08000000
    WS_POPUP = 0x80000000
    SS_WHITERECT = 0x0006
    SS_BLACKRECT = 0x0004
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    LWA_ALPHA = 0x0002

    def __init__(self, region: CaptureRegion) -> None:
        self.region = region
        self.windows: list[int] = []

    @property
    def visible(self) -> bool:
        return bool(self.windows)

    def show(self) -> None:
        if self.visible:
            return
        if sys.platform != "win32":
            raise RuntimeError("屏幕边框仅支持 Windows")
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.CreateWindowExW.argtypes = (
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
        )
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.SetWindowPos.argtypes = (
            wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetLayeredWindowAttributes.argtypes = (
            wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD,
        )
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        extended_style = (
            self.WS_EX_TOPMOST | self.WS_EX_TRANSPARENT | self.WS_EX_TOOLWINDOW
            | self.WS_EX_LAYERED | self.WS_EX_NOACTIVATE
        )
        try:
            # The white 3 px inner line and black 3 px outer line remain visible on
            # both light and dark game backgrounds. Both layers stay outside the ROI.
            for thickness, static_style in ((6, self.SS_BLACKRECT), (3, self.SS_WHITERECT)):
                for x, y, width, height in border_rectangles(self.region, thickness):
                    hwnd = user32.CreateWindowExW(
                        extended_style, "STATIC", None, self.WS_POPUP | static_style,
                        x, y, width, height, None, None, None, None,
                    )
                    if not hwnd:
                        raise ctypes.WinError(ctypes.get_last_error())
                    self.windows.append(hwnd)
                    if not exclude_native_window(int(hwnd)):
                        raise RuntimeError("无法将截图边框排除捕获，已关闭边框")
                    if not user32.SetLayeredWindowAttributes(hwnd, 0, 245, self.LWA_ALPHA):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if not user32.SetWindowPos(
                        hwnd, wintypes.HWND(-1), x, y, width, height,
                        self.SWP_NOACTIVATE | self.SWP_SHOWWINDOW,
                    ):
                        raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.hide()
            raise

    def hide(self) -> None:
        if not self.windows:
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.DestroyWindow.argtypes = (wintypes.HWND,)
        user32.DestroyWindow.restype = wintypes.BOOL
        for hwnd in reversed(self.windows):
            user32.DestroyWindow(hwnd)
        self.windows.clear()


def virtual_screen_bounds(sct: mss.mss) -> CaptureRegion:
    monitor = sct.monitors[0]
    return CaptureRegion(
        int(monitor["left"]),
        int(monitor["top"]),
        int(monitor["width"]),
        int(monitor["height"]),
    )


def validate_region_inside_screen(region: CaptureRegion, screen: CaptureRegion) -> None:
    if (
        region.left < screen.left
        or region.top < screen.top
        or region.left + region.width > screen.left + screen.width
        or region.top + region.height > screen.top + screen.height
    ):
        raise ValueError(
            f"截图区域 {region.as_list()} 超出虚拟桌面 {screen.as_list()}，"
            "请检查显示器排列、分辨率和缩放设置。"
        )


def grab_region(region: CaptureRegion) -> Image.Image:
    with mss.mss() as sct:
        validate_region_inside_screen(region, virtual_screen_bounds(sct))
        shot = sct.grab(region.as_mss_monitor())
        return Image.frombytes("RGB", shot.size, shot.rgb)


def run_app(
    config_path: Path = DEFAULT_CONFIG_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
) -> None:
    from tools.collector.collector_ui import run_app as run_qt_app
    run_qt_app(config_path, labels_path)
