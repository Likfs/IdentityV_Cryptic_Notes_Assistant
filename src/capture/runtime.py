"""Tk-free screen capture and global hotkey services for the PyQt app."""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
from ctypes import wintypes

import cv2
import mss
import numpy as np
from mss.exception import ScreenShotError
from PIL import Image

from src.app_catalog import key_to_vk
from src.capture.region import CaptureRegion


def capture_region(region: CaptureRegion) -> Image.Image:
    try:
        with mss.MSS() as screen:
            shot = screen.grab(region.as_mss_monitor())
    except ScreenShotError as exc:
        raise RuntimeError("Windows 截图失败，请确认屏幕未锁定且游戏地图已打开。") from exc
    image = Image.frombytes("RGB", shot.size, shot.rgb)
    if image.size != (region.width, region.height):
        raise RuntimeError(f"截图尺寸异常：{image.size}，预期 {(region.width, region.height)}")
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


class GlobalActionHotkeys:
    IDS = {"recognize": 0x6401, "project": 0x6402}

    def __init__(self, recognize_key: str, project_key: str) -> None:
        self.keys = {"recognize": recognize_key, "project": project_key}
        self.events: queue.Queue[str] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.thread_id: int | None = None
        self.ready = False
        self.error = ""
        self._stop = threading.Event()

    @staticmethod
    def virtual_key(label: str) -> int:
        return key_to_vk(label)

    def start(self) -> None:
        if sys.platform != "win32":
            self.error = "全局快捷键仅在 Windows 可用。"
            return
        self.thread = threading.Thread(target=self._run, name="route-app-hotkeys", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self.thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        registered = []
        try:
            for action, hotkey_id in self.IDS.items():
                key = self.keys[action]
                if not user32.RegisterHotKey(None, hotkey_id, 0x4000, self.virtual_key(key)):
                    raise RuntimeError(f"{key} 注册失败，可能被其他程序占用。")
                registered.append(hotkey_id)
            self.ready = True
            message = wintypes.MSG()
            while not self._stop.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                if message.message == 0x0312:
                    for action, hotkey_id in self.IDS.items():
                        if int(message.wParam) == hotkey_id:
                            self.events.put(action)
                            break
        except Exception as exc:
            self.error = str(exc)
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)
            self.ready = False
            self.thread_id = None
