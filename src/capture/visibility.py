"""One capture policy for all Qt top-level windows, including future dialogs."""
from __future__ import annotations

import ctypes
import sys
import weakref
from ctypes import wintypes

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget


def set_capture_excluded(hwnd: int, excluded: bool) -> bool:
    if sys.platform != 'win32':
        return False
    setter = ctypes.windll.user32.SetWindowDisplayAffinity
    setter.argtypes = (wintypes.HWND, wintypes.DWORD)
    setter.restype = wintypes.BOOL
    return bool(setter(hwnd, 0x11 if excluded else 0))


def flush_desktop():
    """Wait for native visibility changes before capturing the desktop."""
    if sys.platform == 'win32':
        result = ctypes.windll.dwmapi.DwmFlush()
        if result < 0:
            raise RuntimeError('桌面合成尚未完成，已取消本次截图，请重试')


class CaptureVisibilityPolicy(QObject):
    status_changed = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.excluded = True
        self._captures: set[str] = set()
        self._applying: set[int] = set()
        self._failures: dict[int, str] = {}
        self._windows = weakref.WeakSet()
        self._status = ''
        self.app = QApplication.instance()
        self.app._capture_visibility_policy = self

    @property
    def effective_excluded(self):
        return self.excluded or bool(self._captures)

    @property
    def status(self):
        if self._failures:
            return '部分窗口捕获设置失败，请检查系统支持并重试'
        return '防捕获已开启' if self.excluded else '已允许截图和录屏（取决于捕获方式）'

    def _notify(self):
        status = self.status
        if status != self._status:
            self._status = status
            self.status_changed.emit(status)

    def register(self, widget):
        if widget not in self._windows:
            self._windows.add(widget)
            widget.installEventFilter(self)
        return self.apply(widget)

    def apply(self, widget, *, create=True):
        if sip.isdeleted(widget):
            return True
        identity = id(widget)
        if identity in self._applying:
            return True
        self._applying.add(identity)
        try:
            hwnd = int((widget.winId() if create else widget.effectiveWinId()) or 0)
            if not hwnd:
                return True  # Destruction also sends WinIdChange; never recreate its HWND.
            success = set_capture_excluded(hwnd, self.effective_excluded)
            widget.capture_excluded = success and self.effective_excluded
            if success:
                self._failures.pop(identity, None)
            else:
                self._failures[identity] = widget.windowTitle()
            self._notify()
            return success
        finally:
            self._applying.discard(identity)

    def _apply_all(self):
        success = True
        windows = [widget for widget in self._windows if not sip.isdeleted(widget)]
        identities = {id(widget) for widget in windows}
        self._failures = {key: value for key, value in self._failures.items() if key in identities}
        for widget in windows:
            if widget.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                applied = self.apply(widget, create=False)
                if widget.isVisible() and not applied:
                    success = False
        self._notify()
        return success

    def set_excluded(self, excluded):
        self.excluded = bool(excluded)
        self._apply_all()

    def begin_capture(self, token):
        was_excluded = self.effective_excluded
        self._captures.add(token)
        # Preserve the default capture path; allowing capture needs a clean frame.
        if not was_excluded and not self._apply_all():
            self.end_capture(token)
            raise RuntimeError('无法在内部截图时排除本程序窗口；请开启防捕获或重试')

    def end_capture(self, token):
        if token not in self._captures:
            return
        self._captures.discard(token)
        if not self.effective_excluded:
            self._apply_all()

    def eventFilter(self, watched, event):
        if (isinstance(watched, QWidget) and watched.isWindow()
                and event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange)
                and watched.testAttribute(Qt.WidgetAttribute.WA_WState_Created)):
            self.apply(watched, create=False)
        return False

    def close(self):
        self._captures.clear()
        for widget in self._windows:
            if not sip.isdeleted(widget):
                widget.removeEventFilter(self)
        self._windows.clear()
        if getattr(self.app, '_capture_visibility_policy', None) is self:
            del self.app._capture_visibility_policy


def apply_window_capture_policy(widget):
    policy = getattr(QApplication.instance(), '_capture_visibility_policy', None)
    if policy is None:
        return set_capture_excluded(int(widget.winId()), True)
    return policy.register(widget) and policy.effective_excluded
