"""Passive Windows mouse input; the hook never touches Qt or consumes input."""
import ctypes
from ctypes import wintypes
from collections import deque
import sys
import threading


def contains(rect, point):
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


class PanInput:
    """Bounded mailbox preserving button edges while coalescing mouse moves."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = deque()
        self.target = None
        self.dragging = False

    def arm(self, epoch, hwnd, roi, excluded):
        with self.lock:
            self.events.clear()
            self.dragging = False
            self.target = (epoch, hwnd, tuple(roi), tuple(excluded))

    def clear(self):
        with self.lock:
            self.target = None
            self.dragging = False
            self.events.clear()

    def feed(self, kind, point, hwnd):
        with self.lock:
            if self.target is None:
                return
            epoch, target_hwnd, roi, excluded = self.target
            if hwnd != target_hwnd:
                self.dragging = False
                self.events.clear()
                self.events.append((epoch, 'cancel', point))
                self.target = None
                return
            inside = contains(roi, point)
            if kind == 'down':
                self.dragging = inside and not any(contains(r, point) for r in excluded)
                if not self.dragging:
                    return
            elif kind in ('move', 'up'):
                if not self.dragging:
                    return
                if kind == 'up':
                    self.dragging = False
            elif kind == 'wheel':
                if not inside:
                    return
                self.dragging = False
                self.target = None
            else:
                return
            if len(self.events) >= 256:
                self.events.clear()
                self.events.append((epoch, 'cancel', point))
                self.dragging = False
                self.target = None
                return
            if kind == 'move' and self.events and self.events[-1][1] == 'move':
                self.events[-1] = (epoch, kind, point)
            else:
                self.events.append((epoch, kind, point))

    def drain(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
            return events


class MousePanHook(PanInput):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.thread_id = None
        self.error = ''
        self.ready = False
        self.stopping = threading.Event()

    def start(self):
        if self.thread is not None and self.thread.is_alive() and not self.error:
            return
        if self.thread is not None and self.thread.is_alive():
            self.stop()
            if self.thread.is_alive():
                self.error = '旧鼠标监听线程尚未退出，请重启程序'
                return
        self.error = ''
        self.stopping.clear()
        if sys.platform != 'win32':
            self.error = '鼠标拖动跟随仅支持 Windows'
            return
        self.thread = threading.Thread(target=self._run, name='map-mouse-pan', daemon=True)
        self.thread.start()

    def stop(self):
        self.clear()
        self.stopping.set()
        if self.thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        if self.thread is not None:
            self.thread.join(timeout=1.)

    def _run(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int,
                                          wintypes.WPARAM, wintypes.LPARAM)

        class MouseData(ctypes.Structure):
            _fields_ = [('pt', wintypes.POINT), ('mouseData', wintypes.DWORD),
                        ('flags', wintypes.DWORD), ('time', wintypes.DWORD),
                        ('dwExtraInfo', ctypes.c_size_t)]

        user32.SetWindowsHookExW.argtypes = (ctypes.c_int, callback_type, wintypes.HINSTANCE, wintypes.DWORD)
        user32.SetWindowsHookExW.restype = wintypes.HHOOK
        user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
        user32.GetForegroundWindow.restype = wintypes.HWND
        kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kinds = {0x0201: 'down', 0x0200: 'move', 0x0202: 'up', 0x020A: 'wheel', 0x020E: 'wheel'}

        @callback_type
        def callback(code, message, address):
            try:
                if code >= 0 and message in kinds:
                    data = ctypes.cast(address, ctypes.POINTER(MouseData)).contents
                    self.feed(kinds[message], (data.pt.x, data.pt.y), int(user32.GetForegroundWindow() or 0))
            except Exception as exc:
                self.error = str(exc)
            return user32.CallNextHookEx(None, code, message, address)

        hook = None
        try:
            # Create the message queue before publishing its thread ID for shutdown.
            msg = wintypes.MSG()
            user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
            self.thread_id = int(kernel32.GetCurrentThreadId())
            hook = user32.SetWindowsHookExW(14, callback, kernel32.GetModuleHandleW(None), 0)
            if not hook:
                raise ctypes.WinError()
            self.ready = True
            while not self.stopping.is_set():
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result == -1:
                    raise ctypes.WinError()
                if result == 0:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if hook:
                user32.UnhookWindowsHookEx(hook)
            self.ready = False
            self.thread_id = None
