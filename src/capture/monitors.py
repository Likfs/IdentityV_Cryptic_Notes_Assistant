"""Shared monitor types and a process-owned, once-configured WinAPI binding."""
import ctypes
from ctypes import wintypes
from threading import Lock


class MonitorInfo(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32))


MonitorEnumProc = getattr(ctypes, 'WINFUNCTYPE', ctypes.CFUNCTYPE)(
    wintypes.BOOL, wintypes.HANDLE, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

_api = None
_api_lock = Lock()


def monitor_api():
    """Publish fully declared functions before either GUI or worker can call them."""
    global _api
    if _api is None:
        with _api_lock:
            if _api is None:
                # A private DLL wrapper also isolates these declarations from windll's
                # cached function objects. Queries never change argtypes/restype.
                user = ctypes.WinDLL('user32')
                user.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
                user.MonitorFromPoint.restype = wintypes.HANDLE
                user.GetMonitorInfoW.argtypes = (wintypes.HANDLE, ctypes.POINTER(MonitorInfo))
                user.GetMonitorInfoW.restype = wintypes.BOOL
                user.EnumDisplayMonitors.argtypes = (
                    wintypes.HDC, ctypes.POINTER(wintypes.RECT), MonitorEnumProc, wintypes.LPARAM)
                user.EnumDisplayMonitors.restype = wintypes.BOOL
                _api = user
    return _api
