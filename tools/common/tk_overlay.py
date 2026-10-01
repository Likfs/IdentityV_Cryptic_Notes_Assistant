from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from ctypes import wintypes

import numpy as np
from PIL import Image, ImageTk


class POINT(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class SIZE(ctypes.Structure):
    _fields_ = (("cx", wintypes.LONG), ("cy", wintypes.LONG))


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = (
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    )


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


class BITMAPINFO(ctypes.Structure):
    _fields_ = (("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3))


def exclude_window_from_capture(window: tk.Misc) -> bool:
    """Ask Windows capture APIs to omit this app-owned top-level window."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    get_parent = user32.GetParent
    get_parent.argtypes = (wintypes.HWND,)
    get_parent.restype = wintypes.HWND
    child = int(window.winfo_id())
    hwnd = int(get_parent(child)) or child
    set_affinity = user32.SetWindowDisplayAffinity
    set_affinity.argtypes = (wintypes.HWND, wintypes.DWORD)
    set_affinity.restype = wintypes.BOOL
    return bool(set_affinity(hwnd, 0x00000011))


class RouteOverlayWindow:
    """Click-through, per-pixel-alpha overlay positioned over the capture ROI."""

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_NOACTIVATE = 0x08000000
    ULW_ALPHA = 0x00000002
    AC_SRC_OVER = 0x00
    AC_SRC_ALPHA = 0x01
    BI_RGB = 0
    DIB_RGB_COLORS = 0

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.photo: ImageTk.PhotoImage | None = None
        self.label = tk.Label(self.window, borderwidth=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)
        self.visible = False
        self.capture_excluded = False

    @staticmethod
    def supported() -> bool:
        return sys.platform == "win32"

    def _hwnd(self) -> int:
        child = int(self.window.winfo_id())
        if sys.platform != "win32":
            return child
        get_parent = ctypes.windll.user32.GetParent
        get_parent.argtypes = (wintypes.HWND,)
        get_parent.restype = wintypes.HWND
        parent = int(get_parent(child))
        return parent or child

    def show(self, bgra: np.ndarray, left: int, top: int) -> None:
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("覆盖层必须为 BGRA 四通道图片")
        height, width = bgra.shape[:2]
        self.window.geometry(f"{width}x{height}+{int(left)}+{int(top)}")
        self.window.deiconify()
        self.window.update_idletasks()
        if sys.platform == "win32":
            self._show_win32(bgra)
            self.capture_excluded = exclude_window_from_capture(self.window)
        else:
            rgba = bgra[:, :, [2, 1, 0, 3]]
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgba, "RGBA"))
            self.label.configure(image=self.photo)
            self.capture_excluded = False
        self.visible = True

    def _show_win32(self, bgra: np.ndarray) -> None:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        user32.GetDC.restype = wintypes.HDC
        user32.GetDC.argtypes = (wintypes.HWND,)
        user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
        user32.ReleaseDC.restype = ctypes.c_int
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
        gdi32.CreateDIBSection.restype = wintypes.HANDLE
        gdi32.CreateDIBSection.argtypes = (
            wintypes.HDC,
            ctypes.POINTER(BITMAPINFO),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        )
        gdi32.SelectObject.restype = wintypes.HANDLE
        gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HANDLE)
        gdi32.DeleteObject.argtypes = (wintypes.HANDLE,)
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = (wintypes.HDC,)
        gdi32.DeleteDC.restype = wintypes.BOOL
        hwnd = self._hwnd()
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_style.restype = ctypes.c_ssize_t
        get_style.argtypes = (wintypes.HWND, ctypes.c_int)
        set_style.restype = ctypes.c_ssize_t
        set_style.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
        user32.UpdateLayeredWindow.argtypes = (
            wintypes.HWND,
            wintypes.HDC,
            ctypes.POINTER(POINT),
            ctypes.POINTER(SIZE),
            wintypes.HDC,
            ctypes.POINTER(POINT),
            wintypes.COLORREF,
            ctypes.POINTER(BLENDFUNCTION),
            wintypes.DWORD,
        )
        user32.UpdateLayeredWindow.restype = wintypes.BOOL
        style = int(get_style(hwnd, self.GWL_EXSTYLE))
        set_style(
            hwnd,
            self.GWL_EXSTYLE,
            style | self.WS_EX_LAYERED | self.WS_EX_TRANSPARENT | self.WS_EX_TOOLWINDOW | self.WS_EX_NOACTIVATE,
        )

        pixels = np.ascontiguousarray(bgra.copy())
        alpha = pixels[:, :, 3:4].astype(np.float32) / 255.0
        pixels[:, :, :3] = np.clip(pixels[:, :, :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        height, width = pixels.shape[:2]

        screen_dc = user32.GetDC(None)
        memory_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap_info = BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = -height
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = self.BI_RGB
        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(
            screen_dc,
            ctypes.byref(bitmap_info),
            self.DIB_RGB_COLORS,
            ctypes.byref(bits),
            None,
            0,
        )
        if not bitmap or not bits.value:
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(None, screen_dc)
            raise RuntimeError("创建 Windows 透明覆盖位图失败")
        previous = gdi32.SelectObject(memory_dc, bitmap)
        try:
            ctypes.memmove(bits.value, pixels.ctypes.data, pixels.nbytes)
            destination = POINT(self.window.winfo_x(), self.window.winfo_y())
            size = SIZE(width, height)
            source = POINT(0, 0)
            blend = BLENDFUNCTION(self.AC_SRC_OVER, 0, 255, self.AC_SRC_ALPHA)
            ok = user32.UpdateLayeredWindow(
                hwnd,
                screen_dc,
                ctypes.byref(destination),
                ctypes.byref(size),
                memory_dc,
                ctypes.byref(source),
                0,
                ctypes.byref(blend),
                self.ULW_ALPHA,
            )
            if not ok:
                raise ctypes.WinError()
        finally:
            gdi32.SelectObject(memory_dc, previous)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(None, screen_dc)

    def hide(self) -> None:
        self.window.withdraw()
        self.visible = False

    def destroy(self) -> None:
        try:
            self.window.destroy()
        finally:
            self.visible = False
