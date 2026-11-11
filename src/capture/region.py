"""Capture coordinates shared by GUI and command-line tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass


def crop_image(image, roi=None):
    """Validate an image-local matching ROI; a view preserves original pixels."""
    if roi is None:
        return image
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError('匹配区域必须是四个整数：左、上、宽、高')
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > image.shape[1] or y+height > image.shape[0]:
        raise ValueError('匹配区域必须完全位于截图内且非空')
    return image[y:y+height, x:x+width]


def enable_physical_pixel_coordinates() -> None:
    """Use physical screen pixels for MSS and Windows screen APIs."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_value(cls, value: object) -> "CaptureRegion":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("capture_roi 必须是 [left, top, width, height]")
        try:
            region = cls(*(int(part) for part in value))
        except (TypeError, ValueError) as exc:
            raise ValueError("capture_roi 的四个值必须是整数") from exc
        if region.width <= 0 or region.height <= 0:
            raise ValueError("截图宽度和高度必须大于 0")
        return region

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.width, self.height]

    def as_mss_monitor(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
