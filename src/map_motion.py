"""Input-driven source-to-ROI geometry, independent of Qt and registration."""
import operator
import numpy as np
from src.game_geometry import REFERENCE_MAP_CENTER, REFERENCE_SIZE


# Measured from the user's 2560x1440 map captures. Zoom is additive, with
# the final upward step clamped halfway; reversing subtracts a full step.
ZOOM_CENTER = REFERENCE_MAP_CENTER
ZOOM_REFERENCE_SIZE = REFERENCE_SIZE
ZOOM_MIN = 7
ZOOM_MAX = 26
ZOOM_STEP = 2
WHEEL_DELTA = 120


def zoom_factor(units):
    try:
        value = operator.index(units)
    except TypeError as exc:
        raise ValueError('地图倍率必须为已跟踪的整数步长') from exc
    if isinstance(units, bool) or not ZOOM_MIN <= value <= ZOOM_MAX:
        raise ValueError('地图倍率超出已跟踪范围')
    return value / ZOOM_MIN


class MapMotion:
    def __init__(self, pose, center, *, zoom_units=ZOOM_MIN):
        pose = np.asarray(pose, dtype=np.float64)
        if (pose.shape != (3, 3) or not np.isfinite(pose).all() or pose[0, 0] <= 0
                or not np.allclose(pose[:2, :2], np.eye(2) * pose[0, 0])
                or not np.allclose(pose[2], [0, 0, 1])):
            raise ValueError('鼠标跟随需要等比例缩放和平移矩阵')
        self.scale = float(pose[0, 0])
        self.origin = tuple(pose[:2, 2])
        self.center = tuple(center)
        zoom_factor(zoom_units)
        self.zoom_units = int(zoom_units)
        self.wheel_remainder = 0

    @property
    def zoom(self):
        return self.zoom_units / ZOOM_MIN

    def pan(self, dx, dy):
        self.origin = (self.origin[0] + dx, self.origin[1] + dy)

    def wheel(self, delta):
        total = self.wheel_remainder + delta
        steps = (1 if total >= 0 else -1) * (abs(total) // WHEEL_DELTA)
        self.wheel_remainder = total - steps * WHEEL_DELTA
        previous = self.zoom_units
        self.zoom_units = min(ZOOM_MAX, max(ZOOM_MIN, previous + steps * ZOOM_STEP))
        # Input beyond a boundary must not delay the first reverse wheel step.
        if ((self.zoom_units == ZOOM_MAX and self.wheel_remainder > 0)
                or (self.zoom_units == ZOOM_MIN and self.wheel_remainder < 0)):
            self.wheel_remainder = 0
        ratio = self.zoom_units / previous
        self.scale *= ratio
        self.origin = tuple(c + ratio * (p - c) for p, c in zip(self.origin, self.center))
