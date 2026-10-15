"""Input-driven translation after template registration; no frame tracking."""
import ctypes
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal

from src import route_projection_core as core
from src.app_catalog import selected_template_variant
from src.capture.mouse_pan import MousePanHook
from src.registration.template_matching.floors import load as load_floor


def window_identity(titles):
    client = core.foreground_game_client(titles)
    ctypes.windll.user32.GetForegroundWindow.restype = ctypes.c_void_p
    return (int(ctypes.windll.user32.GetForegroundWindow()), *client.as_list())


def prepare_pan_image(result, opacity):
    """Scale once in the worker; retain source pixels beyond the initial ROI."""
    pose = np.asarray(result.registration.original_to_screen, dtype=np.float64)
    if (pose.shape != (3, 3) or not np.isfinite(pose).all() or pose[0, 0] <= 0
            or not np.allclose(pose[:2, :2], np.eye(2) * pose[0, 0])
            or not np.allclose(pose[2], [0, 0, 1])):
        raise ValueError('鼠标跟随需要等比例缩放和平移矩阵')
    *_, original = load_floor(result.profile_key, result.selected_layout, result.variant_id, result.floor,
        expected_fingerprint=result.registration.validation['template']['source_record_sha256'])
    scale = float(pose[0, 0])
    height, width = original.shape[:2]
    size = (int(np.ceil(width * scale)) + 1, int(np.ceil(height * scale)) + 1)
    image = cv2.warpAffine(original, np.array([[scale, 0, 0], [0, scale, 0]]), size,
                           flags=cv2.INTER_LINEAR)
    image[:, :, 3] = np.rint(image[:, :, 3].astype(np.float32) * opacity).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), tuple(pose[:2, 2])


class MousePanSession(QObject):
    prepared = pyqtSignal(object)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.input = MousePanHook()
        self.epoch = 0
        self.state = None
        self.pending = None
        self.prepared.connect(self._receive, Qt.ConnectionType.QueuedConnection)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)

    def _key(self):
        app = self.app
        return (app.session_id, app.map_generation, app.profile.key, app.locked_layout,
                selected_template_variant(app.settings, app.profile.key),
                app.settings.get('registration_backend'), float(app.settings['opacity']))

    @property
    def map_drag_active(self):
        with self.input.lock:
            return self.state is not None and self.input.dragging

    def stop(self):
        self.epoch += 1
        self.state = None
        self.input.clear()
        self.timer.stop()

    def shutdown(self):
        self.stop()
        self.input.stop()

    def start(self, result, roi):
        self.stop()
        app = self.app
        if (app.settings.get('registration_backend') != 'template'
                or not app.settings.get('template_motion_tracking', True)
                or getattr(getattr(result, 'registration', None), 'original_to_screen', None) is None):
            return
        try:
            identity = window_identity(app.settings['game_window_titles'])
            if getattr(result, 'capture_window_id', None) not in (None, identity[0]):
                raise ValueError('首次截图与当前游戏窗口不同')
            client = core.CaptureRegion.from_value(identity[1:])
            expected = core.game_capture_region(client, app.settings['capture_roi'],
                                                app.settings['capture_reference_size']).as_list()
            if list(roi) != expected:
                raise ValueError('首次截图与当前地图区域不同')
            floor = core.game_capture_region(client, list(core.FLOOR_CONTROL_ROI),
                                             app.settings['capture_reference_size']).as_list()
            self.state = dict(key=self._key(), result=result, roi=list(roi), identity=identity,
                              titles=list(app.settings['game_window_titles']), offset=(0., 0.),
                              grab=None, prepared=False, preparing=False)
            self.input.start()
            self.input.arm(self.epoch, identity[0], roi, [floor])
            self.timer.start()
            self._prepare()
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop()
            app._log(f'鼠标跟随未启动：{exc}；保留静态投影')

    def _valid(self):
        app = self.app
        return (self.state is not None and not app.closed and app.map_state.is_open
                and app.settings_page is None and app.show_overlay_check.isChecked()
                and app.overlay.visible and app.settings.get('template_motion_tracking', True)
                and self.state['key'] == self._key())

    def _prepare(self):
        if self.pending is not None or self.state is None or self.state['preparing']:
            return
        epoch, state = self.epoch, self.state
        self.pending = epoch
        state['preparing'] = True
        opacity = float(self.app.settings['opacity'])

        def worker():
            packet = dict(epoch=epoch, state=state, error=None)
            try:
                packet['image'], packet['origin'] = prepare_pan_image(state['result'], opacity)
            except Exception as exc:
                packet['error'] = str(exc)
            try:
                self.prepared.emit(packet)
            except RuntimeError:
                pass
        threading.Thread(target=worker, name='map-pan-image', daemon=True).start()

    def _receive(self, packet):
        if self.pending == packet['epoch']:
            self.pending = None
        if packet['epoch'] != self.epoch or packet['state'] is not self.state:
            return
        self.tick()  # Recheck the game and consume pending input before exposing pixels.
        if packet['epoch'] != self.epoch or packet['state'] is not self.state or not self._valid():
            return
        if packet['error']:
            moved = self.state['offset'] != (0., 0.)
            self.app._log(f"鼠标跟随准备失败：{packet['error']}；请重新投影")
            if moved:
                self.app.hide_overlay()
            else:
                self.stop()
            return
        self.app.overlay.set_pan_image(packet['image'], packet['origin'], self.state['roi'])
        self.state['prepared'] = True
        self.app.overlay.set_pan_offset(*self.state['offset'])
        self.app._log('鼠标平移跟随已启动：不支持缩放，无自动图像校正')

    def tick(self):
        if self.state is None:
            return
        app = self.app
        app._handle_map_key()
        app._handle_template_floor_click()
        if not self._valid():
            app.hide_overlay()
            return
        try:
            if window_identity(self.state['titles']) != self.state['identity']:
                raise ValueError('游戏窗口已改变')
        except (OSError, ValueError, RuntimeError):
            app.hide_overlay()
            return
        if self.input.error:
            app._log(f'鼠标跟随停止：{self.input.error}；请重新投影')
            app.hide_overlay()
            return
        self._prepare()
        state = self.state
        for epoch, kind, point in self.input.drain():
            if epoch != self.epoch:
                continue
            if kind in ('wheel', 'cancel'):
                app.hide_overlay()
                if kind == 'wheel':
                    app.overlay_var.set('当前不支持缩放跟随，请恢复开图尺度后重新投影')
                    app._log('检测到地图滚轮操作：投影已隐藏；请恢复开图尺度后重新投影')
                return
            if kind == 'down':
                state['grab'] = (point, state['offset'])
            elif kind in ('move', 'up') and state['grab'] is not None:
                anchor, offset = state['grab']
                state['offset'] = (offset[0] + point[0] - anchor[0], offset[1] + point[1] - anchor[1])
                if kind == 'up':
                    state['grab'] = None
        if state['prepared']:
            app.overlay.set_pan_offset(*state['offset'])
