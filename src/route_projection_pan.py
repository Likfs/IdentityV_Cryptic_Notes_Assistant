"""Input-driven pan and zoom after minimum-scale registration; no frame tracking."""
import ctypes
import threading

import cv2
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal

from src import route_projection_core as core
from src.capture.mouse_pan import MousePanHook
from src.map_motion import MapMotion, ZOOM_CENTER, ZOOM_REFERENCE_SIZE, ZOOM_MIN
from src.game_geometry import map_viewport, map_center
from src.registration.template_matching.floors import load as load_floor


def window_identity(titles):
    client = core.foreground_game_client(titles)
    ctypes.windll.user32.GetForegroundWindow.restype = ctypes.c_void_p
    return (int(ctypes.windll.user32.GetForegroundWindow()), *client.as_list())


def prepare_pan_image(result, opacity):
    """Retain one complete source at native resolution, never one image per zoom."""
    *_, original = load_floor(result.profile_key, result.selected_layout, result.variant_id, result.floor,
        expected_fingerprint=result.registration.validation['template']['source_record_sha256'])
    image = cv2.cvtColor(original, cv2.COLOR_BGRA2RGBA)
    image[:, :, 3] = (image[:, :, 3].astype('float32') * opacity).round().astype('uint8')
    return image


class MousePanSession(QObject):
    prepared = pyqtSignal(object)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.input = MousePanHook()
        self.epoch = 0
        self.state = None
        self.pending = None
        self._zoom_memory = None
        self.prepared.connect(self._receive, Qt.ConnectionType.QueuedConnection)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)

    def _key(self):
        app = self.app
        return (app.session_id, app.map_generation, app.profile.key, app.locked_layout,
                app._effective_template_variant(),
                app.settings.get('registration_backend'), float(app.settings['opacity']))

    @property
    def map_drag_active(self):
        with self.input.lock:
            return self.state is not None and self.input.dragging

    def stop(self):
        if self.state is not None:
            # Preserve wheel edges that arrived just before closing, but no pose.
            valid = not self.input.error and self.state['zoom_trusted']
            for epoch, kind, _point, delta in self.input.drain():
                if epoch != self.epoch:
                    continue
                if kind == 'cancel':
                    valid = False
                    break
                if kind == 'wheel':
                    self.state['motion'].wheel(delta)
            self._zoom_memory = ((self.state['zoom_context'], self.state['motion'].zoom_units,
                                  self.state['identity'][0], self.state['motion'].wheel_remainder) if valid else None)
        self.epoch += 1
        self.state = None
        self.input.clear()
        self.timer.stop()

    def _zoom_context(self):
        app = self.app
        return app.session_id, app.profile.key, app.locked_layout

    def registration_zoom(self):
        if self._zoom_memory is not None and self._zoom_memory[0] == self._zoom_context():
            return self._zoom_memory[1:3]
        return ZOOM_MIN, None

    def forget_zoom(self):
        self._zoom_memory = None

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
            captured_client = getattr(result, 'capture_context', {}).get('game_client_screen')
            if captured_client is not None and list(captured_client) != client.as_list():
                raise ValueError('截图后游戏客户区已改变，请重新配准')
            expected = map_viewport(client).as_list()
            if list(roi) != expected:
                raise ValueError('首次截图与当前地图区域不同')
            detection = getattr(result, 'floor_detection', None) or {}
            buttons = core.floor_control_regions(client, detection.get('buttons'))
            absolute_center = map_center(client)
            center = (absolute_center[0]-roi[0], absolute_center[1]-roi[1])
            motion = MapMotion(result.registration.original_to_screen, center,
                               zoom_units=result.registration.validation.get('map_zoom_units', ZOOM_MIN))
            if (self._zoom_memory is not None and self._zoom_memory[:3] == (
                    self._zoom_context(), motion.zoom_units, identity[0])):
                motion.wheel_remainder = self._zoom_memory[3]
            self.state = dict(key=self._key(), result=result, roi=list(roi), identity=identity,
                              zoom_context=self._zoom_context(), zoom_trusted=True,
                              titles=list(app.settings['game_window_titles']), motion=motion, changed=False,
                              grab=None, prepared=False, preparing=False)
            self.input.start()
            self.input.arm(self.epoch, identity[0], roi, [button.as_list() for button in buttons],
                           wheel_roi=client.as_list())
            self.timer.start()
            self._prepare()
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop()
            app._log(f'鼠标跟随未启动：{exc}；保留静态投影')

    def _valid(self):
        app = self.app
        return (self.state is not None and not app.closed and app.map_state.is_open
                and app.settings_page is None and app.settings.get('show_projection', True)
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
                packet['image'] = prepare_pan_image(state['result'], opacity)
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
            moved = self.state['changed']
            self.app._log(f"鼠标跟随准备失败：{packet['error']}；请重新投影")
            if moved:
                self.app.hide_overlay()
            else:
                self.stop()
            return
        motion = self.state['motion']
        self.app.overlay.set_pan_image(packet['image'], motion.origin, self.state['roi'], pixel_scale=motion.scale)
        self.state['prepared'] = True
        self.app._log('地图平移／缩放跟随已启动：固定步长、边界截断，无自动图像校正')

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
        for epoch, kind, point, delta in self.input.drain():
            if epoch != self.epoch:
                continue
            if kind == 'cancel':
                state['zoom_trusted'] = False
                app.hide_overlay()
                return
            if kind == 'down':
                state['grab'] = point
            elif kind in ('move', 'up', 'wheel'):
                if state['grab'] is not None:
                    anchor = state['grab']
                    state['motion'].pan(point[0] - anchor[0], point[1] - anchor[1])
                    state['grab'] = point
                    state['changed'] |= point != anchor
                if kind == 'up':
                    state['grab'] = None
                if kind == 'wheel':
                    previous = state['motion'].zoom
                    state['motion'].wheel(delta)
                    if state['motion'].zoom != previous:
                        state['changed'] = True
                        app.overlay_var.set(f'路线平移／缩放跟随 · {state["motion"].zoom:.3f}×')
        if state['prepared']:
            app.overlay.set_pan_transform(state['motion'].scale, state['motion'].origin)
