import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from PyQt6.QtTest import QTest
from src.capture.mouse_pan import PanInput
from src import route_projection_pan as pan
from src import route_projection_qt_ui as ui
from src.app_catalog import selected_template_variant


def source():
    rgba = np.zeros((80, 300, 4), dtype=np.uint8)
    rgba[20:40, 220:240] = [255, 0, 0, 255]
    return rgba


def test_pan_reveals_full_source_and_clips_to_fixed_viewport_and_clears_before_hide(qt_app):
    overlay = ui.RouteOverlayWindow()
    try:
        overlay.show(np.zeros((80, 100, 4), np.uint8), 100, 150)
        overlay.set_pan_image(source(), (0., 0.), [100, 150, 100, 80])
        geometry = overlay.geometry()
        assert overlay.grab().toImage().pixelColor(30, 30).alpha() == 0
        overlay.set_pan_offset(-200, 0)
        image = overlay.grab().toImage()
        assert image.pixelColor(30, 30).red() == 255
        assert image.pixelColor(30, 30).alpha() == 255
        assert image.width() == 100 and overlay.geometry() == geometry
        overlay.hide()
        assert overlay._pan_pixmap is None
        assert overlay.grab().toImage().pixelColor(30, 30).alpha() == 0
        overlay.show(np.zeros((80, 100, 4), np.uint8), 100, 150)
        assert overlay.label.isVisible()
    finally:
        overlay.close()


@pytest.mark.parametrize('dpr', [1., 1.25, 1.5, 2.])
def test_physical_displacement_and_fractional_origin_at_dpi(qt_app, monkeypatch, dpr):
    monkeypatch.setattr(ui, 'physical_to_qt_geometry',
                        lambda x, y, w, h: (x, y, round(w/dpr), round(h/dpr), dpr))
    overlay = ui.RouteOverlayWindow()
    try:
        overlay.show(np.zeros((80, 150, 4), np.uint8))
        overlay.set_pan_image(source(), (.25, 0.), [0, 0, 150, 80])
        overlay.set_pan_offset(-200, 0)
        assert overlay.grab().toImage().pixelColor(round(30/dpr), round(30/dpr)).alpha() > 240
        original = overlay._pan_pixmap.cacheKey()
        for step in range(100):
            overlay.set_pan_offset(-200 + step * .25, 0)
        assert overlay._pan_offset == (-175.25, 0.)
        assert overlay._pan_pixmap.cacheKey() == original
    finally:
        overlay.close()


class FakeHook(PanInput):
    error = ''
    ready = True
    def start(self):
        pass
    def stop(self):
        self.clear()


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(5)
    assert predicate()


@pytest.fixture
def running(window, monkeypatch):
    monkeypatch.setattr(pan, 'window_identity', lambda _: (42, 0, 0, 2560, 1440))
    monkeypatch.setattr(window, '_handle_map_key', lambda: None)
    monkeypatch.setattr(window, '_handle_template_floor_click', lambda: None)
    monkeypatch.setattr(ui, 'physical_to_qt_geometry', lambda x, y, w, h: (x, y, w, h, 1.))
    monkeypatch.setattr(window.overlay, '_prepare_native_window', lambda: None)
    window.settings.update(registration_backend='template', template_motion_tracking=True,
                           capture_roi=[100, 100, 800, 600])
    session = window._motion
    assert isinstance(session, pan.MousePanSession)
    session.input = FakeHook()
    original = source()[:, :, [2, 1, 0, 3]].copy()
    monkeypatch.setattr(pan, 'load_floor', lambda *_, **kw: (None, None, None, None, original))
    def no_capture(*_):
        pytest.fail('Mouse following must not capture frames')
    monkeypatch.setattr(pan.core, 'capture_bgr', no_capture)
    result = SimpleNamespace(registration=SimpleNamespace(original_to_screen=np.eye(3), validation={'template': {'source_record_sha256': 'test'}}), floor='b1',
                             capture_window_id=42, profile_key=window.profile.key,
                             selected_layout=window.locked_layout,
                             variant_id=selected_template_variant(window.settings, window.profile.key))
    window.overlay.show(np.zeros((600, 800, 4), np.uint8), 100, 100)
    session.start(result, [100, 100, 800, 600])
    wait_for(lambda: session.state is not None and session.state['prepared'])
    session.timer.stop()
    return window, session, result


def drag(session, start, end):
    session.input.feed('down', start, 42)
    session.input.feed('up', end, 42)
    session.tick()


def test_stationary_and_multiple_short_drags_need_no_capture_or_new_image(running):
    window, session, result = running
    pixmap = window.overlay._pan_pixmap.cacheKey()
    for _ in range(10):
        session.tick()
    assert window.overlay.visible
    drag(session, (200, 200), (280, 230))
    assert window.overlay._pan_offset == (80, 30)
    drag(session, (280, 230), (200, 200))
    assert window.overlay._pan_offset == (0, 0)
    assert window.overlay._pan_pixmap.cacheKey() == pixmap
    np.testing.assert_array_equal(result.registration.original_to_screen, np.eye(3))


@pytest.mark.parametrize('change', ['close', 'session', 'generation', 'layout', 'version', 'backend',
                                    'settings', 'disabled', 'foreground', 'window', 'hidden'])
def test_session_changes_discard_pending_drag(running, monkeypatch, change):
    window, session, _ = running
    session.input.feed('down', (200, 200), 42)
    session.input.feed('up', (280, 230), 42)
    if change == 'close': window.map_state.is_open = False
    elif change == 'session': window.session_id = 'new'
    elif change == 'generation': window.map_generation += 1
    elif change == 'layout': window.locked_layout = 'layout_02'
    elif change == 'version': window.settings['template_route_versions'][window.profile.key] = 'route_clean'
    elif change == 'backend': window.settings['registration_backend'] = 'cpu'
    elif change == 'settings': window.settings_page = SimpleNamespace()
    elif change == 'disabled': window.settings['template_motion_tracking'] = False
    elif change == 'hidden': window.hide_overlay()
    elif change == 'foreground': monkeypatch.setattr(pan, 'window_identity', lambda _: (99, 0, 0, 2560, 1440))
    else: monkeypatch.setattr(pan, 'window_identity', lambda _: (42, 5, 0, 2560, 1440))
    session.tick()
    assert session.state is None and not window.overlay.visible
    if change == 'settings': window.settings_page = None


def test_wheel_hides_without_correction_or_resuming_on_mouse_move(running):
    window, session, _ = running
    session.input.feed('wheel', (200, 200), 42)
    session.tick()
    assert session.state is None and not window.overlay.visible
    assert '不支持缩放' in window.overlay_var.get()
    drag(session, (200, 200), (250, 200))
    assert not window.overlay.visible


def test_late_preparation_cannot_restore_hidden_projection(running, monkeypatch):
    window, session, result = running
    entered, release = threading.Event(), threading.Event()
    def prepare(*_):
        entered.set()
        assert release.wait(3)
        return source(), (0., 0.)
    monkeypatch.setattr(pan, 'prepare_pan_image', prepare)
    session.start(result, [100, 100, 800, 600])
    assert entered.wait(2)
    window.hide_overlay()
    release.set()
    wait_for(lambda: session.pending is None)
    assert session.state is None and not window.overlay.visible
    assert window.overlay._pan_pixmap is None


def test_drag_during_preparation_is_applied_when_full_image_arrives(running, monkeypatch):
    window, session, result = running
    entered, release = threading.Event(), threading.Event()
    def prepare(*_):
        entered.set()
        assert release.wait(3)
        return source(), (0., 0.)
    monkeypatch.setattr(pan, 'prepare_pan_image', prepare)
    session.start(result, [100, 100, 800, 600])
    assert entered.wait(2)
    drag(session, (200, 200), (290, 220))
    release.set()
    wait_for(lambda: session.state['prepared'])
    assert window.overlay._pan_offset == (90, 20)


def test_hook_error_stops_and_hides_instead_of_displaying_stale_position(running):
    window, session, _ = running
    session.input.error = 'hook failed'
    session.tick()
    assert session.state is None and not window.overlay.visible


def test_drag_crossing_floor_button_is_not_a_floor_click(running, monkeypatch):
    from src import route_projection_app as app
    window, session, _ = running
    monkeypatch.setattr(app, 'floor_control_mouse_down', lambda *_: True)
    generation = window.map_generation
    session.input.feed('down', (200, 200), 42)
    session.input.feed('move', (1000, 200), 42)
    app.RouteProjectionApp._handle_template_floor_click(window)
    assert window.map_generation == generation and window.overlay.visible
    session.input.feed('up', (1000, 200), 42)
    session.tick()
    assert window.overlay._pan_offset == (800, 0)
