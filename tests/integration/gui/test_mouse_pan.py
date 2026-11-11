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
        overlay.set_pan_transform(1., (-200, 0))
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
        overlay.set_pan_transform(1., (-199.75, 0))
        assert overlay.grab().toImage().pixelColor(round(30/dpr), round(30/dpr)).alpha() > 240
        original = overlay._pan_pixmap.cacheKey()
        for step in range(100):
            overlay.set_pan_transform(1., (-199.75 + step * .25, 0))
        assert overlay._pan_origin == (-175., 0.)
        assert overlay._pan_pixmap.cacheKey() == original
    finally:
        overlay.close()


@pytest.mark.parametrize('dpr', [1., 1.25, 1.5, 2.])
def test_zoom_renders_native_source_at_physical_scale_without_resizing_pixmap(qt_app, monkeypatch, dpr, tmp_path):
    monkeypatch.setattr(ui, 'physical_to_qt_geometry',
                        lambda x, y, w, h: (x, y, round(w/dpr), round(h/dpr), dpr))
    overlay = ui.RouteOverlayWindow()
    try:
        overlay.show(np.zeros((100, 150, 4), np.uint8))
        overlay.set_pan_image(source(), (0, 0), [0, 0, 150, 100], pixel_scale=.5)
        pixmap = overlay._pan_pixmap.cacheKey()
        overlay.set_pan_transform(2., (-400., 0.))
        rendered = overlay.grab().toImage()
        assert rendered.pixelColor(round(60/dpr), round(60/dpr)).red() == 255
        assert rendered.pixelColor(round(60/dpr), round(60/dpr)).alpha() == 255
        assert rendered.pixelColor(round(20/dpr), round(60/dpr)).alpha() == 0
        assert overlay._pan_pixmap.size().width() == 300
        assert overlay._pan_pixmap.cacheKey() == pixmap
        assert rendered.save(str(tmp_path/'zoom_native_source.png'))
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
    # Motion-only cases use a small viewport; the geometry test below restores the real one.
    monkeypatch.setattr(pan, 'map_viewport', lambda client:pan.core.game_capture_region(
        client, window.settings['capture_roi'], [2560, 1440]))
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
    assert window.overlay._pan_origin == (80, 30)
    drag(session, (280, 230), (200, 200))
    assert window.overlay._pan_origin == (0, 0)
    assert window.overlay._pan_pixmap.cacheKey() == pixmap
    np.testing.assert_array_equal(result.registration.original_to_screen, np.eye(3))


@pytest.mark.parametrize('change', ['close', 'session', 'generation', 'layout', 'version', 'backend',
                                    'settings', 'disabled', 'foreground', 'window', 'hidden'])
def test_session_changes_discard_pending_drag(running, monkeypatch, change):
    window, session, _ = running
    session.input.feed('down', (200, 200), 42)
    session.input.feed('up', (280, 230), 42)
    session.input.feed('wheel', (280, 230), 42, 120)
    if change == 'close': window.map_state.is_open = False
    elif change == 'session': window.session_id = 'new'
    elif change == 'generation': window.map_generation += 1
    elif change == 'layout': window.locked_layout = 'layout_02'
    elif change == 'version': window.settings['template_route_versions'][window.profile.key] = 'clean'
    elif change == 'backend': window.settings['registration_backend'] = 'cpu'
    elif change == 'settings': window.settings_page = SimpleNamespace()
    elif change == 'disabled': window.settings['template_motion_tracking'] = False
    elif change == 'hidden': window.hide_overlay()
    elif change == 'foreground': monkeypatch.setattr(pan, 'window_identity', lambda _: (99, 0, 0, 2560, 1440))
    else: monkeypatch.setattr(pan, 'window_identity', lambda _: (42, 5, 0, 2560, 1440))
    session.tick()
    assert session.state is None and not window.overlay.visible
    if change == 'settings': window.settings_page = None


def test_wheel_and_drag_follow_without_new_capture_load_or_pixmap(running, monkeypatch):
    window, session, _ = running
    pixmap = window.overlay._pan_pixmap.cacheKey()
    monkeypatch.setattr(pan, 'load_floor', lambda *a, **kw:pytest.fail('Wheel must reuse its source image'))
    session.input.feed('wheel', (200, 200), 42, 120)
    session.tick()
    motion = session.state['motion']
    assert window.overlay.visible and motion.zoom == pytest.approx(9 / 7)
    assert window.overlay._pan_pixel_scale == pytest.approx(9 / 7)
    np.testing.assert_allclose(window.overlay._pan_origin, np.array(motion.center) * (1 - 9 / 7))
    origin = np.array(motion.origin)
    drag(session, (200, 200), (250, 200))
    np.testing.assert_allclose(window.overlay._pan_origin, origin + [50, 0])
    # The wheel center is independent of cursor position, even outside the crop.
    session.input.feed('wheel', (1500, 900), 42, -120)
    session.tick()
    np.testing.assert_allclose(window.overlay._pan_origin, [50 * 7 / 9, 0], atol=1e-10)
    assert window.overlay._pan_pixel_scale == pytest.approx(1.)
    assert window.overlay._pan_pixmap.cacheKey() == pixmap


def test_wheel_between_drag_events_preserves_both_transforms(running):
    window, session, _ = running
    session.input.feed('down', (200, 200), 42)
    session.input.feed('move', (230, 210), 42)
    session.input.feed('wheel', (230, 210), 42, 120)
    session.input.feed('up', (250, 220), 42)
    session.tick()
    center = np.array(session.state['motion'].center)
    np.testing.assert_allclose(window.overlay._pan_origin, center + 9 / 7 * ([30, 10] - center) + [20, 10])
    assert session.state['grab'] is None and window.overlay.visible


def test_close_remembers_pending_wheel_but_reopen_uses_freshly_registered_position(running):
    window, session, result = running
    session.input.feed('wheel', (200, 200), 42, 120)
    session.tick()
    drag(session, (200, 200), (260, 230))
    session.input.feed('wheel', (260, 230), 42, 120)  # Close before the next GUI tick.
    window.map_state.is_open = False
    window.map_generation += 1
    window.hide_overlay()
    assert session.state is None and session.registration_zoom() == (11, 42)
    window.map_state.is_open = True
    window.map_generation += 1
    result.registration.original_to_screen = np.array([[.6, 0, 17], [0, .6, 29], [0, 0, 1.]])
    result.registration.validation['map_zoom_units'] = 11
    window.overlay.show(np.zeros((600, 800, 4), np.uint8), 100, 100)
    session.start(result, [100, 100, 800, 600])
    wait_for(lambda:session.state['prepared'])
    np.testing.assert_allclose(window.overlay._pan_origin, [17, 29])
    assert session.state['motion'].zoom == pytest.approx(11 / 7)
    session.input.feed('wheel', (200, 200), 42, 120)
    session.tick()
    assert window.overlay._pan_pixel_scale == pytest.approx(.6 * 13 / 11)


def test_new_layout_or_round_cannot_reuse_previous_zoom(running):
    window, session, _ = running
    session.input.feed('wheel', (200, 200), 42, 240)
    window.hide_overlay()
    assert session.registration_zoom() == (11, 42)
    window.clear_layout_lock()
    assert session.registration_zoom() == (7, None)


def test_partial_wheel_is_preserved_across_reprojection(running):
    window, session, result = running
    session.input.feed('wheel', (200, 200), 42, 60)
    window.hide_overlay()
    assert session.registration_zoom() == (7, 42)
    window.overlay.show(np.zeros((600, 800, 4), np.uint8), 100, 100)
    session.start(result, [100, 100, 800, 600])
    wait_for(lambda:session.state['prepared'])
    session.input.feed('wheel', (200, 200), 42, 60)
    session.tick()
    assert session.state['motion'].zoom == pytest.approx(9 / 7)


def test_lost_input_discards_remembered_zoom(running):
    window, session, _ = running
    session.input.feed('wheel', (200, 200), 42, 120)
    session.tick()
    for _ in range(200):
        session.input.feed('wheel', (200, 200), 42, 120)
        session.input.feed('wheel', (200, 200), 42, -120)
    session.tick()
    assert session.state is None and session.registration_zoom() == (7, None)


@pytest.mark.parametrize('identity', [(42, 0, 0, 2560, 1440), (42, -100, 50, 1920, 1080),
                                     (42, 300, -90, 1280, 720), (42, -100, 50, 2560, 1334),
                                     (42, 300, -90, 3440, 1440)])
def test_zoom_center_uses_game_client_and_crop_origin(running, monkeypatch, identity):
    window, session, result = running
    monkeypatch.setattr(pan, 'window_identity', lambda _:identity)
    client = pan.core.CaptureRegion.from_value(identity[1:])
    from src.game_geometry import map_viewport
    monkeypatch.setattr(pan, 'map_viewport', map_viewport)
    roi = map_viewport(client).as_list()
    session.start(result, roi)
    wait_for(lambda:session.state['prepared'])
    from src.game_geometry import map_center
    expected = tuple(a-b for a, b in zip(map_center(client), roi[:2]))
    np.testing.assert_allclose(session.state['motion'].center, expected)
    session.input.feed('wheel', (client.left + 20, client.top + 20), 42, 120)
    session.tick()
    np.testing.assert_allclose(window.overlay._pan_origin, np.array(expected) * (1 - 9 / 7))


def test_resize_cannot_start_motion_from_old_capture_but_keeps_known_zoom(running, monkeypatch):
    window, session, result = running
    session.input.feed('wheel', (200, 200), 42, 120)
    window.hide_overlay()
    assert session.registration_zoom() == (9, 42)
    result.capture_context = dict(game_client_screen=[0, 0, 2560, 1440])
    monkeypatch.setattr(pan, 'window_identity', lambda _:(42, 0, 0, 2560, 1334))
    from src.game_geometry import map_viewport
    monkeypatch.setattr(pan, 'map_viewport', map_viewport)
    client = pan.core.CaptureRegion(0, 0, 2560, 1334)
    session.start(result, map_viewport(client).as_list())
    assert session.state is None
    assert session.registration_zoom() == (9, 42)


def test_late_preparation_cannot_restore_hidden_projection(running, monkeypatch):
    window, session, result = running
    entered, release = threading.Event(), threading.Event()
    def prepare(*_):
        entered.set()
        assert release.wait(3)
        return source()
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
        return source()
    monkeypatch.setattr(pan, 'prepare_pan_image', prepare)
    session.start(result, [100, 100, 800, 600])
    assert entered.wait(2)
    drag(session, (200, 200), (290, 220))
    session.input.feed('wheel', (290, 220), 42, 240)
    session.tick()
    release.set()
    wait_for(lambda: session.state['prepared'])
    center = np.array(session.state['motion'].center)
    np.testing.assert_allclose(window.overlay._pan_origin, center + 11 / 7 * ([90, 20] - center))
    assert window.overlay._pan_pixel_scale == pytest.approx(11 / 7)


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
    assert window.overlay._pan_origin == (800, 0)
