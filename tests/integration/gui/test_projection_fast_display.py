from dataclasses import replace
from pathlib import Path
import threading
import time
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from src import projection_workflow as workflow
from src import route_projection_app as app
from src.registration.template_matching import direct
from tests.support.app_fakes import FakeApp
from tests.support.projection_results import result_for


def enable_template(window,monkeypatch):
    monkeypatch.setattr(window,'_handle_map_key',lambda:None)
    # Captures are simulated; real desktop focus must not change their HWND.
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:42)
    monkeypatch.setattr(app,'floor_control_mouse_down',lambda *a:False)
    window.apply_settings({**window.settings,'registration_backend':'template'})
    window.settings['show_projection'] = True


def test_deferred_preview_keeps_identical_layer_and_composite(window,direct_inputs,monkeypatch):
    expected=result_for(window,direct_inputs,False)
    def forbidden(*a):raise AssertionError('preview computed before layer delivery')
    monkeypatch.setattr(direct,'composite_route_layer',forbidden)
    ready=result_for(window,direct_inputs)
    assert ready.accepted and ready.projection is None and ready.projection_preview_pending
    np.testing.assert_array_equal(ready.layer,expected.layer)
    complete=workflow.complete_projection_preview(ready)
    np.testing.assert_array_equal(complete.projection,expected.projection)
    assert ready.projection is None and ready.projection_preview_pending
    assert not complete.projection_preview_pending
    assert 'projection_composite' not in ready.timings_ms
    assert complete.timings_ms['projection_compute_total']>=complete.timings_ms['projection_layer']


def test_real_worker_displays_before_slow_preview_and_saves_complete_result(window,direct_inputs,monkeypatch):
    from tests.support.floor_ui import projection_frame, template_control
    enable_template(window,monkeypatch)
    from src.capture.region import CaptureRegion
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    window.settings['capture_roi'] = [100, 100, 40, 40]
    monkeypatch.setattr(app, 'map_match_region', lambda client, view:view)
    # Tiny injected viewport isolates layer delivery from full-map geometry.
    monkeypatch.setattr(app, 'map_viewport', lambda c:app.game_capture_region(
        c, window.settings['capture_roi'], [2560, 1440]))
    monkeypatch.setattr(app,'capture_bgr',lambda region:projection_frame(
        region, direct_inputs, (100, 100), control=template_control(4)))
    shown=[];saved=[];entered=threading.Event();release=threading.Event();done=threading.Event()
    owner=threading.get_ident()
    monkeypatch.setattr(window.overlay,'show',lambda *a:shown.append(threading.get_ident()))
    real=app.complete_projection_preview
    def slow(result):
        entered.set();assert release.wait(3)
        return real(result)
    monkeypatch.setattr(app,'complete_projection_preview',slow)
    def save(result):
        saved.append(result);done.set();return Path('complete-only')
    monkeypatch.setattr(app,'save_workflow_result',save)
    try:
        window.projection_triggered_at = time.perf_counter() - .15
        window.request_work('project',auto=True)
        qt=QApplication.instance();deadline=time.monotonic()+2
        while not entered.is_set() and time.monotonic()<deadline:
            qt.processEvents();time.sleep(.002)
        assert entered.is_set() and shown==[owner]
        assert window.last_result.projection is None and not window.busy
        assert window.last_result.timings_ms['key_to_overlay_submit'] >= 150
        assert window.last_result.timings_ms['result_queue'] >= 0
        assert window.last_result.timings_ms['request_setup'] >= 0
        assert window.last_result.timings_ms['worker_start_delay'] >= 0
        assert '按键检测至覆盖提交' in window.metric_var.get()
        assert window.last_result.gui_diagnostics['at_delivery']['log_entries'] > 0
        assert not saved
        release.set();assert done.wait(2)
        qt.processEvents()
        assert shown==[owner]  # Preview does not submit the overlay a second time.
        assert window.last_result.projection is not None
        assert saved[0].projection is not None and not saved[0].projection_preview_pending
        assert saved[0].floor_detection['floor'] == '2nd'
        assert window._floor_buttons == saved[0].floor_detection['buttons']
        assert saved[0].capture_context['game_client_size'] == [2560, 1440]
        assert saved[0].capture_context['map_roi_screen'] == [100, 100, 40, 40]
        assert saved[0].floor_ui_image is not None and saved[0].floor_ui_normalized is not None
        assert 'overlay_submit' in saved[0].timings_ms
        assert saved[0].diagnostics()['gui_diagnostics']['at_request']['log_entries'] >= 0
    finally:
        release.set();done.wait(2)


@pytest.mark.parametrize('failure', ['no_match', 'template_error'])
def test_real_worker_saves_ui_crop_and_client_context_when_floor_detection_fails(window, monkeypatch, tmp_path, failure):
    from src.capture.region import CaptureRegion
    from src import floor_indicator
    from tests.support.floor_ui import projection_frame
    from tests.support.qt_wait import wait_for
    enable_template(window, monkeypatch)
    window.settings['capture_roi'] = [100, 100, 40, 40]
    monkeypatch.setattr(app, 'map_match_region', lambda client, view:view)
    # Tiny injected viewport isolates layer delivery from full-map geometry.
    monkeypatch.setattr(app, 'map_viewport', lambda c:app.game_capture_region(
        c, window.settings['capture_roi'], [2560, 1440]))
    client = CaptureRegion(240, 100, 1920, 1080)
    map_region = app.game_capture_region(client, window.settings['capture_roi'], [2560, 1440])
    monkeypatch.setattr(app, 'foreground_game_client', lambda _:client)
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _:None)
    monkeypatch.setattr(app, 'capture_bgr', lambda region:projection_frame(region,
        np.full((30, 30, 3), 120, np.uint8), (map_region.left, map_region.top),
        client=client, control=np.full((510, 140, 3), 100, np.uint8)))
    if failure == 'template_error':
        def broken_templates():
            raise ValueError('测试模板无法读取')
        monkeypatch.setattr(floor_indicator, '_templates', broken_templates)
    directory = tmp_path/'failed_ui'
    saved = threading.Event()
    def save(result):
        path = workflow.save_workflow_result(result, directory)
        saved.set()
        return path
    monkeypatch.setattr(app, 'save_workflow_result', save)
    window.request_work('project', auto=False)
    wait_for(saved.is_set)
    import json
    import cv2
    data = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    assert not data['accepted'] and data['registration'] is None
    assert data['capture_context']['game_client_size'] == [1920, 1080]
    assert data['capture_context']['game_client_screen'] == [240, 100, 1920, 1080]
    assert data['capture_context']['floor_ui_roi_screen'] == [1867, 235, 106, 383]
    assert cv2.imread(str(directory/'02_floor_ui_raw.png')).shape == (383, 106, 3)
    if failure == 'no_match':
        assert len(data['floor_detection']['candidates']) == 7
        assert (directory/'03_floor_ui_normalized.png').is_file()
    else:
        assert '模板无法读取' in data['floor_detection']['error']
        assert data['floor_ui_images']['normalized'] is None


@pytest.mark.parametrize('change',['closed','generation','session','new_result','backend'])
def test_late_preview_never_restores_old_ui(window,direct_inputs,monkeypatch,change):
    enable_template(window,monkeypatch)
    ready=result_for(window,direct_inputs);complete=workflow.complete_projection_preview(ready)
    generation,session=window.map_generation,window.session_id
    window.last_result=ready
    if change=='closed':window.map_state.is_open=False
    elif change=='generation':window.map_generation+=1
    elif change=='session':window.session_id='new'
    elif change=='new_result':window.last_result=replace(ready)
    else:window.settings['registration_backend']='cpu'
    before=window.last_result
    monkeypatch.setattr(window,'_set_preview',lambda *a:pytest.fail('stale screenshot applied'))
    window._receive_work_event(('preview',generation,session,ready,complete))
    assert window.last_result is before


def test_completion_starts_only_latest_pending_request_without_100ms_timer(monkeypatch):
    window=FakeApp();window.busy=True
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    app.RouteProjectionApp._auto_if_open(window,7)
    window.map_generation=8
    app.RouteProjectionApp._auto_if_open(window,8)
    assert not window.scheduled and not window.requests
    window.busy=False
    app.RouteProjectionApp._drain_pending_projection(window)
    assert window.requests==[('project',True)] and window._pending_auto_projection is None


def test_pending_request_cancelled_by_close(monkeypatch):
    window=FakeApp();window.busy=True
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    app.RouteProjectionApp._auto_if_open(window,7)
    window.map_generation=8;window.map_state.is_open=False;window.busy=False
    app.RouteProjectionApp._drain_pending_projection(window)
    assert not window.requests and window._pending_auto_projection is None


def test_preview_failure_does_not_reject_ready_overlay(window,direct_inputs,monkeypatch):
    ready=result_for(window,direct_inputs)
    monkeypatch.setattr(workflow,'composite_route_layer',lambda *a:(_ for _ in ()).throw(RuntimeError('preview failure')))
    complete=workflow.complete_projection_preview(ready)
    assert complete.accepted and complete.layer is ready.layer
    assert complete.projection is None and complete.preview_error=='preview failure'


@pytest.mark.parametrize('change',['closed_map','generation','session','backend','layout','focus'])
def test_stale_ready_layer_is_discarded_and_worker_released(window,direct_inputs,monkeypatch,change):
    enable_template(window,monkeypatch)
    ready=result_for(window,direct_inputs)
    generation,session=window.map_generation,window.session_id
    if change=='closed_map':window.map_state.is_open=False
    elif change=='generation':window.map_generation+=1
    elif change=='session':window.session_id='new'
    elif change=='backend':window.settings['registration_backend']='cpu'
    elif change=='layout':window.locked_layout='layout_12'
    else:monkeypatch.setattr(app,'game_window_is_foreground',lambda _:False)
    monkeypatch.setattr(window.overlay,'show',lambda *a:pytest.fail('stale overlay shown'))
    meta=dict(posted_at=time.perf_counter(),requested_at=time.perf_counter(),session=session,ack=threading.Event())
    window.busy=True
    window._receive_work_event(('ready',('work','project',True,generation,[0,0,40,40],ready,None),meta))
    assert meta['ack'].is_set() and not window.busy and window.last_result is None


def test_overlay_submission_error_releases_worker_and_busy_state(window,direct_inputs,monkeypatch):
    enable_template(window,monkeypatch)
    ready=result_for(window,direct_inputs)
    monkeypatch.setattr(window.overlay,'show',lambda *a:(_ for _ in ()).throw(RuntimeError('submit failure')))
    meta=dict(posted_at=time.perf_counter(),requested_at=time.perf_counter(),session=window.session_id,ack=threading.Event())
    window.busy=True
    window._receive_work_event(('ready',('work','project',True,window.map_generation,[0,0,40,40],ready,None),meta))
    assert meta['ack'].is_set() and not window.busy and window.live_projection is None
