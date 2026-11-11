from tests.support.floor_ui import projection_frame, template_control
from tests.support.template_inputs import install_floor_inputs
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from src.registration.template_matching import direct
from src import projection_workflow as workflow
from src import route_projection_app as app
from tests.support.app_fakes import FakeApp, InlineThread
from tests.support.qt_wait import wait_for


def test_accepted_candidate_warning_is_visible_without_rejecting_projection(window):
    result = SimpleNamespace(accepted=True, layer=np.zeros((4, 4, 4), np.uint8),
        selected_layout=window.locked_layout, floor='1st', profile_key=window.profile.key,
        variant_id=window.route_combo.currentData(),
        registration=SimpleNamespace(validation={'warnings': ['匹配位置仍在搜索边界，候选可能错位，请核对位置']}))
    window._finish_map_feedback(window.map_generation, window.settings['capture_roi'], result)
    assert window.toast.title.text() == '模板候选可能错位'
    assert '可能错位' in window.status_var.get()
    assert result.accepted and window.settings['reference_floor'] == '1st'


def test_template_open_waits_then_projects_without_floor_capture(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template';clock=[100.01]
    window.settings.pop('map_capture_delay_ms')  # Exercise the default, not the fixture's custom delay.
    monkeypatch.setattr(app.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    app.RouteProjectionApp._project_after_delay(window,7,'session')
    assert not window.requests
    clock[0]=100.029
    window.scheduled[-1][1]()
    assert not window.requests
    clock[0]=100.031
    window.scheduled[-1][1]()
    assert window.requests==[('project',True)] and not window.projection_delay_pending
    assert window.results.empty()


def test_template_work_detects_floor_in_the_same_capture_and_keeps_client_size(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template'
    window.settings['template_route_versions'][window.profile.key] = '测试路线'
    from src.capture.region import CaptureRegion
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    client = [CaptureRegion(0,0,1920,1080)]
    # This is a simulated window resize, not a foreground-window change.
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:42)
    monkeypatch.setattr(app,'foreground_game_client',lambda _:client[0])
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    def capture(roi):
        client[0] = CaptureRegion(0,0,3072,1920)  # A later resize must not change this capture's scale.
        return np.full((roi[3],roi[2],3),100,np.uint8)
    monkeypatch.setattr(app,'capture_bgr',capture)
    calls=[]
    def process(image,**kw):
        calls.append(kw);return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app,'process_screenshot',process)
    monkeypatch.setattr(app,'save_workflow_result',lambda _:Path('memory-only'))
    app.RouteProjectionApp.request_work(window,'project',auto=False)
    assert len(calls)==1 and calls[0]['template_variant_id']=='测试路线'
    assert calls[0]['capture_client_size'] == (1920,1080)
    assert calls[0]['floor_detection'].floor is None  # Blank fake UI is explicit, never guessed.
    assert not {'floor', 'model_bundle', 'registration_priors'} & calls[0].keys()


def test_floor_ui_work_still_rejects_a_foreground_window_change(monkeypatch):
    from src.capture.region import CaptureRegion
    window = FakeApp()
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    monkeypatch.setattr(app, 'foreground_game_client', lambda _:CaptureRegion(0, 0, 1920, 1080))
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _:None)
    handle = [42]
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:handle[0])
    def capture(roi):
        handle[0] = 99
        return np.full((roi[3], roi[2], 3), 100, np.uint8)
    monkeypatch.setattr(app, 'capture_bgr', capture)
    calls = []
    monkeypatch.setattr(app, 'process_screenshot', lambda *a, **kw:calls.append(kw))
    app.RouteProjectionApp.request_work(window, 'project', auto=False)
    assert not calls
    _, _, event = window.results.get_nowait()
    assert event[5] is None and '截图期间游戏窗口发生变化' in event[6]


@pytest.mark.parametrize('outcome', ['accepted', 'rejected', 'error', 'single_floor'])
def test_capture_and_floor_ui_timings_are_separate(monkeypatch, outcome):
    from src.capture.region import CaptureRegion
    from src.floor_indicator import FloorDetection
    from src import route_projection_core as core
    window = FakeApp()
    if outcome == 'single_floor':
        window.profile.floors_for_layout = lambda _:('1st',)
    clock = [100.]
    # Replace the module binding, never the shared time module used by Qt tests.
    monkeypatch.setattr(app, 'time', SimpleNamespace(**{
        **vars(app.time), 'perf_counter':lambda:clock[0]}))
    monkeypatch.setattr(core, 'perf_counter', lambda:clock[0])
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:42)
    monkeypatch.setattr(app, 'foreground_game_client', lambda _:CaptureRegion(0, 0, 1920, 1080))
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _:None)
    def capture(roi):
        clock[0] += .020
        return np.full((roi[3], roi[2], 3), 30, np.uint8)
    def detect(_):
        clock[0] += .180
        if outcome == 'error':
            raise ValueError('invalid template')
        return FloorDetection('1st' if outcome == 'accepted' else None)
    monkeypatch.setattr(app, 'capture_bgr', capture)
    monkeypatch.setattr(core, 'detect_floor', detect)
    monkeypatch.setattr(app, 'process_screenshot', lambda *a, **kw:SimpleNamespace(timings_ms={}))
    saved = []
    monkeypatch.setattr(app, 'save_workflow_result', saved.append)
    app.RouteProjectionApp.request_work(window, 'project', auto=False)
    assert len(saved) == 1
    timing = saved[0].timings_ms
    assert timing['capture'] == pytest.approx(20.)
    assert timing['floor_ui'] == pytest.approx(0. if outcome == 'single_floor' else 180.)
    assert timing['frame_split'] == pytest.approx(0.)
    assert timing['capture_pipeline'] == pytest.approx(20.+timing['floor_ui'])


def test_single_floor_worker_uses_geometry_and_retains_zoom_when_buttons_are_hidden(monkeypatch):
    from src.capture.region import CaptureRegion
    from src.game_geometry import map_viewport, map_center, map_match_region
    window = FakeApp()
    window.profile.floors_for_layout = lambda _:('1st',)
    window._motion = SimpleNamespace(registration_zoom=lambda:(13, 42))
    # Historical personal ROI remains stored but must not drive GUI capture.
    window.settings['capture_roi'] = [1, 2, 30, 40]
    client = CaptureRegion(595, 281, 1004, 755)
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:42)
    monkeypatch.setattr(app, 'foreground_game_client', lambda _:client)
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _:None)
    monkeypatch.setattr(app, 'capture_bgr', lambda roi:np.full((roi[3], roi[2], 3), 30, np.uint8))
    calls, saved = [], []
    def process(image, **kwargs):
        calls.append((image, kwargs))
        return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app, 'process_screenshot', process)
    monkeypatch.setattr(app, 'save_workflow_result', lambda result:saved.append(result))
    app.RouteProjectionApp.request_work(window, 'project', auto=False)
    assert len(calls) == 1 and len(saved) == 1
    image, options = calls[0]
    view = map_viewport(client)
    assert image.shape == (view.height, view.width, 3)
    assert options['map_zoom_units'] == 13
    assert options['capture_client_size'] == (1004, 755)
    match = map_match_region(client, view)
    assert options['matching_roi'] == (match.left-view.left, match.top-view.top, match.width, match.height)
    detection = options['floor_detection']
    assert detection.floor == '1st' and detection.buttons == ()
    assert detection.selection_source == 'single_floor_layout'
    assert saved[0].capture_context['map_roi_screen'] == view.as_list()
    assert saved[0].capture_context['map_zoom_center_screen'] == list(map_center(client))


@pytest.mark.parametrize('capture_hwnd,expected', [(42, 13), (99, 7)])
@pytest.mark.parametrize('client_size', [(1920, 1080), (2560, 1334), (3440, 1440)])
def test_project_uses_remembered_zoom_only_for_the_same_game_window(monkeypatch, capture_hwnd, expected, client_size):
    from tests.support.floor_ui import projection_frame, template_control
    window = FakeApp()
    window._motion = SimpleNamespace(registration_zoom=lambda:(13, 42))
    from src.capture.region import CaptureRegion
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    monkeypatch.setattr(app, 'foreground_game_client', lambda _:CaptureRegion(0, 0, *client_size))
    monkeypatch.setattr(app.ctypes.windll.user32, 'GetForegroundWindow', lambda:capture_hwnd)
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _:None)
    client = CaptureRegion(0, 0, *client_size)
    map_region = app.map_viewport(client)
    map_image = np.full((map_region.height, map_region.width, 3), 100, np.uint8)
    monkeypatch.setattr(app, 'capture_bgr', lambda region:projection_frame(
        region, map_image, map_region.as_list()[:2], client=client, control=template_control(4)))
    calls = []
    def process(image, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app, 'process_screenshot', process)
    monkeypatch.setattr(app, 'save_workflow_result', lambda _:Path('memory-only'))
    app.RouteProjectionApp.request_work(window, 'project', auto=False)
    assert len(calls) == 1 and calls[0]['map_zoom_units'] == expected
    assert calls[0]['floor_detection'].floor == '2nd'


def test_template_floor_button_click_invalidates_then_requests_ui_recheck(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template';down=[True]
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    monkeypatch.setattr(app,'floor_control_mouse_down',lambda *a:down[0])
    app.RouteProjectionApp._handle_template_floor_click(window)
    assert window.map_generation==8 and window.hidden==1 and not window.requests
    app.RouteProjectionApp._handle_template_floor_click(window)
    assert window.map_generation==8
    down[0]=False
    app.RouteProjectionApp._handle_template_floor_click(window)
    assert window.scheduled[-1][0]==35
    window.map_generation+=1  # Closing the map cancels the delayed replacement.
    window.scheduled[-1][1]()
    assert not window.requests


@pytest.mark.parametrize('matching_roi', [None, (8,0,24,40)])
def test_template_full_ui_result_reference_and_old_backend_setting_migrates(window,monkeypatch,matching_roi):
    original=np.full((40,40,3),120,np.uint8)
    rgba=np.dstack((original,np.full((40,40),255,np.uint8)))
    entry={'saved_scale':1.,'source_sha256':'verified','source_image':'original.png'}
    install_floor_inputs(monkeypatch, entry, original, rgba)
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a, **kw:dict(score=2.,location=(0,0)))
    versions=json.loads(json.dumps(window.settings['route_versions']))
    window.apply_settings({**window.settings,'registration_backend':'template'})
    assert window.route_combo.isEnabled() and window.floor_combo.isEnabled()
    assert window.route_combo.currentData()=='route_zhanshi_fast'
    result=workflow.process_screenshot(original, catalog=window.catalog, profile=window.profile, layout_override=window.locked_layout, opacity=0.3, registration_backend='template', matching_roi=matching_roi)
    shown=[]
    monkeypatch.setattr(window.overlay,'show',lambda *args:shown.append(args))
    generation=window.map_generation
    window.settings['show_projection'] = True
    window._apply_result('project',True,generation,[0,0,40,40],result,None)
    assert len(shown)==1 and window.live_projection[0] is result
    assert shown[0][0].shape[:2] == original.shape[:2]
    assert window.reference_context[2]=='2nd'
    np.testing.assert_array_equal(window.last_result.registration.source_image, original)
    assert window.capture_panel._pixmap.width() == (original.shape[1] if matching_roi is None else matching_roi[2])
    assert window.last_result.floor == '2nd'
    assert 'floor_ms' not in window.status_rows
    references=[]
    monkeypatch.setattr(window.reference_overlay,'show_map',lambda *args:references.append(args))
    window.reference_visible=True;window._refresh_reference_map()
    wait_for(lambda: bool(references))
    assert references and '2 楼' in references[-1][2]
    window.apply_settings({**window.settings,'registration_backend':'cpu'})
    assert window.route_combo.isEnabled() and window.floor_combo.isEnabled()
    assert window.settings["registration_backend"] == "template"
    assert window.settings['route_versions']==versions and window.live_projection is None
    window._apply_result('project',True,generation,[0,0,40,40],result,None)
    assert len(shown)==1  # Late 定尺度模板匹配 result cannot restore the overlay after backend change.


def test_template_work_captures_map_and_ui_once_and_passes_selected_floor(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template'
    window.settings['template_route_versions'][window.profile.key] = '测试路线'
    from src.capture.region import CaptureRegion
    from tests.support.floor_ui import projection_frame, template_control
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    monkeypatch.setattr(app.ctypes.windll.user32,'GetForegroundWindow',lambda:42)
    client=CaptureRegion(0,0,2560,1440)
    view=app.map_viewport(client)
    captures=[]
    def capture(roi):
        captures.append(roi)
        return projection_frame(roi, np.full((view.height,view.width,3),100,np.uint8), (view.left,view.top))
    monkeypatch.setattr(app,'capture_bgr',capture)
    calls=[]
    def process(image,**kw):
        assert image.shape == (view.height,view.width,3)
        x,y,w,h=kw['matching_roi']
        assert np.all(image[y:y+h,x:x+w] == 100)
        calls.append(kw);return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app,'process_screenshot',process)
    monkeypatch.setattr(app,'save_workflow_result',lambda _:Path('memory-only'))
    app.RouteProjectionApp.request_work(window,'project',auto=False)
    assert len(calls)==1 and calls[0]['template_variant_id']=='测试路线'
    assert calls[0]['floor_detection'].floor == '1st'
    assert captures == [app.projection_capture_regions(client,view)[0].as_list()]
    assert not {'floor', 'model_bundle', 'registration_priors'} & calls[0].keys()


@pytest.mark.parametrize('control,expected', [(np.zeros((510,140,3),np.uint8), 'unknown'),
                                           (template_control(3), 'b4')])
def test_worker_rejects_unknown_or_missing_floor_without_projection(monkeypatch, control, expected):
    window = FakeApp()
    window.profile.floors_for_layout = lambda _: ('2nd', '1st', 'b1')
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    from src.capture.region import CaptureRegion
    from tests.support.floor_ui import projection_frame, template_control
    monkeypatch.setattr(app, 'foreground_game_client', lambda _: CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app, 'ensure_region_on_screen', lambda _: None)
    monkeypatch.setattr(app.ctypes.windll.user32,'GetForegroundWindow',lambda:42)
    client=CaptureRegion(0,0,2560,1440)
    view=app.map_viewport(client)
    monkeypatch.setattr(app, 'capture_bgr', lambda roi: projection_frame(
        roi, np.full((view.height,view.width,3),100,np.uint8), (view.left,view.top), control=control))
    monkeypatch.setattr(app, 'save_workflow_result', lambda _: Path('memory-only'))
    monkeypatch.setattr(direct, 'project_original', lambda *a, **kw: pytest.fail('unexpected matching'))
    app.RouteProjectionApp.request_work(window, 'project', auto=False)
    event = next(item[2] for item in list(window.results.queue) if item[0] == 'work_result')
    result, error = event[-2:]
    assert error is None and result.floor == expected
    assert not result.accepted and result.layer is None and result.reasons


def test_late_ui_result_cannot_replace_current_button_regions(window, monkeypatch):
    monkeypatch.setattr(app, 'floor_control_mouse_down', lambda *a: False)
    current = [[2200,315,88,115], [2200,430,88,115], [2200,545,88,115]]
    window._floor_buttons = current
    result = SimpleNamespace(profile_key=window.profile.key, selected_layout=window.locked_layout,
                             floor_detection={'buttons': [[2190,205,88,115]]})
    import threading
    import time
    meta = dict(posted_at=time.perf_counter(), requested_at=time.perf_counter(),
                session=window.session_id, ack=threading.Event())
    window._receive_work_event(('ready', ('work','project',True,window.map_generation-1,
                                         [0,0,4,4],result,None),meta))
    assert window._floor_buttons == current and meta['ack'].is_set()
