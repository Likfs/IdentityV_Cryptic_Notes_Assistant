from tests.support.template_inputs import install_floor_inputs
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
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
    clock[0]=100.099
    window.scheduled[-1][1]()
    assert not window.requests
    clock[0]=100.151
    window.scheduled[-1][1]()
    assert window.requests==[('project',True)] and not window.projection_delay_pending
    assert window.results.empty()


def test_template_work_skips_before_and_after_floor_checks(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template'
    from src.capture.region import CaptureRegion
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    monkeypatch.setattr(app,'capture_bgr',lambda roi:np.full((roi[3],roi[2],3),100,np.uint8))
    calls=[]
    def process(image,**kw):
        calls.append(kw);return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app,'process_screenshot',process)
    monkeypatch.setattr(app,'save_workflow_result',lambda _:Path('memory-only'))
    app.RouteProjectionApp.request_work(window,'project',auto=False)
    assert len(calls)==1 and calls[0]['template_variant_id']=='route_zhanshi_fast'
    assert not {'floor', 'model_bundle', 'registration_priors'} & calls[0].keys()


def test_template_floor_button_click_invalidates_then_rematches_without_identification(monkeypatch):
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


def test_template_full_ui_result_reference_and_old_backend_setting_migrates(window,monkeypatch):
    original=np.full((40,40,3),120,np.uint8)
    rgba=np.dstack((original,np.full((40,40),255,np.uint8)))
    entry={'saved_scale':1.,'source_sha256':'verified','source_image':'original.png'}
    install_floor_inputs(monkeypatch, entry, original, rgba)
    monkeypatch.setattr(direct,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a, **kw:dict(score=2.,location=(0,0)))
    versions=json.loads(json.dumps(window.settings['route_versions']))
    window.apply_settings({**window.settings,'registration_backend':'template'})
    assert window.route_combo.isEnabled() and window.floor_combo.isEnabled()
    assert window.route_combo.currentData()=='route_zhanshi_fast'
    result=workflow.process_screenshot(original, catalog=window.catalog, profile=window.profile, layout_override=window.locked_layout, opacity=0.3, registration_backend='template')
    shown=[]
    monkeypatch.setattr(window.overlay,'show',lambda *args:shown.append(args))
    generation=window.map_generation
    window.show_overlay_check.setChecked(True)
    window._apply_result('project',True,generation,[0,0,40,40],result,None)
    assert len(shown)==1 and window.live_projection[0] is result
    assert window.reference_context[2]=='2nd'
    assert window.template_panel._pixmap.width() == original.shape[1]
    assert '2 楼' in window.template_panel.caption.text()
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
