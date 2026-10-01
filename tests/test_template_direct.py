from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
import pytest
from src.registration.template_matching import direct
from src import projection_workflow as workflow
from src import route_projection_app as app
from test_immediate_map_probe import FakeApp, InlineThread
from test_reference_map_ui import window


@pytest.mark.parametrize('origin', [(0,0),(20,0),(0,20)])
@pytest.mark.parametrize('scale', [1.,2.])
def test_original_projection_matches_any_floor_without_clean_coordinates(monkeypatch,origin,scale):
    original=np.zeros((40,40,3),np.uint8)
    original[:20,:20]=(70,90,180);original[:20,20:]=(200,80,60);original[20:,:20]=(90,190,70)
    rgba=np.dstack((original,np.uint8(np.any(original>0,axis=2))*255))
    entry={'saved_scale':scale,'source_sha256':'verified','source_image':'original.png'}
    monkeypatch.setattr(direct,'load_original',lambda *a:(entry,Path('original.png'),{'perf_mode':2},original,rgba))
    monkeypatch.setattr(direct.runtime,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a:dict(score=3.,location=origin))
    monkeypatch.setattr(workflow,'prepare_registration_template',lambda *a,**k:pytest.fail('clean template was loaded'))
    monkeypatch.setattr(workflow,'register_screenshot',lambda *a,**k:pytest.fail('formal dispatcher was called'))
    monkeypatch.setattr(direct.matcher,'find_best_scale',lambda *a:pytest.fail('scale search was called'))
    result=workflow.process_screenshot(np.full((10,10,3),100,np.uint8),catalog=object(),
        profile=SimpleNamespace(key='map/difficulty'),model_bundle=None,floor='unknown',layout_override='layout_16',
        route_versions={},opacity=1.,prepared_cache={},asset_cache={},registration_backend='template')
    x,y=origin
    assert result.accepted and result.floor=='all' and result.variant_id=='full_original'
    np.testing.assert_array_equal(result.layer[:,:,:3],original[y:y+10,x:x+10])
    assert np.all(result.layer[:,:,3]==255)
    assert result.registration.clean_to_screen is None and result.registration.screenshot_to_clean is None
    np.testing.assert_allclose(result.registration.screenshot_to_original@np.array([2,3,1]),[2*scale+x,3*scale+y,1])
    assert result.template_path is None and result.calibration_path is None
    assert result.floor_detection is None and not result.registration.validation['trusted_history_eligible']
    assert result.registration.diagnostics()['coordinate_space']=='original_source_image'
    with pytest.raises(ValueError,match='三层完整原图'):
        workflow.switch_result_version(result,catalog=object(),profile=SimpleNamespace(key='map/difficulty'),
            variant_id='route_lhp_solo',opacity=1,asset_cache={},prepared_cache={})


def test_source_lookup_has_no_clean_or_floor_calibration_dependency(tmp_path,monkeypatch):
    monkeypatch.setattr(direct,'PROJECT_ROOT',tmp_path)
    source=tmp_path/'original.png'
    cv2.imencode('.png',np.full((32,32,3),90,np.uint8))[1].tofile(str(source))
    entry=dict(layout='layout_16',source_image='original.png',
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),saved_scale=1.4)
    (tmp_path/'configs').mkdir()
    config=tmp_path/'configs/template_matching.json'
    config.write_text(json.dumps({'entries':{'assets/maps/test/nightmare/clean/layout_16/missing.png':entry},'perf_mode':2}))
    assert direct.load_original('test/nightmare','layout_16')[1]==source
    source.write_bytes(b'replaced')
    with pytest.raises(ValueError,match='资源已改变'):
        direct.load_original('test/nightmare','layout_16')


def test_template_open_waits_then_projects_without_floor_capture(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template';clock=[100.01]
    window.settings.pop('map_capture_delay_ms')  # Exercise the default, not the fixture's custom delay.
    monkeypatch.setattr(app.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**k:pytest.fail('floor capture in 定尺度模板匹配'))
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True,'session')
    assert not window.requests
    clock[0]=100.099
    window.scheduled[-1][1]()
    assert not window.requests
    clock[0]=100.101
    window.scheduled[-1][1]()
    assert window.requests==[('project',True)] and not window.map_probe_pending
    assert not window.floor_monitor_active and window.opening_floor_sample is None
    assert window.results.empty()


def test_template_work_skips_before_and_after_floor_checks(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template'
    from src.capture.region import CaptureRegion
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**k:pytest.fail('定尺度模板匹配 should never read floor'))
    monkeypatch.setattr(app,'capture_bgr',lambda roi:np.full((roi[3],roi[2],3),100,np.uint8))
    calls=[]
    def process(image,**kw):
        calls.append(kw);return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app,'process_screenshot',process)
    monkeypatch.setattr(app,'save_workflow_result',lambda _:Path('memory-only'))
    app.RouteProjectionApp.request_work(window,'project',auto=False)
    assert len(calls)==1 and calls[0]['floor']=='all' and calls[0]['registration_backend']=='template'
    assert not window.floor_monitor_active


def test_template_floor_button_click_invalidates_then_rematches_without_identification(monkeypatch):
    window=FakeApp();window.settings['registration_backend']='template';down=[True]
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    monkeypatch.setattr(app,'floor_control_mouse_down',lambda *a:down[0])
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**k:pytest.fail('floor recognition'))
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


def test_blank_capture_does_not_create_random_overlay(monkeypatch):
    monkeypatch.setattr(direct,'load_original',lambda *a:pytest.fail('empty frame must not search'))
    result,_,layer,_,_=direct.project_original(np.zeros((20,20,3),np.uint8),'map/diff','layout_16',.5)
    assert not result.accepted and layer is None


def test_template_full_ui_result_reference_and_switch_back_to_cpu(window,monkeypatch):
    original=np.full((40,40,3),120,np.uint8)
    rgba=np.dstack((original,np.full((40,40),255,np.uint8)))
    entry={'saved_scale':1.,'source_sha256':'verified','source_image':'original.png'}
    monkeypatch.setattr(direct,'load_original',lambda *a:(entry,Path('original.png'),{'perf_mode':2},original,rgba))
    monkeypatch.setattr(direct.runtime,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a:dict(score=2.,location=(0,0)))
    monkeypatch.setattr(window,'_start_registration_backend',lambda:None)
    versions=json.loads(json.dumps(window.settings['route_versions']))
    window.apply_settings({**window.settings,'registration_backend':'template'})
    assert not window.route_combo.isEnabled() and not window.floor_combo.isEnabled()
    assert window.route_combo.currentData()=='full_original'
    result=workflow.process_screenshot(original,catalog=window.catalog,profile=window.profile,
        model_bundle=None,floor='unknown',layout_override=window.locked_layout,route_versions=versions,
        opacity=.3,prepared_cache={},asset_cache={},registration_backend='template')
    shown=[]
    monkeypatch.setattr(window.overlay,'show',lambda *args:shown.append(args))
    generation=window.map_generation
    window.show_overlay_check.setChecked(True)
    window._apply_result('project',True,generation,[0,0,40,40],result,None)
    assert len(shown)==1 and window.live_projection[0] is result
    assert window.reference_context[2]=='all'
    assert '已跳过' in window.status_rows['floor_ms'].text()
    references=[]
    monkeypatch.setattr(window.reference_overlay,'show_map',lambda *args:references.append(args))
    window.reference_override=True;window._refresh_reference_map()
    assert references and '三层完整原图' in references[-1][2]
    window.apply_settings({**window.settings,'registration_backend':'cpu'})
    assert window.route_combo.isEnabled() and window.floor_combo.isEnabled()
    assert window.settings['route_versions']==versions and window.live_projection is None
    window._apply_result('project',True,generation,[0,0,40,40],result,None)
    assert len(shown)==1  # Late 定尺度模板匹配 result cannot restore the overlay after backend change.
