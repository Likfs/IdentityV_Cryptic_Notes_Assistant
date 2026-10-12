from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
import pytest
from src.registration.template_matching import direct
from src import projection_workflow as workflow


@pytest.mark.parametrize('origin', [(0,0),(20,0),(0,20)])
@pytest.mark.parametrize('scale', [1.,2.])
def test_original_projection_matches_any_floor_without_clean_coordinates(monkeypatch,origin,scale):
    original=np.zeros((40,40,3),np.uint8)
    original[:20,:20]=(70,90,180);original[:20,20:]=(200,80,60);original[20:,:20]=(90,190,70)
    rgba=np.dstack((original,np.uint8(np.any(original>0,axis=2))*255))
    entry={'saved_scale':scale,'source_sha256':'verified','source_image':'original.png'}
    monkeypatch.setattr(direct,'load_original',lambda *a:(entry,Path('original.png'),{'perf_mode':2},original,rgba))
    monkeypatch.setattr(direct,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a:dict(score=3.,location=origin))
    monkeypatch.setattr(direct.matcher,'find_best_scale',lambda *a:pytest.fail('scale search was called'))
    result=workflow.process_screenshot(np.full((10, 10, 3), 100, np.uint8), catalog=object(), profile=SimpleNamespace(key='map/difficulty'), layout_override='layout_16', opacity=1.0, registration_backend='template')
    x,y=origin
    assert result.accepted and result.floor=='all' and result.variant_id=='route_zhanshi_fast'
    np.testing.assert_array_equal(result.layer[:,:,:3],original[y:y+10,x:x+10])
    assert np.all(result.layer[:,:,3]==255)
    assert result.registration.diagnostics()['clean_to_screen'] is None and result.registration.diagnostics()['screenshot_to_clean'] is None
    np.testing.assert_allclose(result.registration.screenshot_to_original@np.array([2,3,1]),[2*scale+x,3*scale+y,1])
    assert result.template_path is None and result.calibration_path is None
    assert result.diagnostics()['floor_detection'] is None and not result.registration.validation['trusted_history_eligible']
    assert result.registration.diagnostics()['coordinate_space']=='original_source_image'


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


def test_blank_capture_does_not_create_random_overlay(monkeypatch):
    monkeypatch.setattr(direct,'load_original',lambda *a:pytest.fail('empty frame must not search'))
    result,_,layer,_,_=direct.project_original(np.zeros((20,20,3),np.uint8),'map/diff','layout_16',.5)
    assert not result.accepted and layer is None
