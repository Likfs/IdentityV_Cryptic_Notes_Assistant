from tests.support.template_inputs import install_floor_inputs
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
    install_floor_inputs(monkeypatch, entry, original, rgba, variants=('route_zhanshi_fast', 'clean'))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a, **kw:dict(score=3.,location=origin))
    monkeypatch.setattr(direct.matcher,'find_best_scale',lambda *a:pytest.fail('scale search was called'))
    profile = SimpleNamespace(key='map/difficulty', template_versions=('route_zhanshi_fast',),
                              version_names={'route_zhanshi_fast': '测试路线'})
    result=workflow.process_screenshot(np.full((10, 10, 3), 100, np.uint8), catalog=object(), profile=profile, layout_override='layout_16', opacity=1.0, registration_backend='template')
    x,y=origin
    assert result.accepted and result.floor=='2nd' and result.variant_id=='route_zhanshi_fast'
    np.testing.assert_array_equal(result.layer[:,:,:3],original[y:y+10,x:x+10])
    assert np.all(result.layer[:,:,3]==255)
    assert result.registration.diagnostics()['clean_to_screen'] is None and result.registration.diagnostics()['screenshot_to_clean'] is None
    np.testing.assert_allclose(result.registration.screenshot_to_original@np.array([2,3,1]),[2*scale+x,3*scale+y,1])
    assert result.template_path is None and result.calibration_path is None
    assert result.diagnostics()['floor_detection'] is None and not result.registration.validation['trusted_history_eligible']
    assert result.registration.diagnostics()['coordinate_space']=='single_floor_source_image'




def test_blank_capture_does_not_create_random_overlay(monkeypatch):
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a, **kw:pytest.fail('empty frame must not search'))
    result,_,layer,_,_=direct.project_original(np.zeros((20,20,3),np.uint8),'map/diff','layout_16',.5)
    assert not result.accepted and layer is None


@pytest.mark.parametrize('units', [7, 11, 24, 26])
@pytest.mark.parametrize('client_size,factor', [((1920, 1080), 4/3),
                                            ((2560, 1334), 1920000/1779556)])
def test_tracked_zoom_adjusts_the_single_matching_scale_without_changing_saved_scale(monkeypatch, units, client_size, factor):
    original = np.full((80, 80, 3), 120, np.uint8)
    rgba = np.dstack((original, np.full((80, 80), 255, np.uint8)))
    entry = dict(saved_scale=1.5, source_sha256='verified', source_image='original.png')
    install_floor_inputs(monkeypatch, entry, original, rgba, variants=('route_zhanshi_fast', 'clean'))
    scales = []
    def match(_screen, _path, scale, **kwargs):
        scales.append((scale, kwargs['ds']))
        return dict(score=3., location=(4, 8))
    monkeypatch.setattr(direct.matcher, '_match_template', match)
    monkeypatch.setattr(direct.matcher, 'find_best_scale', lambda *a:pytest.fail('No scale search is permitted'))
    profile = SimpleNamespace(key='map/difficulty', template_versions=('route_zhanshi_fast',),
                              version_names={'route_zhanshi_fast':'测试路线'})
    result = workflow.process_screenshot(original[:20, :20], catalog=object(), profile=profile,
        layout_override='layout_16', opacity=.5, template_variant_id='route_zhanshi_fast',
        capture_client_size=client_size, map_zoom_units=units)
    assert result.accepted, result.reasons
    assert scales and all(scale == pytest.approx(1.5 * factor / (units / 7)) for scale, _ds in scales)
    # Equal floor scores still trigger the existing 25% -> 50% review at minimum zoom.
    assert {ds for _scale, ds in scales} == ({.25, .5} if units == 7 else {.5})
    assert result.registration.validation['template']['saved_scale'] == 1.5
    assert result.registration.validation['map_zoom_units'] == units
    assert result.registration.screenshot_to_original[0, 0] == pytest.approx(1.5 * factor / (units / 7))


def test_negative_score_still_projects_candidate_with_explicit_warning(monkeypatch):
    original = np.full((40, 40, 3), 120, np.uint8)
    rgba = np.dstack((original, np.full((40, 40), 255, np.uint8)))
    install_floor_inputs(monkeypatch, dict(saved_scale=1., source_sha256='verified',
        source_image='original.png'), original, rgba, variants=('route_zhanshi_fast', 'clean'))
    monkeypatch.setattr(direct.matcher, '_match_template', lambda *a, **kw:
        dict(score=-3., location=(4, 8), search={'boundary_edges': []}))
    result, _, layer, _, _ = direct.project_original(original[:10, :10], 'map/difficulty', 'layout_16', .5,
                                                    variant_id='route_zhanshi_fast')
    assert result.accepted and layer is not None
    assert any('分数为负' in warning for warning in result.validation['warnings'])
    np.testing.assert_array_equal(result.screenshot_to_original, [[1, 0, 4], [0, 1, 8], [0, 0, 1]])
