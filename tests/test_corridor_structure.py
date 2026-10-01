from pathlib import Path

import cv2
import numpy as np
import pytest

from src.registration_core.credible_geometry import corridor_structure_signature, review_geometry, screenshot_evidence


def walls_at(levels, across=(10, 50)):
    walls = np.float32([[10,y,90,y] for y in levels]+[[x,10,x,180] for x in across])
    directions = np.array([0]*len(levels)+[1]*len(across))
    return walls, directions, np.ones(len(walls),bool)


@pytest.mark.parametrize('levels', [(10,110), (10,60,110), (10,12,110), (10,60,112)])
def test_rectangles_repeated_stripes_and_duplicate_edges_are_not_distinctive(levels):
    assert not corridor_structure_signature(*walls_at(levels))['distinctive']


@pytest.mark.parametrize('rotate', [False, True])
def test_unequal_opening_spacing_supplies_additional_corridor_evidence(rotate):
    walls, directions, supported = walls_at((10, 128, 172))
    if rotate:
        walls = walls[:,[1,0,3,2]]; directions = 1-directions
    assert corridor_structure_signature(walls, directions, supported)['distinctive']
    supported[2] = False
    assert not corridor_structure_signature(walls, directions, supported)['distinctive']


def test_repeated_locations_stay_ambiguous_with_corridor_policy_enabled():
    from test_credible_geometry import colored_scene
    template, observation, truth, image = colored_scene()
    for name in ('image','room_mask','corridor_mask','map_mask','edge_mask','stair_mask','distance_transform'):
        value=getattr(template,name);setattr(template,name,np.concatenate((value,value),axis=1))
    other=truth.copy();other[0,2]+=460
    _,details=review_geometry(template,screenshot_evidence(image,observation,reliable=True),[truth,other],
        scale_min=.7,scale_max=1.4,viewport=(300,300),origin=(0,0),allow_corridor_structure=True)
    assert not details['accepted'] and not details['local_accepted']
    assert details['competitive_branches']>=1


def test_template_reported_corridor_corrects_scale_and_displays_all_versions(monkeypatch):
    # This archived behavior specifically requires the template route-rescue path.
    # component_tolerance parity (including its refusal of this input) is checked separately.
    from src import projection_workflow as workflow
    from src.registration.runtime_registration import register_screenshot as baseline_register
    monkeypatch.setattr(workflow, 'register_screenshot', baseline_register)
    from src.app_catalog import MapCatalog
    from src.projection_workflow import process_screenshot, switch_result_version
    from src.registration.temporal_evidence import TemporalEvidence
    root=Path(__file__).resolve().parents[1]
    path=root/'runtime/diagnostics/route_projection/app_runs/20260926_193245_093635_layout_11_2nd/01_screenshot.png'
    if not path.exists():pytest.skip('Local diagnostic is not distributed')
    image=cv2.imdecode(np.fromfile(path,np.uint8),1)
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    versions={};prepared={};assets={}
    catalog.select_version_for_all(profile,'route_zhanshi_fast',versions)
    result=process_screenshot(image,catalog=catalog,profile=profile,model_bundle=None,
        floor='2nd',layout_override='layout_11',route_versions=versions,opacity=.65,
        prepared_cache=prepared,asset_cache=assets)
    assert result.accepted,result.reasons
    reg=result.registration
    assert reg.validation['reliable_geometry']['corridor_structure']['confirmed']
    assert reg.validation['route_corroborated']
    assert reg.validation['edge_median_screen_px']<1
    assert 1.60<reg.screenshot_to_clean[0,0]<1.64
    assert result.overlay_validation['structure_coverage']>.98
    template=prepared[(profile.key,'layout_11','2nd')]
    assert not TemporalEvidence.empty(template).add_registration(reg,template)
    for variant in ('route_clean','route_lhp_solo','route_zhanshi_fast'):
        switched=switch_result_version(result,catalog=catalog,profile=profile,variant_id=variant,opacity=.65,
            asset_cache=assets,prepared_cache=prepared)
        assert switched.accepted,switched.reasons
        assert switched.overlay_validation['projection_scope']=='full'
        assert switched.overlay_validation['evidence_scope']=='local'
