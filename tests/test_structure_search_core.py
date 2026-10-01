from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import projection_workflow as workflow
from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template
from src.registration.temporal_evidence import TemporalEvidence
from src.registration.multiscale_structure.cpu.search_core.runtime import register_screenshot
from src.registration.multiscale_structure.cpu.search_core.template import prepared_adapter

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def capture(tmp_path_factory):
    cv2.setNumThreads(4);catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    image=cv2.imdecode(np.fromfile(ROOT/'tests/fixtures/registration/game2.png',np.uint8),1)
    prepared=prepare_registration_template('layout_05','2nd',catalog.clean_template(profile,'layout_05','2nd'),tmp_path_factory.mktemp('component_search_cache'))
    result=register_screenshot(image,prepared)
    assert result.accepted
    return catalog,profile,image,prepared,result


def test_historical_component_search_workflow_without_route_veto_and_switches_versions(capture,monkeypatch):
    catalog,profile,image,prepared,reg=capture
    monkeypatch.setattr(workflow,'register_screenshot',register_screenshot)
    assert workflow.register_screenshot is register_screenshot
    monkeypatch.setattr(workflow,'validate_route_overlay',lambda *a,**kw:pytest.fail('route image must not veto component_tolerance'))
    cache={(profile.key,'layout_05','2nd'):prepared};assets={}
    result=workflow.process_screenshot(image,catalog=catalog,profile=profile,model_bundle=None,
        floor='2nd',layout_override='layout_05',route_versions={},opacity=.65,prepared_cache=cache,asset_cache=assets)
    assert result.accepted and result.registration.validation['registration_algorithm']=='component_tolerance'
    np.testing.assert_array_equal(result.registration.clean_to_screen,reg.clean_to_screen)
    monkeypatch.setattr(workflow,'register_screenshot',lambda *a,**kw:pytest.fail('version switch must reuse pose'))
    for version in ('route_clean','route_lhp_solo','route_zhanshi_fast'):
        changed=workflow.switch_result_version(result,catalog=catalog,profile=profile,variant_id=version,
            opacity=.65,asset_cache=assets,prepared_cache=cache)
        assert changed.accepted and not changed.overlay_validation['enabled']
        assert changed.layer is not None and changed.registration is result.registration


def test_component_search_rejections_cannot_be_rescued_by_route_content(capture):
    catalog,profile,_,_,reg=capture
    rejected=replace(reg,accepted=False,rejection_reasons=['position_ambiguous'])
    assert not workflow._route_can_corroborate(rejected)
    check=workflow.validate_display_asset(rejected,None,SimpleNamespace(variant='route_lhp_solo'),catalog=catalog,profile=profile,asset_cache={})
    assert not check['accepted']


def test_component_search_cannot_read_or_write_trusted_history(capture):
    _,_,image,prepared,reg=capture
    memory=TemporalEvidence.empty(prepared)
    assert not memory.add_registration(reg,prepared,source='component_search-accepted')
    assert not memory.updates
    wrong=np.float32([[10,0,9999],[0,10,9999]])
    repeated=register_screenshot(image,prepared,wrong,memory,object())
    assert repeated.accepted==reg.accepted
    np.testing.assert_array_equal(repeated.clean_to_screen,reg.clean_to_screen)
    assert not memory.updates


def test_template_adapter_is_bound_to_prepared_object(capture):
    _,_,_,prepared,_=capture
    first,lock=prepared_adapter(prepared)
    second=replace(prepared,layout='other_layout')
    other,other_lock=prepared_adapter(second)
    assert first is prepared_adapter(prepared)[0]
    assert other is not first and other.layout=='other_layout' and other_lock is not lock


def test_blank_frame_has_no_pose_or_route_rescue(capture):
    _,_,image,prepared,_=capture
    result=register_screenshot(np.zeros_like(image),prepared)
    assert not result.accepted and result.clean_to_screen is None
    assert not workflow._route_can_corroborate(result)


@pytest.mark.parametrize('source,layout,floor,accepted',[
    ('20260926_193245_093635_layout_11_2nd','layout_11','2nd',False),
    ('20260926_184200_218934_layout_10_b1','layout_10','b1',True)])
def test_component_search_retains_reviewed_behavior_on_legacy_workflow_cases(source,layout,floor,accepted,tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'register_screenshot',register_screenshot)
    path=ROOT/'runtime/diagnostics/route_projection/app_runs'/source/'01_screenshot.png'
    if not path.exists():pytest.skip('Local screenshot is not distributed')
    image=cv2.imdecode(np.fromfile(path,np.uint8),1);catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    result=workflow.process_screenshot(image,catalog=catalog,profile=profile,model_bundle=None,
        floor=floor,layout_override=layout,route_versions={},opacity=.65,prepared_cache={},asset_cache={})
    assert result.accepted==accepted
    assert result.registration.validation['registration_algorithm']=='component_tolerance'
    if not accepted:
        assert 'structure_not_distinctive' in result.registration.rejection_reasons
        assert result.layer is None
