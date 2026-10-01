"""Production integration and frozen reviewed multiscale_structure regression inputs."""
from dataclasses import replace
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import projection_workflow as workflow
from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template
from src.registration.temporal_evidence import TemporalEvidence
from src.registration.multiscale_structure.runtime import register_screenshot
from src.registration_core.utils import imread_unicode

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def capture(tmp_path_factory):
    cv2.setNumThreads(4);catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    image=imread_unicode(ROOT/'tests/fixtures/registration/game2.png')
    prepared=prepare_registration_template('layout_05','2nd',catalog.clean_template(profile,'layout_05','2nd'),tmp_path_factory.mktemp('visible_acceptance_cache'))
    reg=register_screenshot(image,prepared)
    assert reg.accepted
    return catalog,profile,image,prepared,reg


def test_formal_entry_and_all_route_versions_without_route_veto(capture,monkeypatch):
    catalog,profile,image,prepared,reg=capture
    assert workflow.register_screenshot is register_screenshot
    monkeypatch.setattr(workflow,'validate_route_overlay',lambda *a,**k:pytest.fail('route content cannot veto multiscale_structure'))
    cache={(profile.key,'layout_05','2nd'):prepared};assets={}
    result=workflow.process_screenshot(image,catalog=catalog,profile=profile,model_bundle=None,
        floor='2nd',layout_override='layout_05',route_versions={},opacity=.65,prepared_cache=cache,asset_cache=assets)
    assert result.accepted and result.registration.validation['registration_algorithm']=='multiscale_structure'
    np.testing.assert_array_equal(result.registration.clean_to_screen,reg.clean_to_screen)
    monkeypatch.setattr(workflow,'register_screenshot',lambda *a,**k:pytest.fail('version switch must reuse current pose'))
    for variant in ('route_clean','route_lhp_solo','route_zhanshi_fast'):
        switched=workflow.switch_result_version(result,catalog=catalog,profile=profile,variant_id=variant,
            opacity=.65,asset_cache=assets,prepared_cache=cache)
        assert switched.accepted and not switched.overlay_validation['enabled']
        assert np.count_nonzero(switched.layer[:,:,3]) and switched.registration is result.registration


def test_rejected_pose_cannot_be_rescued(capture):
    catalog,profile,_,_,reg=capture
    rejected=replace(reg,accepted=False,rejection_reasons=['position_ambiguous'])
    assert not workflow._route_can_corroborate(rejected)
    check=workflow.validate_display_asset(rejected,None,SimpleNamespace(variant='route_lhp_solo'),catalog=catalog,profile=profile,asset_cache={})
    assert not check['accepted']


def test_history_is_neither_read_nor_written(capture):
    _,_,image,prepared,reg=capture
    memory=TemporalEvidence.empty(prepared)
    result=register_screenshot(image,prepared,np.float32([[10,0,9999],[0,10,9999]]),memory,object())
    np.testing.assert_array_equal(result.clean_to_screen,reg.clean_to_screen)
    assert not memory.add_registration(result,prepared,source='visible_acceptance-test') and not memory.updates
    assert not result.validation['trusted_history_eligible']


def test_blank_frame_has_no_projection(capture):
    _,_,image,prepared,_=capture
    reg=register_screenshot(np.zeros_like(image),prepared)
    assert not reg.accepted and reg.clean_to_screen is None


def test_formal_import_and_execution_do_not_require_experiment_modules():
    code='''
import sys,importlib.abc
class BlockExperiments(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='tmp' or fullname.startswith('tmp.'):
   raise RuntimeError('Formal app imported temporary code: '+fullname)
sys.meta_path.insert(0,BlockExperiments())
from src import projection_workflow as w
from src.registration.multiscale_structure.runtime import register_screenshot
assert w.register_screenshot is register_screenshot
from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template
from pathlib import Path
import numpy as np,cv2
c=MapCatalog();p=c.profile('a_bringer_of_doom','nightmare')
t=prepare_registration_template('layout_05','2nd',c.clean_template(p,'layout_05','2nd'),Path('workspace/cache/visible_acceptance_import_test'))
r=register_screenshot(np.zeros((400,500,3),np.uint8),t)
assert not r.accepted and r.validation['registration_algorithm']=='multiscale_structure'
'''
    result=subprocess.run([sys.executable,'-X','utf8','-B','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('name',['layout_13_1st_narrow','layout_15_b1_narrow'])
def test_partial_geometry_counterexamples_remain_rejected(name,tmp_path):
    root=ROOT/'tests/fixtures/registration/multiscale_structure'
    meta=json.loads((root/f'{name}.json').read_text(encoding='utf8'));image=imread_unicode(root/f'{name}.png')
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    prepared=prepare_registration_template(meta['layout'],meta['floor'],catalog.clean_template(profile,meta['layout'],meta['floor']),tmp_path)
    reg=register_screenshot(image,prepared)
    assert not reg.accepted
    assert not reg.validation['visible_acceptance']['visible_review']['checks']['partial_fit_redundancy']['passed']
