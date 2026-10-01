"""Reviewed positive plus independent failure boundaries for the exception."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template
from src.registration.multiscale_structure.runtime import register_screenshot
from src.registration.multiscale_structure.cpu.joint_acceptance import review_single_category
from src.registration_core.utils import imread_unicode

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/registration/multiscale_structure/layout_16_b1_semantic_positive.json'


def test_reviewed_room_is_accepted_without_changing_pose(tmp_path):
    cv2.setNumThreads(4)
    truth=json.loads(FIXTURE.read_text(encoding='utf8'))
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    prepared=prepare_registration_template('layout_16','b1',catalog.clean_template(profile,'layout_16','b1'),tmp_path)
    result=register_screenshot(imread_unicode(FIXTURE.with_suffix('.png')),prepared)
    assert result.accepted and result.validation['visible_acceptance']['basis']=='single_category_geometry_support'
    np.testing.assert_array_equal(result.clean_to_screen,truth['clean_to_screen'])
    assert not result.validation['trusted_history_eligible']
    review=result.validation['visible_acceptance']['visible_review']
    assert review['normal_reasons']==['selected_pose_quality']
    assert review['supplemental_review']['accepted']
    assert not review['checks']['selected_pose_quality']['normal_passed']


@pytest.mark.parametrize('change',[
    'semantic_below_floor','structure_conflict','interior_conflict','low_support',
    'single_axis','large_residual','unstable_visible','unstable_viewport','unstable_holdout',
    'unresolved_competitor','hard_gate_failed','additional_failed_gate','missing_gate','nonfinite',
])
def test_one_good_score_cannot_compensate_other_failures(change):
    truth=json.loads(FIXTURE.read_text(encoding='utf8'))
    review=deepcopy(truth['review']);raster=dict(truth['raster']);checks=review['checks'];q=checks['selected_pose_quality']
    if change=='semantic_below_floor':raster['semantic_precision']=.679
    elif change=='structure_conflict':raster['structure_precision']=.94
    elif change=='interior_conflict':q['interior']=.94
    elif change=='low_support':q['support']=.94
    elif change=='single_axis':q['orientations']=[6,2]
    elif change=='large_residual':q['median_screen_px']=1.51
    elif change=='unstable_visible':checks['visible_perturbation']['p95_screen_px']=4.01
    elif change=='unstable_viewport':checks['full_viewport_bound']['viewport_p95_px']=12.01
    elif change=='unstable_holdout':checks['full_viewport_bound']['holdout_p95_px']=16.01
    elif change=='unresolved_competitor':checks['all_competitors_resolved']['passed']=False
    elif change=='hard_gate_failed':checks['original_hard_gates']['passed']=False
    elif change=='additional_failed_gate':checks['extended_scale_probe']={'passed':False}
    elif change=='missing_gate':del checks['background_and_extent']
    elif change=='nonfinite':q['semantic']=float('nan')
    assert not review_single_category(review,raster)['accepted']
