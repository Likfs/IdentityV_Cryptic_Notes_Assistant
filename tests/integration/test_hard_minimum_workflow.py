"""New minimum-zoom captures after confirmation must produce independent poses."""
import numpy as np
import pytest
from src.app_catalog import MapCatalog
from src.layout_retrieval import create_layout_retriever
from src.projection_workflow import process_screenshot
from tests.support.replay import read_cases, replay_image


@pytest.fixture
def minimum_retriever(hard_replay_package):
    profile=MapCatalog().profile('a_bringer_of_doom','hard')
    retriever=create_layout_retriever('pure_structure',profile.clean_root,floors=profile.floors,workers=2)
    yield retriever
    retriever.close()


@pytest.mark.parametrize('pair', [(1,2),(3,4)])
def test_minimum_captures_agree_without_transferring_recognition_pose(minimum_retriever,pair,monkeypatch):
    from src.registration.template_matching import direct,matcher
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','hard')
    inputs=[next(c for c in read_cases() if c['id']==f'hard_minimum_{number:02d}') for number in pair]
    results=[process_screenshot(replay_image(case),catalog=catalog,profile=profile,opacity=.5,
        recognize_only=True,layout_recognition_backend='pure_structure',layout_retriever=minimum_retriever)
        for case in inputs]
    for case,result in zip(inputs,results):
        assert result.decision['best_label']==case['expected_top1']
        assert result.accepted
        assert not result.decision['accepted'] and result.decision['layout'] is None
        assert result.registration is None and result.layer is None
    assert results[0].decision['best_label']==results[1].decision['best_label']
    monkeypatch.setattr(matcher,'find_best_scale',lambda *a:pytest.fail('Live scale search'))
    layout=results[0].decision['best_label']
    projected=process_screenshot(replay_image(inputs[1]),catalog=catalog,profile=profile,
        opacity=.5,layout_override=layout,template_variant_id='route_zhanshi_full')
    assert projected.accepted and projected.layer is not None
    assert projected.layout_retrieval is None
    pose=projected.registration.screenshot_to_original
    assert pose[0,0]==pose[1,1]==1.5
    np.testing.assert_allclose(pose[:2,2],inputs[1]['expected_source_origin'],atol=6)
    assert np.linalg.norm(pose[:2,2]-inputs[0]['expected_source_origin'])>50
