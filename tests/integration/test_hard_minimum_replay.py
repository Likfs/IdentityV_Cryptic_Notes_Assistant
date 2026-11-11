"""User-reviewed minimum-zoom captures against the published aligned assets."""
import numpy as np
import pytest
from src.app_catalog import MapCatalog
from src.layout_retrieval import create_layout_retriever
from src.registration.template_matching import direct, matcher
from tests.support.replay import read_cases, replay_image


@pytest.mark.parametrize('case', [c for c in read_cases() if c['id'] in ('hard_minimum_01','hard_minimum_03')], ids=lambda c:c['id'])
def test_real_minimum_capture_recognition_and_both_projections(monkeypatch, case, hard_replay_package):
    profile = MapCatalog().profile(*case['profile'].split('/'))
    shot = replay_image(case)
    retriever = create_layout_retriever('pure_structure', profile.clean_root, floors=profile.floors, workers=2)
    try:
        result = retriever.recognize(shot, case['reference_size'])
        assert result.candidates[0].layout == case['expected_top1'], result.diagnostics()
        assert result.candidates[0].details['floor'] == case['expected_floor']
        assert not result.decision()['accepted']
        assert len(retriever._targets) == len(MapCatalog().layouts(profile))*2
        assert all(tuple(t['scales']) == (1.5,) for t in retriever._targets)
    finally:
        retriever.close()
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('Live scale search'))
    expected = np.float64([[1.5,0,case['expected_source_origin'][0]], [0,1.5,case['expected_source_origin'][1]], [0,0,1]])
    registrations=[]
    for version in profile.template_versions:
        registration, _, layer, _, _ = direct.project_original(shot, profile.key, case['expected_top1'], .5, variant_id=version)
        assert registration.accepted, registration.rejection_reasons
        assert layer is not None
        assert registration.validation['template']['candidate_floor'] == '1st'
        assert registration.validation['pending_scale_floors'] == []
        assert [c['candidate_floor'] for c in registration.validation['fixed_candidates']] == ['2nd','1st']
        assert '/clean/' in registration.validation['matching_source_image']
        np.testing.assert_allclose(registration.screenshot_to_original, expected, atol=6)
        registrations.append(registration)
    np.testing.assert_array_equal(registrations[0].screenshot_to_original, registrations[1].screenshot_to_original)
