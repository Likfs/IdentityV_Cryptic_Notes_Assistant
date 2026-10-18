"""User-labelled calibration replay, not an independent accuracy benchmark."""
import json

import numpy as np

from src.app_catalog import MapCatalog, PROJECT_ROOT
from src.structure_layout.retriever import PureStructureRetriever
from src.registration.template_matching import direct, matcher
from tests.support.replay import read_cases, replay_image


def test_hard_layout05_real_capture_recognition_and_both_projections(monkeypatch):
    case = next(c for c in read_cases() if c['id'] == 'hard_layout05_1st_game')
    shot = replay_image(case)
    profile = MapCatalog().profile(*case['profile'].split('/'))
    retriever = PureStructureRetriever(profile.clean_root, floors=profile.floors)
    try:
        result = retriever.recognize(shot, case['reference_size'])
        assert result.candidates[0].layout == case['expected_top1'], result.diagnostics()
        assert result.candidates[0].details['floor'] == case['expected_floor']
        assert len(result.candidates) == len(MapCatalog().layouts(profile))
        assert not result.decision()['accepted']  # Human confirmation remains mandatory.
        assert len(retriever._targets) == len(MapCatalog().layouts(profile))*2
        assert all(t['scale_provenance'] == 'verified_source_override' for t in retriever._targets)
    finally:
        retriever.close()
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: (_ for _ in ()).throw(AssertionError('Live scale search')))
    # Independent wall-geometry diagnostic pose, rounded to source pixels.
    screen_points = np.float64([[366,683,1], [605,400,1], [445,600,1]]).T
    expected = np.float64([[3.178,0,1277.28],[0,3.178,-112.839],[0,0,1]]) @ screen_points
    for version in ('route_clean', 'route_zhanshi_full', 'route_zhanshi_full_clean'):
        registration, _, layer, _, _ = direct.project_original(shot, profile.key, 'layout_05', .5, variant_id=version)
        assert registration.accepted, registration.rejection_reasons
        assert layer is not None
        assert registration.validation['template']['candidate_floor'] == '1st'
        assert registration.validation['pending_scale_floors'] == []
        assert [c['candidate_floor'] for c in registration.validation['fixed_candidates']] == ['2nd', '1st']
        # Less than 2 screen pixels from the separately fitted structural pose.
        translation = np.float64([[1,0,256],[0,1,256],[0,0,1]]) if version == 'route_zhanshi_full_clean' else np.eye(3)
        np.testing.assert_allclose(registration.screenshot_to_original @ screen_points, translation @ expected, atol=6.356)
    config = json.loads((PROJECT_ROOT/'maintenance/configs/template_matching.json').read_text(encoding='utf-8'))
    records = config['floor_variants'][profile.key]
    assert {(layout, version, floor) for layout, versions in records.items()
            for version, floors in versions.items() for floor, item in floors.items()
            if item['saved_scale'] is not None} == {
                (layout, version, floor) for layout in MapCatalog().layouts(profile)
                for version in profile.template_versions for floor in profile.floors}
    assert records['layout_05']['route_clean']['1st']['scale_status'] == 'verified_capture'
    assert all(item['scale_status'] == 'derived_geometry' for layout, versions in records.items()
               for floors in versions.values() for floor, item in floors.items()
               if (layout, floor) != ('layout_05', '1st'))

