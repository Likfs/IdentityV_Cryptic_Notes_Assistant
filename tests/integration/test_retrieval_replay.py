"""Real icon/structure matching, with separate labelled synthetic and unlabelled game inputs."""
import shutil

import numpy as np
import pytest

from src.layout_retrieval import LayoutRetriever
from tests.support.replay import read_cases, replay_image


@pytest.mark.parametrize('case', read_cases(), ids=lambda case: case['id'])
def test_real_retrieval_replay_is_ranked_and_requires_confirmation(retrieval_scene, case):
    image = replay_image(case)
    result = retrieval_scene.recognize(image, case['reference_size'])
    assert result.accepted, result.reasons
    assert len(result.candidates) == 3
    assert [c.rank for c in result.candidates] == [1, 2, 3]
    assert all(np.isfinite(c.score) and c.details['rotation'] == 0 for c in result.candidates)
    assert [c.score for c in result.candidates] == sorted((c.score for c in result.candidates), reverse=True)
    decision = result.decision()
    assert decision['accepted'] is False and decision['layout'] is None
    assert decision['score_is_probability'] is False
    if case.get('expected_top1'):
        assert result.candidates[0].layout == case['expected_top1']
    if case.get('expected_strong_evidence') is not None:
        assert result.strong_evidence == case['expected_strong_evidence']


def test_identical_layout_evidence_is_a_manual_tie(tmp_path, retrieval_scene):
    root = tmp_path / 'clean'
    original = retrieval_scene.clean_root / 'layout_05/layout_05_2nd.png'
    for layout in ('layout_05', 'layout_28'):
        target = root / layout / f'{layout}_2nd.png'
        target.parent.mkdir(parents=True)
        shutil.copy2(original, target)
    case = read_cases()[0]
    result = LayoutRetriever(root, cache_root=tmp_path/'cache').recognize(replay_image(case), case['reference_size'])
    assert [c.layout for c in result.candidates] == ['layout_05', 'layout_28']
    assert result.margin == pytest.approx(0, abs=1e-7)
    assert not result.decision()['accepted'] and result.decision()['layout'] is None


def test_blank_frame_with_valid_resources_has_no_candidates(retrieval_scene):
    result = retrieval_scene.recognize(np.zeros((1050, 1180, 3), np.uint8))
    assert not result.accepted and result.decision() is None
    assert not result.candidates
    assert result.reasons == ['截图没有有效墙线结构，请保持地图打开后重试']
