"""Regression of a selected-layout pose; this is not a gameplay layout label."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.registration.template_matching import direct, floors, matcher
from src.projection_workflow import process_screenshot
from src.app_catalog import MapCatalog


def test_full_extra_room_screenshot_recovers_partial_width_consensus(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    case = json.loads((root/'tests/fixtures/retrieval/cases.json').read_text(encoding='utf-8'))['projection_regressions'][0]
    path = root/case['screenshot']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    game = matcher.read_image(path)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1.))
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *_: pytest.fail('continuous scale search'))
    catalog = MapCatalog()
    profile = catalog.profile(*case['profile'].split('/'))
    result = process_screenshot(game, catalog=catalog, profile=profile,
        layout_override=case['selected_layout'], opacity=.34,
        registration_backend='template', template_variant_id=case['variant_id'])
    assert result.accepted and result.floor == '1st'
    template = result.registration.validation['template']
    assert template['source_sha256'] == case['expected_source_sha256']
    assert template['saved_scale'] == 1.4899999999999998
    assert template['location'] == case['expected_origin']
    assert template['search']['expanded'] and not template['search']['boundary_edges']
    attempts = result.registration.validation['fixed_candidates']
    assert [a['candidate_floor'] for a in attempts] == ['2nd', '1st', 'b1']
    assert all(a['retry_trigger'] == 'best_at_boundary' for a in attempts)
    assert all(a['search']['expanded'] for a in attempts)
    assert attempts[1]['initial_attempt']['location'] == [272, 56]
    assert attempts[1]['initial_attempt']['score'] == pytest.approx(-22.170291900634766, abs=1e-4)
    assert attempts[1]['score'] > max(attempts[0]['score'], attempts[2]['score'])
    assert not result.registration.validation['warnings']
    np.testing.assert_allclose(result.registration.screenshot_to_original,
        [[template['scale'], 0, 476], [0, template['scale'], 56], [0, 0, 1]])
    np.testing.assert_allclose(result.registration.original_to_screen @ result.registration.screenshot_to_original,
                               np.eye(3), atol=1e-12)
    rgba = floors.load(case['profile'], case['selected_layout'], case['variant_id'], '1st')[4]
    expected = cv2.warpPerspective(rgba, result.registration.original_to_screen, game.shape[1::-1])
    expected[:, :, 3] = np.rint(expected[:, :, 3].astype(np.float32)*.34).astype(np.uint8)
    np.testing.assert_array_equal(result.layer, expected)
