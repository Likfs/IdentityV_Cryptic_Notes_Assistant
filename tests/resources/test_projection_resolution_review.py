import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.app_catalog import MapCatalog
from src.projection_workflow import process_screenshot
from src.registration.template_matching import direct, floors, matcher


def test_small_floor_margin_reviews_same_fixed_scale_and_delivers_basement_pixels(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root/'tests/fixtures/retrieval/cases.json').read_text(encoding='utf-8'))
    case = next(c for c in cases['projection_regressions'] if c['id'] == 'layout19_floor_ambiguity')
    path = root/case['screenshot']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1.))
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *_: pytest.fail('continuous scale search'))
    catalog = MapCatalog()
    game = matcher.read_image(path)
    result = process_screenshot(game, catalog=catalog, profile=catalog.profile(*case['profile'].split('/')),
        layout_override=case['selected_layout'], opacity=.34, registration_backend='template',
        template_variant_id=case['variant_id'])
    assert result.accepted and result.floor == case['expected_floor']
    validation = result.registration.validation
    template, review = validation['template'], validation['resolution_review']
    assert template['source_sha256'] == case['expected_source_sha256']
    assert template['saved_scale'] == template['scale'] == 1.49
    assert template['ds'] == .5 and template['perf_mode'] == 2
    assert template['location'] == case['expected_origin']
    assert review['triggered'] and review['initial_floor'] == '2nd' and review['final_floor'] == 'b1'
    assert review['initial_gap'] == pytest.approx(.3286781311, abs=1e-4)
    assert review['review_gap'] == pytest.approx(.7438049316, abs=1e-4)
    assert any('楼层仍难区分' in warning for warning in validation['warnings'])
    assert all(a['ds'] == .5 for a in validation['fixed_candidates'])
    np.testing.assert_array_equal(result.registration.screenshot_to_original,
                                  [[1.49, 0, -186], [0, 1.49, -502], [0, 0, 1]])
    rgba = floors.load(case['profile'], case['selected_layout'], case['variant_id'], 'b1')[4]
    expected = cv2.warpPerspective(rgba, result.registration.original_to_screen, game.shape[1::-1])
    expected[:, :, 3] = np.rint(expected[:, :, 3].astype(np.float32)*.34).astype(np.uint8)
    np.testing.assert_array_equal(result.layer, expected)
