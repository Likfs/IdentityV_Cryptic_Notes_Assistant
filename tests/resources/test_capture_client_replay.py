"""User-labelled b2 capture: a 1920x1080 game on a 3072x1920 desktop."""
import hashlib
import json

import numpy as np
import cv2
import pytest

from src.app_catalog import MapCatalog
from src.paths import PROJECT_ROOT
from src.projection_workflow import process_screenshot
from src.registration.template_matching import matcher, floors
from tests.support.aligned_runtime import activate, build_phosphorescence_replay_package


def test_capture_client_size_recovers_layout13_b2(tmp_path, monkeypatch):
    case = next(c for c in json.loads((PROJECT_ROOT/'tests/fixtures/retrieval/cases.json').read_text(
        encoding='utf-8'))['projection_regressions'] if c['id'] == 'phosphorescence_layout13_b2_windowed')
    screenshot = PROJECT_ROOT/case['screenshot']
    assert hashlib.sha256(screenshot.read_bytes()).hexdigest() == case['sha256']
    fixture = PROJECT_ROOT/'tests/fixtures/registration/aligned_phosphorescence/clean/layout_13'
    for name, expected in case['source_sha256'].items():
        assert hashlib.sha256((fixture/name).read_bytes()).hexdigest() == expected
    root = build_phosphorescence_replay_package(tmp_path/'replay')
    activate(root, monkeypatch)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *_: pytest.fail('Runtime scale search'))
    image = matcher.read_image(screenshot)
    catalog = MapCatalog()
    profile = catalog.profile('phosphorescence_disease', 'nightmare')
    assert profile.minimum_scale_for_layout('layout_13') == 1.5
    try:
        def project(size):
            return process_screenshot(image, catalog=catalog, profile=profile, opacity=.34,
                layout_override='layout_13', template_variant_id='clean', capture_client_size=size)
        # Reproduce the former desktop-based failure, then change only the supplied size.
        wrong = project(tuple(case['desktop_size']))
        assert wrong.floor == 'b3'
        result = project(tuple(case['capture_client_size']))
        assert result.accepted and result.floor == case['expected_floor']
        template = result.registration.validation['template']
        assert template['saved_scale'] == case['minimum_scale'] == 1.5
        assert template['scale'] == 2.
        assert template['capture_client']['width'] == 1920
        assert template['capture_client']['height'] == 1080
        np.testing.assert_allclose(template['location'], case['expected_origin'], atol=2)
        np.testing.assert_allclose(result.registration.original_to_screen @
            result.registration.screenshot_to_original, np.eye(3), atol=1e-12)
    finally:
        floors._close_fixed_session()


@pytest.mark.parametrize('zoom_units', [7, 13])
def test_resampled_capture_replays_fixed_height_scale_without_map_scale_search(tmp_path, monkeypatch, zoom_units):
    """Synthetic resizing of a labelled capture checks geometry, not native FH accuracy."""
    from src.floor_indicator import FloorDetection
    case = next(c for c in json.loads((PROJECT_ROOT/'tests/fixtures/retrieval/cases.json').read_text(
        encoding='utf-8'))['projection_regressions'] if c['id'] == 'phosphorescence_layout13_b2_windowed')
    root = build_phosphorescence_replay_package(tmp_path/'replay')
    activate(root, monkeypatch)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *_:pytest.fail('Runtime scale search'))
    image = matcher.read_image(PROJECT_ROOT/case['screenshot'])
    factor = 1920000/1779556  # (2560/1334) / (1334/750), not 2560/frame_width.
    expected_scale = 1.5*factor/(zoom_units/7)
    image = cv2.resize(image, None, fx=2/expected_scale, fy=2/expected_scale, interpolation=cv2.INTER_LINEAR)
    profile = MapCatalog().profile('phosphorescence_disease', 'nightmare')
    try:
        result = process_screenshot(image, catalog=object(), profile=profile, opacity=.34,
            layout_override='layout_13', template_variant_id='clean', capture_client_size=(2560, 1334),
            map_zoom_units=zoom_units, floor_detection=FloorDetection('b2'))
        assert result.accepted and result.floor == 'b2', result.reasons
        template = result.registration.validation['template']
        assert template['saved_scale'] == 1.5
        assert template['scale'] == pytest.approx(expected_scale)
        assert template['capture_client']['geometry_model'] == 'fixed_height'
        np.testing.assert_allclose(template['location'], case['expected_origin'], atol=3)
        np.testing.assert_allclose(result.registration.original_to_screen @
            result.registration.screenshot_to_original, np.eye(3), atol=1e-12)
    finally:
        floors._close_fixed_session()
