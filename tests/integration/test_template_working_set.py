"""A live working set survives reference/prewarm reads without changing geometry."""
import cv2
import numpy as np
import pytest

from src.app_catalog import MapCatalog
from src.registration.template_matching import direct, floors


@pytest.fixture
def scene(aligned_runtime, monkeypatch):
    profile = MapCatalog().profile('a_bringer_of_doom', 'nightmare')
    path = profile.clean_root/'layout_12/layout_12_1st.png'
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    shot = cv2.resize(image[30:240,30:300], (180,140))
    opened = []
    original = floors.Image.open
    def counted(*args, **kwargs):
        opened.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(floors.Image, 'open', counted)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    return profile, shot, opened


def project(scene, layout='layout_12', variant='route_zhanshi_fast'):
    profile, shot, _ = scene
    result, _, layer, _, _ = direct.project_original(shot, profile.key, layout, .5, variant_id=variant)
    assert result.accepted, result.rejection_reasons
    return result, layer


def assert_identical(first, second):
    np.testing.assert_array_equal(first[0].screenshot_to_original, second[0].screenshot_to_original)
    np.testing.assert_array_equal(first[0].original_to_screen, second[0].original_to_screen)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].validation == second[0].validation


def test_repeated_projection_keeps_three_clean_floors_and_route(scene):
    first = project(scene)
    assert len(scene[2]) == 4
    assert_identical(first, project(scene))
    assert len(scene[2]) == 4, 'Warm projection decoded images again'
    assert len(floors._images) == 4


def test_reference_and_prewarm_do_not_evict_active_projection(scene):
    first = project(scene)
    for floor in scene[0].floors:
        floors.load(scene[0].key, 'layout_12', 'route_custom_a', floor)
    assert floors.prewarm(scene[0].key, 'layout_13', 'route_custom_b')
    count = len(scene[2])
    assert_identical(first, project(scene))
    assert len(scene[2]) == count, 'Reference/prewarm evicted active source pixels'
    assert len(floors._images) <= floors.DECODED_IMAGE_LIMIT


def test_version_switch_reuses_clean_floors_and_layout_switch_releases_old_set(scene):
    first = project(scene)
    count = len(scene[2])
    second = project(scene, variant='route_lhp_solo')
    assert len(scene[2]) == count+1, 'Version change decoded clean images again'
    np.testing.assert_array_equal(first[0].screenshot_to_original, second[0].screenshot_to_original)
    assert not np.array_equal(first[1], second[1])
    switched = project(scene, layout='layout_13', variant='route_lhp_solo')
    count = len(scene[2])
    assert_identical(switched, project(scene, layout='layout_13', variant='route_lhp_solo'))
    assert len(scene[2]) == count
    assert len(floors._images) <= floors.DECODED_IMAGE_LIMIT
    assert len(floors._retained_keys) == 4
    assert all('/layout_13/' in key[0].replace('\\','/') for key in floors._retained_keys)
