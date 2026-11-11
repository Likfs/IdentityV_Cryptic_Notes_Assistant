import json
import shutil

import cv2
import numpy as np
import pytest

from src import asset_sync
from src.app_catalog import MapCatalog, load_settings
from src.floor_definitions import FLOORS
from src.layout_retrieval import create_layout_retriever
from src.registration.template_matching import direct, floors
from src.runtime_resources import runtime_files
from tests.support.aligned_runtime import activate, build_multifloor_package


@pytest.fixture
def scene(tmp_path, monkeypatch):
    root = build_multifloor_package(tmp_path/'package')
    activate(root, monkeypatch)
    yield root, MapCatalog().profile('phosphorescence_disease', 'hard')
    floors._close_fixed_session()


def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def test_discovers_only_actual_floors_and_preserves_settings_and_scales(scene):
    root, profile = scene
    personal = root/'personal.json'
    raw = json.dumps(dict(map_id=profile.map_id, difficulty_id=profile.difficulty_id,
                          reference_floor='b4', template_route_versions={profile.key: '测试路线'})).encode()
    personal.write_bytes(raw)
    assert profile.floors == FLOORS
    assert profile.floors_for_layout('layout_01') == ('b1','b2','b3','b4')
    assert not (profile.clean_root/'layout_01/layout_01_2nd.png').exists()
    assert not asset_sync.synchronize(root)
    settings = load_settings(MapCatalog(), personal)
    assert settings['reference_floor'] == 'b4' and personal.read_bytes() == raw
    runtime_files(root)
    # Remove one complete floor across clean and routes, then restore it.
    saved = {}
    for version in profile.asset_versions:
        path = root/version['source_root']/'layout_02/layout_02_3rd.png'
        saved[path] = path.read_bytes()
        path.unlink()
    asset_sync.synchronize(root)
    updated = MapCatalog().profile(profile.map_id, profile.difficulty_id)
    assert '3rd' not in updated.floors
    assert '3rd' not in updated.floors_for_layout('layout_02')
    for path, raw_image in saved.items():
        path.write_bytes(raw_image)
    asset_sync.synchronize(root)
    assert MapCatalog().profile(profile.map_id, profile.difficulty_id).floors == FLOORS


@pytest.mark.parametrize('floor', FLOORS)
@pytest.mark.parametrize('over_budget', [False, True])
def test_all_supported_floors_register_and_keep_bounded_cache(scene, floor, over_budget, monkeypatch):
    root, profile = scene
    image = read(profile.clean_root/f'layout_02/layout_02_{floor}.png')
    shot = cv2.resize(image[45:315,60:390], (220,180), interpolation=cv2.INTER_AREA)
    if over_budget:
        monkeypatch.setattr(floors, 'DECODED_IMAGE_BYTES', image.shape[0] * image.shape[1] * 7 * 4)
    opened = []
    original = floors.Image.open
    def counted(*args, **kwargs):
        opened.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(floors.Image, 'open', counted)
    first = direct.project_original(shot, profile.key, 'layout_02', .5, variant_id='测试路线')
    cold_decodes = len(opened)
    second = direct.project_original(shot, profile.key, 'layout_02', .5, variant_id='测试路线')
    assert first[0].accepted, first[0].rejection_reasons
    assert first[0].validation['template']['candidate_floor'] == floor
    assert len(first[0].validation['fixed_candidates']) == 7
    assert first[0].validation == second[0].validation
    np.testing.assert_array_equal(first[0].original_to_screen, second[0].original_to_screen)
    np.testing.assert_array_equal(first[2], second[2])
    if not over_budget:
        assert cold_decodes == 8 and len(opened) == cold_decodes, 'Warm registration decoded retained pixels'
        assert len(floors._images) == 8
    else:
        assert len(floors._images) <= 4
    assert sum(a.nbytes+b.nbytes for a,b in floors._images.values()) <= floors.DECODED_IMAGE_BYTES
    assert len(floors._retained_keys) == (2 if over_budget else 8)
    # Reference reads and prewarm keep the winning clean/route pixels protected.
    retained = {key: floors._images[key] for key in floors._retained_keys}
    for other in profile.floors:
        floors.load(profile.key, 'layout_02', '测试路线', other)
    assert floors.prewarm(profile.key, 'layout_01', '测试路线')
    assert all(floors._images[key] is value for key, value in retained.items())


def test_four_basements_recognize_without_an_upper_floor(scene):
    root, profile = scene
    # An independent all-basement profile exercises the full disk recognition path.
    for version in profile.asset_versions:
        shutil.rmtree(root/version['source_root']/'layout_02')
    asset_sync.synchronize(root)
    profile = MapCatalog().profile(profile.map_id, profile.difficulty_id)
    assert profile.floors == ('b1','b2','b3','b4')
    image = read(profile.clean_root/'layout_01/layout_01_b4.png')
    shot = cv2.resize(image, (300,240), interpolation=cv2.INTER_AREA)
    retriever = create_layout_retriever('pure_structure', profile.clean_root, floors=profile.floors, cache_root=root/'features')
    try:
        result = retriever.recognize(shot, shot.shape[1::-1])
        assert result.candidates, result.diagnostics()
        assert result.candidates[0].layout == 'layout_01'
        assert result.candidates[0].details['floor'] == 'b4'
        assert {target['floor'] for target in retriever._targets} == set(profile.floors)
        assert all(not isinstance(value, np.ndarray) for target in retriever._targets for value in target.values())
    finally:
        retriever.close()


def test_decode_byte_budget_returns_large_images_without_retaining_them(scene, monkeypatch):
    root, profile = scene
    monkeypatch.setattr(floors, 'DECODED_IMAGE_BYTES', 1)
    image = read(profile.clean_root/'layout_02/layout_02_b3.png')
    shot = cv2.resize(image[45:315,60:390], (220,180), interpolation=cv2.INTER_AREA)
    result, _, layer, _, _ = direct.project_original(shot, profile.key, 'layout_02', .5, variant_id='测试路线')
    assert result.accepted, result.rejection_reasons
    assert result.validation['template']['candidate_floor'] == 'b3' and layer is not None
    assert not floors._images
    assert not floors.matcher._map_cache


def test_clean_only_registration_and_recognition_match_the_published_clean_version(scene):
    root, profile = scene
    image = read(profile.clean_root/'layout_02/layout_02_b4.png')
    shot = cv2.resize(image[45:315,60:390], (220,180), interpolation=cv2.INTER_AREA)
    before = direct.project_original(shot, profile.key, 'layout_02', .5, variant_id='clean')
    assert before[0].accepted, before[0].rejection_reasons
    floors._close_fixed_session()
    scales = (root/'configs/scale_config.json').read_bytes()
    shutil.rmtree(root/'assets/maps/phosphorescence_disease/hard/clean-with-route')
    assert asset_sync.synchronize(root)
    profile = MapCatalog().profile(profile.map_id, profile.difficulty_id)
    assert profile.template_versions == ('clean',) and profile.asset_status == 'ready'
    assert (root/'configs/scale_config.json').read_bytes() == scales
    after = direct.project_original(shot, profile.key, 'layout_02', .5)
    assert after[0].accepted, after[0].rejection_reasons
    assert after[0].validation == before[0].validation
    np.testing.assert_array_equal(after[0].original_to_screen, before[0].original_to_screen)
    np.testing.assert_array_equal(after[2], before[2])
    assert len(floors._retained_keys) == 7 and len(floors._images) <= 8
    full_shot = cv2.resize(image, (300,240), interpolation=cv2.INTER_AREA)
    retriever = create_layout_retriever('pure_structure', profile.clean_root, floors=profile.floors, cache_root=root/'features')
    try:
        result = retriever.recognize(full_shot, full_shot.shape[1::-1])
        assert result.candidates, result.diagnostics()
        assert result.candidates[0].layout == 'layout_02'
        assert result.candidates[0].details['floor'] == 'b4'
    finally:
        retriever.close()
