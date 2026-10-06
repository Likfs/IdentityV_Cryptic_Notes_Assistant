from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.app_catalog import FLOORS, MapCatalog
from src.projection.route_overlay import load_route_overlay_asset, warp_route_overlay, warp_route_structure
from src.registration.runtime_registration import runtime_registration_config, validate_route_overlay
from src.registration_core.color_segmenter import segment_map


def test_three_complete_sets_can_be_selected_without_changing_default():
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    saved = {"unrelated/profile": {"layout_01": "untouched"}}
    assert catalog.selected_variant(profile, "layout_01", saved).variant_id == "route_lhp_solo"
    assert [v.label for v in catalog.variants(profile, "layout_01")] == ["原版干净底图", "凉哈皮单人速刷路线图", "展十速刷路线图"]
    for version in ("route_clean", "route_zhanshi_fast", "route_lhp_solo"):
        catalog.select_version_for_all(profile, version, saved)
        for layout in catalog.layouts(profile):
            variant = catalog.selected_variant(profile, layout, saved)
            assert variant.variant_id == version
            for floor in FLOORS:
                asset = load_route_overlay_asset(variant.calibration_path, floor)
                assert asset.rgba.shape[2] == 4
                assert np.isfinite(asset.asset_to_clean).all()
                assert (asset.structure_mask is not None) == (version == "route_zhanshi_fast")
                assert (asset.asset_kind == "clean") == (version == "route_clean")
                if version == "route_clean":
                    assert np.any(asset.rgba[:, :, 3] == 0)
    before = json.loads(json.dumps(saved))
    with pytest.raises(ValueError, match="未更改"):
        catalog.select_version_for_all(profile, "missing", saved)
    assert saved == before
    assert saved["unrelated/profile"]["layout_01"] == "untouched"


def test_new_palette_validates_correct_pose_and_rejects_displaced_route():
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    variant = next(v for v in catalog.variants(profile, "layout_01") if v.variant_id == "route_zhanshi_fast")
    floor = "1st"
    asset = load_route_overlay_asset(variant.calibration_path, floor)
    clean = cv2.imdecode(np.fromfile(str(catalog.clean_template(profile, "layout_01", floor)), np.uint8), 1)
    size = (clean.shape[1], clean.shape[0])
    observed = segment_map(clean, runtime_registration_config(), infer_known_background=False,
                           include_stairs=False).map_mask
    registration = SimpleNamespace(observation_bbox=(0, 0, *size), observation_mask=observed)
    for dx, expected in ((0, True), (220, False)):
        pose = np.array([[1., 0., dx], [0., 1., 0.]])
        layer, _ = warp_route_overlay(asset, pose, size)
        structure = warp_route_structure(asset, pose, size)
        validation = validate_route_overlay(registration, layer, structure)
        assert validation["accepted"] == expected, validation
    with pytest.raises(ValueError, match="尺寸不一致"):
        validate_route_overlay(registration, layer, np.zeros((1, 1), np.uint8))


def test_malformed_structure_asset_fails_closed(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    cv2.imwrite(str(tmp_path / "route.png"), rgba)
    cv2.imwrite(str(tmp_path / "mask.png"), np.zeros((8, 8), np.uint8))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "layout": "layout_01", "variant_id": "route_new", "floors": {"1st": {
            "status": "PASS", "asset": str(tmp_path / "route.png"),
            "structure_mask": str(tmp_path / "mask.png"), "asset_to_clean_homography": np.eye(3).tolist(),
        }},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="同尺寸"):
        load_route_overlay_asset(calibration, "1st")


def test_clean_bgra_background_is_transparent_and_existing_alpha_is_preserved(tmp_path):
    image = np.array([[[24, 32, 39, 255], [140, 140, 140, 128], [140, 140, 140, 0]]], np.uint8)
    source = tmp_path/'clean.png'
    cv2.imwrite(str(source), image)
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(layout='layout_01', variant_id='route_clean', asset_kind='clean',
        floors={'2nd': dict(status='PASS', asset=str(source), asset_to_clean_homography=np.eye(3).tolist())})))
    asset = load_route_overlay_asset(calibration, '2nd')
    np.testing.assert_array_equal(asset.rgba[:, :, 3], [[0, 128, 0]])
