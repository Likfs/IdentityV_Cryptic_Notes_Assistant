"""Layered resources still used by offline asset maintenance and historical ID migration."""
from __future__ import annotations
import json
import numpy as np
import pytest
import cv2
from src.app_catalog import FLOORS, MapCatalog
from tools.common.layered_assets import load_route_overlay_asset


def test_route_version_is_selected_per_map_and_difficulty(tmp_path) -> None:
    profiles = []
    for map_id in ("first", "second"):
        clean = tmp_path / map_id / 'clean/layout_05'
        clean.mkdir(parents=True)
        calibration = tmp_path / map_id / "calibration"
        for variant_id in ("route_lhp_solo", "route_02"):
            directory = calibration / variant_id / "layout_05"
            directory.mkdir(parents=True)
            (directory / "calibration.json").write_text(json.dumps({
                "layout": "layout_05", "variant_id": variant_id,
                "route": f"{variant_id}.png",
                "floors": {floor: {"status": "PASS"} for floor in ("2nd", "1st", "b1")},
            }), encoding="utf-8")
        profiles.append({
            "id": map_id, "name": map_id,
            "difficulties": [{"id": "nightmare", "name": "噩梦",
                              "clean_root": str(clean.parent),
                              "calibration_root": str(calibration)}],
        })
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "maps": profiles}), encoding="utf-8")
    catalog = MapCatalog(catalog_path)
    saved = {"first/nightmare": {"layout_05": "route_02"}}
    first = catalog.selected_variant(catalog.profile("first", "nightmare"), "layout_05", saved)
    second = catalog.selected_variant(catalog.profile("second", "nightmare"), "layout_05", saved)
    assert first is not None and first.variant_id == "route_02"
    assert second is not None and second.variant_id == "route_lhp_solo"


def test_three_complete_asset_sets_load_without_changing_default():
    catalog = MapCatalog(offline=True)
    profile = next(iter(catalog.profiles.values()))
    saved = {"unrelated/profile": {"layout_01": "untouched"}}
    assert catalog.selected_variant(profile, "layout_01", saved).variant_id == "route_lhp_solo"
    assert [v.label for v in catalog.variants(profile, "layout_01")] == ["原版干净底图", "凉哈皮单人速刷路线图", "展十速刷路线图"]
    for version in ("route_clean", "route_zhanshi_fast", "route_lhp_solo"):
        for layout in catalog.layouts(profile):
            variant = next(v for v in catalog.variants(profile, layout) if v.variant_id == version)
            assert variant.variant_id == version
            for floor in FLOORS:
                asset = load_route_overlay_asset(variant.calibration_path, floor)
                assert asset.rgba.shape[2] == 4
                assert np.isfinite(asset.asset_to_clean).all()
                assert (asset.structure_mask is not None) == (version == "route_zhanshi_fast")
                assert (asset.asset_kind == "clean") == (version == "route_clean")
                if version == "route_clean":
                    assert np.any(asset.rgba[:, :, 3] == 0)
    assert saved["unrelated/profile"]["layout_01"] == "untouched"


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


def test_saved_route_alias_selects_canonical_version_and_relocated_assets():
    catalog = MapCatalog(offline=True)
    profile = next(iter(catalog.profiles.values()))
    saved = {profile.key: {layout: 'route_01' for layout in catalog.layouts(profile)}}
    for layout in catalog.layouts(profile):
        variant = catalog.selected_variant(profile, layout, saved)
        assert variant.variant_id == 'route_lhp_solo'
        assert saved[profile.key][layout] == 'route_01'  # Reading does not rewrite preferences.
        assert variant.calibration_path == profile.calibration_root / 'route_lhp_solo' / layout / 'calibration.json'
        for floor in FLOORS:
            asset = load_route_overlay_asset(variant.calibration_path, floor)
            assert asset.asset_path.parent == variant.calibration_path.parent


from pathlib import Path
from src import route_projection_core as app_module
from tools.common.layered_assets import RouteOverlayAsset, warp_route_overlay


def test_opacity_percent_and_projected_alpha_agree() -> None:
    assert app_module.opacity_to_percent(0.72) == 72
    assert app_module.percent_to_opacity(72) == 0.72
    assert app_module.percent_to_opacity(0) == 0.0
    assert app_module.percent_to_opacity(100) == 1.0
    rgba = np.full((16, 16, 4), 200, dtype=np.uint8)
    asset = RouteOverlayAsset("layout_05", "2nd", "route_lhp_solo", rgba, np.eye(3),
                              Path("calibration.json"), Path("asset.png"))
    opaque, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 1.0)
    half, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 0.5)
    invisible, _ = warp_route_overlay(asset, np.eye(3), (16, 16), 0.0)
    assert opaque[8, 8, 3] == 200
    assert half[8, 8, 3] == 100
    assert invisible[8, 8, 3] == 0
    assert np.array_equal(opaque[8, 8, :3], half[8, 8, :3])
