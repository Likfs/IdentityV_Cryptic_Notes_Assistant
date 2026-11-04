"""Published single-floor resources and configured route selection."""
from __future__ import annotations
import numpy as np
from src.app_catalog import MapCatalog


def test_complete_aligned_asset_sets_load_with_declared_default(aligned_runtime):
    from src.registration.template_matching import floors
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    saved = {"unrelated/profile": {"layout_01": "untouched"}}
    assert catalog.selected_variant(profile, "layout_01", saved).variant_id == profile.asset_versions[0]['id']
    assert [v.label for v in catalog.variants(profile, "layout_01")] == list(profile.version_names[v] for v in profile.template_versions)
    for version in ("clean", "route_zhanshi_fast", "route_lhp_solo"):
        for layout in catalog.layouts(profile):
            variant = next(v for v in catalog.variants(profile, layout) if v.variant_id == version)
            assert variant.variant_id == version and variant.calibration_path is None
            for floor in profile.floors:
                record, path, config, image, rgba = floors.load(profile.key, layout, version, floor)
                assert rgba.shape == (*image.shape[:2], 4)
                assert record['size'] == list(image.shape[1::-1])
                assert record['saved_scale'] == profile.minimum_scale_for_layout(layout)
                assert config['asset_contract'] == 'aligned_floor_images'
                assert path.is_file()
                assert (variant.asset_kind == 'clean') == (version == 'clean')
                if version == 'clean':
                    assert np.any(rgba[:,:,3] == 0)
    assert saved["unrelated/profile"]["layout_01"] == "untouched"
