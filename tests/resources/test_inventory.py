import json
from src.app_catalog import PROJECT_ROOT, MapCatalog
from src.app_catalog import MapCatalog
from src.registration.template_matching import direct


def test_calibration_asset_references_are_portable_and_exist():
    def check(value):
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            normalized = value.replace('\\', '/')
            assert not (':/' in normalized and '/assets/' in normalized), value
            if normalized.startswith('assets/'):
                assert (PROJECT_ROOT / normalized).exists(), value
    for path in (PROJECT_ROOT / 'assets/maps').rglob('calibration.json'):
        check(json.loads(path.read_text(encoding='utf-8')))


def test_all_production_originals_are_verified_and_in_release_manifest():
    catalog = MapCatalog()
    profile = catalog.profile('a_bringer_of_doom', 'nightmare')
    layouts = catalog.layouts(profile)
    summary = direct.check_original_resources(profile.key, layouts)
    assert summary['route_lhp_solo']['layouts'] == len(layouts)
    assert summary['route_lhp_solo']['fixed_candidates'] == len(layouts)*3
    assert summary['route_zhanshi_fast']['fixed_candidates'] == len(layouts)
    assert summary['route_clean']['layouts'] == len(layouts)
    assert summary['route_clean']['fixed_candidates'] == len(layouts)*3
    assert summary['route_lhp_solo_clean']['fixed_candidates'] == len(layouts)*3
    assert summary['route_zhanshi_fast_clean']['fixed_candidates'] == len(layouts)
    assert all(not item['errors'] for item in summary.values())
    from tools.release.build_windows_exe import runtime_files, ROOT
    manifest = set(runtime_files())
    for layout in layouts:
        entry, source, _ = direct._source(profile.key, layout, 'route_lhp_solo')
        assert source.resolve() in manifest
        for candidate in entry['candidates']:
            assert all((ROOT/d['path']).resolve() in manifest for d in candidate['dependencies'])
        clean, _, _ = direct._source(profile.key, layout, 'route_clean')
        assert all((ROOT/part['path']).resolve() in manifest for part in clean['source_parts'])
        for variant_id in ('route_lhp_solo_clean', 'route_zhanshi_fast_clean'):
            source_variant, path, _ = direct._source(profile.key, layout, variant_id)
            assert path.resolve() in manifest
            parent, _, _ = direct._source(profile.key, layout, source_variant['reference_variant_id'])
            assert [c['saved_scale'] for c in source_variant['candidates']] == [c['saved_scale'] for c in parent['candidates']]
