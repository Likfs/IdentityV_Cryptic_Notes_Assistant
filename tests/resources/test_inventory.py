import numpy as np
import pytest
from src.app_catalog import PROJECT_ROOT, MapCatalog
from src.registration.template_matching import direct
from tests.support.replay import read_cases, replay_image


def test_runtime_asset_references_are_portable_and_exist():
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
    from src.runtime_resources import read_bundle
    check(read_bundle())


def test_all_published_floor_sources_are_verified_without_maintenance_dependencies():
    catalog = MapCatalog()
    for profile in catalog.profiles.values():
        if profile.asset_status == 'pending':
            continue
        layouts = catalog.layouts(profile)
        versions = {version['id']: version for version in profile.asset_versions}
        coverage = {version: {layout for layout in layouts
                    if (PROJECT_ROOT/versions[version]['source_root']/layout).is_dir()}
                    for version in profile.template_versions}
        assert coverage['clean'] == set(layouts)
        summary = direct.check_original_resources(profile.key, layouts)
        assert set(summary) == set(profile.template_versions)
        for version, item in summary.items():
            assert item['layouts'] == len(coverage[version])
            assert item['fixed_candidates'] == sum(len(profile.floors_for_layout(layout))
                                                   for layout in coverage[version])
            assert not item['errors']
        for layout in layouts:
            clean, _ = direct.floors.sources(profile.key, layout, 'clean')
            for version in profile.template_versions:
                if layout not in coverage[version]:
                    continue
                records, config = direct.floors.sources(profile.key, layout, version)
                assert config['asset_contract'] == 'aligned_floor_images'
                assert set(records) == set(profile.floors_for_layout(layout))
                for floor, record in records.items():
                    assert (PROJECT_ROOT/record['source_image']).is_file()
                    assert record['size'] == clean[floor]['size']
                    assert record['saved_scale'] == clean[floor]['saved_scale'] == profile.minimum_scale_for_layout(layout)
                    assert not record.get('dependencies')


def test_every_published_single_floor_source_is_shipped():
    from tools.release.build_windows_exe import runtime_files
    manifest = set(runtime_files())
    from src.runtime_resources import template_config
    config = template_config()
    catalog = MapCatalog()
    for difficulty in ('hard', 'nightmare'):
        profile = catalog.profile('a_bringer_of_doom', difficulty)
        assert profile.asset_status == 'ready' and profile.minimum_scale == 1.5
    assert set(config['floor_variants']) == {profile.key for profile in catalog.profiles.values()}
    expected = set()
    for profile in catalog.profiles.values():
        if profile.asset_status == 'pending':
            assert catalog.layouts(profile) == []
            assert config['floor_variants'][profile.key] == {}
            continue
        layouts = {path.name for path in profile.clean_root.glob('layout_*') if path.is_dir()}
        assert layouts and profile.floors
        assert set(catalog.layouts(profile)) == layouts
        assert set(config['floor_variants'][profile.key]) == layouts
        versions = {version['id']: version for version in profile.asset_versions}
        for layout in layouts:
            expected_versions = {version for version in profile.template_versions
                if (PROJECT_ROOT/versions[version]['source_root']/layout).is_dir()}
            assert 'clean' in expected_versions
            assert set(config['floor_variants'][profile.key][layout]) == expected_versions
            for version in expected_versions:
                entries, _ = direct.floors.sources(profile.key, layout, version)
                assert set(entries) == set(profile.floors_for_layout(layout))
                for floor, entry in entries.items():
                    path = (PROJECT_ROOT/versions[version]['source_root']/layout/f'{layout}_{floor}.png').resolve()
                    assert entry['source_image'] == path.relative_to(PROJECT_ROOT).as_posix()
                    assert path.is_file() and path in manifest
                    assert entry['saved_scale'] == profile.minimum_scale_for_layout(layout)
                    expected.add(path)
    assert {p for p in manifest if p.is_relative_to(PROJECT_ROOT/'assets/maps')} == expected


@pytest.mark.parametrize('case', [c for c in read_cases() if c['id'].startswith('hard_aligned_')],
                         ids=lambda case: case['id'])
def test_published_hard_crop_keeps_pose_across_route_versions(monkeypatch, case, hard_replay_package):
    """Known source crops verify geometry and display, not gameplay accuracy."""
    profile = MapCatalog().profile('a_bringer_of_doom', 'hard')
    layout, floor = case['expected_top1'], case['expected_floor']
    screenshot = replay_image(case)
    x, y = case['expected_source_origin']
    expected = np.float64([[1.5,0,x], [0,1.5,y], [0,0,1]])
    from src.registration.template_matching import matcher
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a, **k: pytest.fail('Live scale search'))
    results = []
    for version in profile.template_versions:
        result, _, layer, _, _ = direct.project_original(screenshot, profile.key, layout, 1., variant_id=version)
        assert result.accepted, result.rejection_reasons
        assert result.validation['template']['candidate_floor'] == floor
        np.testing.assert_allclose(result.screenshot_to_original, expected, atol=4)
        np.testing.assert_allclose(result.original_to_screen @ result.screenshot_to_original, np.eye(3), atol=1e-10)
        assert '/clean/' in result.validation['matching_source_image']
        results.append((result, layer))
    np.testing.assert_array_equal(results[0][0].screenshot_to_original, results[1][0].screenshot_to_original)
    assert not np.array_equal(results[0][1], results[1][1])


def test_published_hard_recognition_uses_clean_minimum_scale(hard_replay_package):
    from src.layout_retrieval import create_layout_retriever
    profile = MapCatalog().profile('a_bringer_of_doom', 'hard')
    case = next(c for c in read_cases() if c['id'] == 'hard_aligned_layout_23_1st_crop')
    screenshot = replay_image(case)
    retriever = create_layout_retriever('pure_structure', profile.clean_root, floors=profile.floors, workers=2)
    try:
        result = retriever.recognize(screenshot, case['reference_size'])
        assert result.candidates[0].layout == 'layout_23', result.diagnostics()
        assert result.candidates[0].details['floor'] == '1st'
        assert not result.decision()['accepted']  # Confirmation remains explicit.
        assert retriever.fixed_resources
        expected_targets = {(layout, floor) for layout in MapCatalog().layouts(profile) for floor in profile.floors}
        assert len(retriever._targets) == len(expected_targets)
        assert {(target['layout'], target['floor']) for target in retriever._targets} == expected_targets
        assert {target['layout'] for target in retriever._targets} == set(MapCatalog().layouts(profile))
        assert all(tuple(target['scales']) == (1.5,) for target in retriever._targets)
    finally:
        retriever.close()
