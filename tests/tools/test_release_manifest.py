from __future__ import annotations
import json
from src.paths import PROJECT_ROOT


def test_release_manifest_contains_only_live_dependencies():
    from tools.release.build_windows_exe import runtime_files
    paths = [p.relative_to(PROJECT_ROOT).as_posix() for p in runtime_files()]
    assert not any('/sources/' in p for p in paths)
    assert 'configs/runtime_resources.json' in paths
    assert 'configs/scale_config.json' in paths
    assert not any(p.startswith(('maintenance/', 'retired_features/', 'models/', 'workspace/', 'runtime/')) for p in paths)
    from src.app_catalog import MapCatalog
    from src.runtime_resources import read_bundle
    declared = set(read_bundle()['template']['images'])
    assert {p for p in paths if p.startswith('assets/maps/')} == declared
    expected = set()
    for profile in MapCatalog().profiles.values():
        if profile.asset_status == 'pending':
            prefix = f'assets/maps/{profile.key}/'
            assert not any(p.startswith(prefix) for p in paths)
            continue
        layouts = {path.name for path in profile.clean_root.glob('layout_*') if path.is_dir()}
        assert layouts and profile.floors
        versions = {version['id']: version for version in profile.asset_versions}
        expected.update(
            f"{versions[version]['source_root']}/{layout}/{layout}_{floor}.png"
            for layout in layouts for version in profile.template_versions
            for floor in profile.floors_for_layout(layout)
        )
    assert declared == expected
    from tools.release.build_windows_exe import maintenance_files
    assert all(p.is_file() for p in maintenance_files())
    assert not any(p.endswith('_structure.png') for p in paths)


def test_release_includes_required_icons_and_excludes_trial():
    from tools.release.build_windows_exe import runtime_files, ROOT
    paths = [p.relative_to(ROOT).as_posix() for p in runtime_files()]
    assert all(f'assets/layout_retrieval/icons/{name}.png' in paths for name in ('query', 'clean_09', 'clean_15'))
    assert not any(p.startswith('tmp/') for p in paths)
