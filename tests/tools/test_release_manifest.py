from __future__ import annotations
import json
from src.paths import PROJECT_ROOT


def test_release_manifest_contains_only_live_dependencies():
    from tools.release.build_windows_exe import runtime_files
    paths = [p.relative_to(PROJECT_ROOT).as_posix() for p in runtime_files()]
    assert not any('/sources/' in p for p in paths)
    assert any('/clean/' in p for p in paths)
    assert 'configs/runtime_resources.json' in paths
    assert 'configs/scale_config.json' in paths
    assert 'maintenance/configs/template_matching.json' not in paths
    assert 'maintenance/maps/a_bringer_of_doom/hard/recognition_scales.json' not in paths
    recognition='assets/maps/a_bringer_of_doom/hard/recognition/'
    assert recognition+'index.json' in paths
    index=json.loads((PROJECT_ROOT/recognition/'index.json').read_text(encoding='utf-8'))
    assert {recognition+r['path'] for r in index['targets'].values()} <= set(paths)
    assert not any(p.endswith('/calibration.json') for p in paths)
    assert not any(p.startswith(('maintenance/', 'retired_features/', 'models/', 'workspace/', 'runtime/')) for p in paths)
    config = json.loads((PROJECT_ROOT/'maintenance/configs/template_matching.json').read_text(encoding='utf-8'))
    records = [record for layouts in config['floor_variants'].values()
               for variants in layouts.values() for floors in variants.values() for record in floors.values()]
    referenced = {record['source_image'] for record in records}
    # Ship the final source pixels only. Full derivation chains remain offline.
    shipped = {p for p in paths if '/routes/' in p and p.endswith('.png')}
    assert shipped == {p for p in referenced if '/routes/' in p and p.endswith('.png')}
    from tools.release.build_windows_exe import maintenance_files
    offline = {p.relative_to(PROJECT_ROOT).as_posix() for p in maintenance_files()}
    assert 'maintenance/maps/a_bringer_of_doom/hard/sources/screen_scale_calibration.json' in offline
    assert all('maintenance/maps/a_bringer_of_doom/hard/sources/recognition_zoom50/'+f'{mode}_{i:02}.png' in offline
               for mode in ('minimum','50_percent') for i in (1,2))
    for record in records:
        assert {dep['path'] for dep in record.get('dependencies', [])} <= offline
    assert not any(p.endswith('_structure.png') for p in paths)


def test_release_includes_required_icons_and_excludes_trial():
    from tools.release.build_windows_exe import runtime_files, ROOT
    paths = [p.relative_to(ROOT).as_posix() for p in runtime_files()]
    assert all(f'assets/layout_retrieval/icons/{name}.png' in paths for name in ('query', 'clean_09', 'clean_15'))
    assert not any(p.startswith('tmp/') for p in paths)
