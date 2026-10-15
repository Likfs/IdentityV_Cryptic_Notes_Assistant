from __future__ import annotations
import json
from src.paths import PROJECT_ROOT


def test_release_manifest_contains_only_live_dependencies():
    from tools.release.build_windows_exe import runtime_files
    paths = [p.relative_to(PROJECT_ROOT).as_posix() for p in runtime_files()]
    assert any('/sources/route_lhp_solo/' in p for p in paths)
    assert any('/clean/' in p for p in paths)
    assert any('/routes/' in p and p.endswith('calibration.json') for p in paths)
    assert not any(p.startswith(('retired_features/', 'models/', 'workspace/', 'runtime/')) for p in paths)
    config = json.loads((PROJECT_ROOT/'configs/template_matching.json').read_text(encoding='utf-8'))
    records = [record for layouts in config['floor_variants'].values()
               for variants in layouts.values() for floors in variants.values() for record in floors.values()]
    referenced = {record['source_image'] for record in records}
    # Rectified images still validate their original PNGs at runtime. Their explicit
    # dependencies belong in the release; unrelated legacy PNGs remain excluded.
    referenced.update(dep['path'] for record in records for dep in record.get('dependencies', []))
    shipped = {p for p in paths if '/routes/' in p and p.endswith('.png')}
    assert shipped == {p for p in referenced if '/routes/' in p and p.endswith('.png')}
    for record in records:
        if record.get('geometry_verification', {}).get('method') == 'calibrated_rectification':
            dependencies = {dep['path'] for dep in record['dependencies']}
            assert any('/routes/route_lhp_solo/' in p and p.endswith('.png') for p in dependencies)
            assert any(p.endswith('/rectification.json') for p in dependencies)
            assert dependencies <= set(paths)
    assert not any(p.endswith('_structure.png') for p in paths)


def test_release_includes_required_icons_and_excludes_trial():
    from tools.release.build_windows_exe import runtime_files, ROOT
    paths = [p.relative_to(ROOT).as_posix() for p in runtime_files()]
    assert all(f'assets/layout_retrieval/icons/{name}.png' in paths for name in ('query', 'clean_09', 'clean_15'))
    assert not any(p.startswith('tmp/') for p in paths)
