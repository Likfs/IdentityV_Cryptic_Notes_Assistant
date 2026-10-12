from __future__ import annotations
from src.paths import PROJECT_ROOT


def test_release_manifest_contains_only_live_dependencies():
    from tools.release.build_windows_exe import runtime_files
    paths = [p.relative_to(PROJECT_ROOT).as_posix() for p in runtime_files()]
    assert any('/sources/route_lhp_solo/' in p for p in paths)
    assert any('/clean/' in p for p in paths)
    assert any('/routes/' in p and p.endswith('calibration.json') for p in paths)
    assert not any(p.startswith(('retired_features/', 'models/', 'workspace/', 'runtime/')) for p in paths)
    assert not any('/routes/' in p and p.endswith('.png') for p in paths)


def test_release_includes_required_icons_and_excludes_trial():
    from tools.release.build_windows_exe import runtime_files, ROOT
    paths = [p.relative_to(ROOT).as_posix() for p in runtime_files()]
    assert all(f'assets/layout_retrieval/icons/{name}.png' in paths for name in ('query', 'clean_09', 'clean_15'))
    assert not any(p.startswith('tmp/') for p in paths)
