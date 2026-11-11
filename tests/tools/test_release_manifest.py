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
            if (PROJECT_ROOT/versions[version]['source_root']/layout).is_dir()
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


def test_build_places_dependencies_inside_and_editable_resources_outside(tmp_path, monkeypatch):
    import sys
    from tools.release import build_windows_exe as build
    from tools.assets.imports import export_runtime_resources as export

    resources = {
        'assets/maps/example/clean/layout_01/layout_01_1st.png': b'image',
        'configs/runtime_resources.json': b'{}',
        'configs/scale_config.json': b'{}',
        'src/registration/template_matching/provenance.json': b'{}',
    }
    for name, raw in resources.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for document in ('README.md', '项目目录与文件说明.md', '新增布局教程.md'):
        (tmp_path/document).write_text('readme', encoding='utf-8')
    (tmp_path/'runtime').mkdir()
    (tmp_path/'runtime/settings.json').write_bytes(b'personal')
    output = tmp_path/'workspace/releases/test'
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'version', lambda _: '6.22.3')
    monkeypatch.setattr(build, 'runtime_files', lambda: [tmp_path/name for name in resources])
    monkeypatch.setattr(export, 'generate', lambda: {})
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'argv', ['build', '--output', str(output)])

    def bundle(command, **kwargs):
        assert command[command.index('--contents-directory')+1] == '_internal'
        data = [command[i+1] for i, value in enumerate(command) if value == '--add-data']
        assert len(data) == 1 and data[0].endswith(':src')
        app = output/build.NAME
        metadata = app/'_internal/src/registration/template_matching/provenance.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_bytes(b'{}')
        (app/f'{build.NAME}.exe').write_bytes(b'exe')
    monkeypatch.setattr(build.subprocess, 'run', bundle)
    build.main()
    app = output/build.NAME
    assert {path.name for path in app.iterdir()} == {
        '_internal', 'assets', 'configs', f'{build.NAME}.exe',
        'README.md', '项目目录与文件说明.md', '新增布局教程.md'}
    for name, raw in resources.items():
        base = app/'_internal' if name.startswith('src/') else app
        assert (base/name).read_bytes() == raw
    assert not (app/'_internal/assets').exists()
    assert not (app/'_internal/configs').exists()
    info = json.loads((output/'build_info.json').read_text(encoding='utf-8'))
    assert info['format'] == 'onedir' and info['contents_directory'] == '_internal'
