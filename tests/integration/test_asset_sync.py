"""Startup discovers complete assets without altering personal or declared profile data."""
import json
from pathlib import Path

import numpy as np
import pytest

from src import asset_sync as sync
from src.app_catalog import MapCatalog, load_settings
from src.runtime_resources import runtime_files, read_bundle
from tests.support.aligned_runtime import difficulty, publish, write_png


@pytest.fixture
def package(tmp_path):
    definition = difficulty('hard', '人工困难', ('1st',), {'clean': '底图', 'custom': '人工路线'})
    definition['note'] = 'keep'
    image = np.full((80, 100, 3), 100, np.uint8)
    for version in definition['asset_versions']:
        write_png(tmp_path/version['source_root']/'layout_01/layout_01_1st.png', image)
    catalog = dict(schema_version=2, custom='keep', maps=[dict(
        id='a_bringer_of_doom', name='厄运之女', difficulties=[definition])])
    publish(tmp_path, catalog, scales={'maps': {'a_bringer_of_doom': {'hard': dict(
        minimum_scale=1.25, layout_minimum_scales={})}}})
    (tmp_path/'runtime').mkdir()
    (tmp_path/'runtime/settings.json').write_bytes(b'{"opacity":0.73,"personal":"keep"}')
    return tmp_path, catalog, image


def add_assets(root, identifier='easy', layouts=('layout_01', 'layout_02'), floors=('1st', '2nd'), route_name='route_lhp_solo',
               *, map_id='a_bringer_of_doom'):
    base = root/'assets/maps'/map_id/identifier
    for layout in layouts:
        for floor in floors:
            image = np.full((80, 100, 3), 100, np.uint8)
            image[20:60, 20:80] = 150
            image[25:55, 25:75] = 100
            for source in ('clean', f'clean-with-route/{route_name}'):
                write_png(base/source/layout/f'{layout}_{floor}.png', image)
    return base


def config_snapshot(root):
    return {name: ((root/name).read_bytes(), (root/name).stat().st_mtime_ns) for name in sync.CONFIG_FILES}


def test_cached_startup_and_full_offline_audit_produce_identical_manifest(package, monkeypatch):
    from src.aligned_assets import generate
    from src.asset_audit import AssetAudit
    from PIL import Image
    root, _, _ = package
    add_assets(root)
    sync.synchronize(root)
    before = config_snapshot(root)
    expected, expected_scales = generate(root)
    with monkeypatch.context() as patch:
        patch.setattr(Image.Image, 'load', lambda *_: pytest.fail('Cached startup decoded an unchanged PNG'))
        assert not sync.synchronize(root)
        with AssetAudit(root) as audit:
            actual, actual_scales = generate(root, audit=audit)
    assert actual == expected and actual_scales == expected_scales
    assert config_snapshot(root) == before


def test_frozen_startup_audits_internal_metadata_and_publishes_external_configs(package, monkeypatch):
    import sys
    from src import paths
    from src.aligned_assets import generate
    from src.runtime_resources import runtime_files, read_bundle

    root, _, _ = package
    name = paths.TEMPLATE_PROVENANCE_PATH
    internal = root/'_internal'
    target = internal/name
    target.parent.mkdir(parents=True)
    (root/name).replace(target)
    monkeypatch.setattr(paths, 'PROJECT_ROOT', root)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(internal), raising=False)
    assert not sync.synchronize(root)
    assert target in runtime_files(root)
    assert generate(root)[0]['payload'] == read_bundle(root)

    add_assets(root)
    internal_before = target.read_bytes()
    assert sync.synchronize(root)
    assert name in read_bundle(root)['files']
    assert not any(key.startswith('_internal/') for key in read_bundle(root)['files'])
    assert target.read_bytes() == internal_before
    assert not (internal/'configs').exists()
    assert not sync.synchronize(root)
    target.write_bytes(internal_before+b'changed')
    with pytest.raises(ValueError, match='指纹'):
        runtime_files(root)


def test_cached_dimensions_do_not_hide_changed_route_canvas(package):
    root, _, _ = package
    add_assets(root)
    sync.synchronize(root)
    before = config_snapshot(root)
    path = root/'assets/maps/a_bringer_of_doom/easy/clean-with-route/route_lhp_solo/layout_01/layout_01_1st.png'
    write_png(path, np.full((81, 100, 3), 100, np.uint8))
    with pytest.raises(ValueError, match='画布尺寸'):
        sync.synchronize(root)
    assert config_snapshot(root) == before


def test_cancelled_rescan_preserves_previous_validation_cache_and_configs(package):
    root, _, _ = package
    sync.synchronize(root)
    cache = root/'runtime/cache/asset_audit/png_validation.json'
    previous = cache.read_bytes()
    before = config_snapshot(root)
    add_assets(root)
    class Cancelled(Exception):
        pass
    def progress(stage, done, total):
        if stage == '核对发布输入':
            raise Cancelled()
    with pytest.raises(Cancelled):
        sync.synchronize(root, progress=progress)
    assert cache.read_bytes() == previous and config_snapshot(root) == before


def test_cancellation_before_commit_does_not_publish_partial_configs(package):
    root, _, _ = package
    add_assets(root)
    before = config_snapshot(root)
    class Cancelled(Exception):
        pass
    stages = []
    def progress(stage, done, total):
        stages.append(stage)
        if stage == '更新配置和资源清单':
            raise Cancelled()
    with pytest.raises(Cancelled):
        sync.synchronize(root, progress=progress)
    assert config_snapshot(root) == before
    assert any(stage.startswith('校验素材') for stage in stages)
    assert '核对导出输入' in stages


def test_new_map_is_registered_only_after_its_first_complete_difficulty(package):
    root, _, _ = package
    map_id = '新地图'
    (root/'assets/maps'/map_id/'normal').mkdir(parents=True)
    before = config_snapshot(root)
    assert any('暂未接入' in message for message in sync.synchronize(root))
    assert config_snapshot(root) == before

    add_assets(root, 'normal', route_name='中文路线', map_id=map_id)
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile(map_id, 'normal')
    assert (profile.map_name, profile.difficulty_name, profile.asset_status) == (map_id, '普通', 'ready')
    assert profile.template_versions == ('中文路线', 'clean') and profile.minimum_scale == 1.5
    assert set(read_bundle(root)['template']['groups'][profile.key]) == {'layout_01', 'layout_02'}
    runtime_files(root)
    assert not sync.synchronize(root)


@pytest.mark.parametrize('minimum_scale', [None, 1.6])
def test_pending_difficulty_waits_for_complete_assets_then_activates(package, minimum_scale):
    root, catalog, image = package
    pending = difficulty('normal', '自定义普通', (), {'clean': '原版干净底图'})
    pending_root = pending.pop('clean_root')
    pending.update(asset_status='pending', layout_floors={}, asset_versions=[], template_versions=[])
    catalog['maps'][0]['difficulties'].append(pending)
    scales = json.loads((root/sync.SCALE_PATH).read_bytes())
    scales['maps']['a_bringer_of_doom']['normal'] = dict(
        minimum_scale=minimum_scale, layout_minimum_scales={'layout_02': 2.0})
    publish(root, catalog, scales=scales)
    settings = (root/'runtime/settings.json').read_bytes()
    before = config_snapshot(root)
    (root/pending_root/'layout_01').mkdir(parents=True)
    assert any('暂未接入' in message for message in sync.synchronize(root))
    assert config_snapshot(root) == before

    write_png(root/pending_root/'layout_01/layout_01_1st.png', image)
    assert any('已启用待配置难度' in message for message in sync.synchronize(root))
    clean_profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'normal')
    assert clean_profile.asset_status == 'ready' and clean_profile.template_versions == ('clean',)
    assert clean_profile.floors == ('1st',)
    assert clean_profile.minimum_scale == (1.5 if minimum_scale is None else minimum_scale)
    assert clean_profile.layout_minimum_scales == {'layout_02': 2.0}
    runtime_files(root)
    add_assets(root, 'normal')
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'normal')
    assert (profile.difficulty_name, profile.asset_status, profile.floors) == ('自定义普通', 'ready', ('2nd', '1st'))
    assert profile.minimum_scale == (1.5 if minimum_scale is None else minimum_scale)
    assert profile.layout_minimum_scales == {'layout_02': 2.0}
    assert (root/'runtime/settings.json').read_bytes() == settings
    runtime_files(root)
    assert not sync.synchronize(root)


def test_startup_migrates_old_clean_id_and_preserves_preference(package):
    root, catalog, _ = package
    definition = catalog['maps'][0]['difficulties'][0]
    definition['asset_versions'][0]['id'] = 'route_clean'
    definition['template_versions'] = ['custom', 'route_clean']
    (root/sync.CATALOG_PATH).write_text(json.dumps(catalog), encoding='utf-8')
    key = 'a_bringer_of_doom/hard'
    personal = root/'runtime/settings.json'
    original = json.dumps(dict(template_route_versions={key: 'route_clean'})).encode()
    personal.write_bytes(original)

    assert sync.synchronize(root)
    updated = MapCatalog(root/sync.CATALOG_PATH)
    profile = updated.profile('a_bringer_of_doom', 'hard')
    assert profile.template_versions == ('custom', 'clean')
    assert profile.version_names == {'custom': '人工路线', 'clean': '底图'}
    assert set(read_bundle(root)['template']['groups'][key]['layout_01']) == {'custom', 'clean'}
    assert load_settings(updated, personal)['template_route_versions'][key] == 'clean'
    assert personal.read_bytes() == original
    assert not sync.synchronize(root)


def test_startup_preserves_layout_scale_overrides_and_publishes_changes(package):
    root, original, image = package
    definition = original['maps'][0]['difficulties'][0]
    scale_data = json.loads((root/sync.SCALE_PATH).read_text())
    settings = scale_data['maps']['a_bringer_of_doom']['hard']
    settings.update(minimum_scale=1.5, layout_minimum_scales={'layout_02': 2.0})
    for version in definition['asset_versions']:
        write_png(root/version['source_root']/'layout_02/layout_02_1st.png', image)
    publish(root, original, scales=scale_data)
    for layout in ('layout_01', 'layout_02'):
        write_png(root/f'assets/maps/a_bringer_of_doom/hard/clean-with-route/custom/{layout}/{layout}_1st.png', image)
    personal = (root/'runtime/settings.json').read_bytes()
    sync.synchronize(root)
    updated = json.loads((root/sync.CATALOG_PATH).read_text(encoding='utf-8'))
    current = updated['maps'][0]['difficulties'][0]
    assert 'minimum_scale' not in current and 'layout_minimum_scales' not in current
    assert json.loads((root/sync.SCALE_PATH).read_text()) == scale_data
    assert current['asset_versions'][1]['name'] == 'custom'
    settings['layout_minimum_scales']['layout_02'] = 1.6
    edited = json.dumps(scale_data).encode()
    (root/sync.SCALE_PATH).write_bytes(edited)
    sync.synchronize(root)
    scales = json.loads((root/sync.SCALE_PATH).read_text(encoding='utf-8'))['maps']
    assert scales['a_bringer_of_doom']['hard'] == settings
    assert (root/sync.SCALE_PATH).read_bytes() == edited
    before = config_snapshot(root)
    assert sync.synchronize(root) == []
    assert config_snapshot(root) == before and (root/'runtime/settings.json').read_bytes() == personal


@pytest.mark.parametrize('overrides', [{'layout_01': 0}])
def test_invalid_layout_scale_override_never_publishes_a_partial_configuration(package, overrides):
    root, original, _ = package
    path = root/sync.SCALE_PATH
    data = json.loads(path.read_text())
    data['maps']['a_bringer_of_doom']['hard']['layout_minimum_scales'] = overrides
    path.write_text(json.dumps(data), encoding='utf-8')
    before = config_snapshot(root)
    with pytest.raises(ValueError, match='layout_minimum_scales'):
        sync.synchronize(root)
    assert config_snapshot(root) == before


@pytest.mark.parametrize('version', ['zhanshi_fast', 'route_zhanshi_full', '我的单人路线'])
def test_new_version_uses_folder_name_without_translation_or_cross_profile_names(package, version):
    root, original, _ = package
    # A name from another difficulty must not become the new folder's display name.
    other = original['maps'][0]['difficulties'][0]
    other['asset_versions'][1]['id'] = version
    other['asset_versions'][1]['name'] = '另一个难度的人工名称'
    other['template_versions'] = ['clean', version]
    publish(root, original)
    add_assets(root, route_name=version)
    personal = (root/'runtime/settings.json').read_bytes()
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert profile.version_names == {'clean': '原版干净底图', version: version}
    assert next(v for v in profile.asset_versions if v['id'] == version)['source_root'].endswith('/'+version)
    assert (root/'runtime/settings.json').read_bytes() == personal


def test_runtime_versions_keep_declared_order_and_sync_names_from_folders(package, monkeypatch):
    from tests.support.aligned_runtime import activate
    from src.registration.template_matching import direct
    root, _, _ = package
    add_assets(root, route_name='route_zhanshi_fast')
    add_assets(root, route_name='我的路线')
    sync.synchronize(root)
    data = json.loads((root/sync.CATALOG_PATH).read_text(encoding='utf-8'))
    definition = data['maps'][0]['difficulties'][1]
    definition['template_versions'] = ['我的路线', 'route_zhanshi_fast', 'clean']
    definition['asset_versions'][1]['name'] = '保留的人工名称'
    publish(root, data)
    assert sync.synchronize(root)
    activate(root, monkeypatch)
    catalog = MapCatalog(root/sync.CATALOG_PATH)
    profile = catalog.profile('a_bringer_of_doom', 'easy')
    try:
        assert direct.available_variants(profile.key, 'layout_01') == definition['template_versions']
        assert profile.version_names['route_zhanshi_fast'] == 'route_zhanshi_fast'
        assert {variant.variant_id: variant.label for variant in catalog.variants(profile, 'layout_01')} == profile.version_names
        assert set(profile.version_names) == {v['id'] for v in definition['asset_versions']}
    finally:
        direct.floors._close_fixed_session()


@pytest.mark.parametrize(('identifier', 'label'), [('novice', '新手'), ('easy', '简单'), ('normal', '普通')])
def test_new_difficulty_auto_publishes_versions_and_scale(package, identifier, label):
    root, original, _ = package
    settings = (root/'runtime/settings.json').read_bytes()
    add_assets(root, identifier)
    assert sync.synchronize(root)
    updated = json.loads((root/sync.CATALOG_PATH).read_text(encoding='utf-8'))
    assert updated['custom'] == original['custom']
    assert updated['maps'][0]['difficulties'][0] == original['maps'][0]['difficulties'][0]
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', identifier)
    assert profile.difficulty_name == label and profile.minimum_scale == 1.5
    assert profile.floors == ('2nd', '1st') and profile.asset_status == 'ready'
    assert profile.template_versions == ('route_lhp_solo', 'clean')
    assert profile.asset_versions[1]['source_root'].endswith('clean-with-route/route_lhp_solo')
    assert len(read_bundle(root)['recognition'][profile.clean_root.relative_to(root).as_posix()]['targets']) == 4
    runtime_files(root)
    assert (root/'runtime/settings.json').read_bytes() == settings
    before = config_snapshot(root)
    assert sync.synchronize(root) == []
    assert config_snapshot(root) == before


@pytest.mark.parametrize('problem', ['missing_floor', 'dimensions', 'corrupt', 'extra_file'])
def test_incomplete_new_difficulty_is_reported_without_publication(package, problem):
    root, _, _ = package
    before = config_snapshot(root)
    base = add_assets(root)
    image = base/'clean-with-route/route_lhp_solo/layout_02/layout_02_2nd.png'
    if problem == 'missing_floor':
        image.unlink()
    elif problem == 'dimensions':
        write_png(image, np.zeros((20, 30, 3), np.uint8))
    elif problem == 'corrupt':
        image.write_bytes(b'not PNG')
    else:
        write_png(image.with_name('layout_02_merged.png'), np.zeros((80, 100, 3), np.uint8))
    assert any('暂未接入' in message for message in sync.synchronize(root))
    assert config_snapshot(root) == before
    assert not (root/sync.SYNC_DIRECTORY).exists()


def test_authoritative_route_directory_updates_paths_and_preserves_profile_metadata(package):
    root, original, image = package
    pending = difficulty('normal', '手动待配置', (), {'clean': '原版干净底图'})
    pending_root = pending.pop('clean_root')
    pending.update(asset_status='pending', layout_floors={}, asset_versions=[], template_versions=[])
    original['maps'][0]['difficulties'].append(pending)
    publish(root, original)
    write_png(root/pending_root/'layout_01/layout_01_1st.png', image)
    base = root/'assets/maps/a_bringer_of_doom/hard'
    write_png(base/'clean-with-route/custom/layout_01/layout_01_1st.png', image)
    write_png(base/'clean-with-route/personal/layout_01/layout_01_1st.png', image)
    sync.synchronize(root)
    actual = json.loads((root/sync.CATALOG_PATH).read_text(encoding='utf-8'))
    hard, normal = actual['maps'][0]['difficulties']
    assert normal['asset_status'] == 'ready' and normal['template_versions'] == ['clean']
    assert normal['name'] == pending['name'] and normal['layout_floors'] == {'layout_01': ['1st']}
    assert hard['name'] == '人工困难'
    assert MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard').minimum_scale == 1.25
    assert hard['asset_versions'][0] == dict(original['maps'][0]['difficulties'][0]['asset_versions'][0], name='原版干净底图')
    assert hard['asset_versions'][1]['name'] == 'custom'
    assert hard['asset_versions'][1]['source_root'].endswith('clean-with-route/custom')
    assert hard['asset_versions'][2]['id'] == 'personal'


def test_missing_legacy_version_and_renamed_folder_no_longer_block_startup(package):
    root, original, image = package
    old = original['maps'][0]['difficulties'][0]['asset_versions'][1]
    (root/old['source_root']/'layout_01/layout_01_1st.png').unlink()
    base = root/'assets/maps/a_bringer_of_doom/hard'
    write_png(base/'clean-with-route/renamed/layout_01/layout_01_1st.png', image)
    settings = root/'runtime/settings.json'
    before = settings.read_bytes()
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard')
    assert {v['id'] for v in profile.asset_versions} == {'clean', 'renamed'}
    assert profile.template_versions == ('renamed', 'clean')
    assert profile.minimum_scale == 1.25
    runtime_files(root)
    assert settings.read_bytes() == before


def test_empty_route_tree_keeps_clean_ready_and_complete_routes_join_later(package):
    root, _, image = package
    base = root/'assets/maps/a_bringer_of_doom/hard/clean-with-route'
    base.mkdir()
    scales = (root/sync.SCALE_PATH).read_bytes()
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard')
    assert profile.asset_status == 'ready' and profile.floors == ('1st',)
    assert profile.template_versions == ('clean',) and profile.clean_root.is_dir()
    assert {v['id'] for v in profile.asset_versions} == {'clean'}
    runtime_files(root)
    write_png(base/'good/layout_01/layout_01_1st.png', image)
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard')
    assert profile.asset_status == 'ready' and profile.template_versions == ('good', 'clean')
    assert (root/sync.SCALE_PATH).read_bytes() == scales
    runtime_files(root)


@pytest.mark.parametrize('missing', ['map', 'difficulty', 'clean', 'route', 'route_tree', 'floor'])
def test_deleted_directories_are_pruned_and_incomplete_materials_remain_pending(package, missing):
    root, _, _ = package
    base = add_assets(root)
    sync.synchronize(root)
    scale_path = root/sync.SCALE_PATH
    scale_data = json.loads(scale_path.read_bytes())
    key = 'a_bringer_of_doom/easy'
    scale_data['maps']['a_bringer_of_doom']['easy'].update(minimum_scale=1.6, layout_minimum_scales={'layout_02': 2.0})
    scale_path.write_text(json.dumps(scale_data), encoding='utf-8')
    sync.synchronize(root)
    scales = scale_path.read_bytes()
    personal = (root/'runtime/settings.json').read_bytes()
    targets = dict(map=base.parent, difficulty=base, clean=base/'clean',
                   route=base/'clean-with-route/route_lhp_solo', route_tree=base/'clean-with-route',
                   floor=base/'clean/layout_01/layout_01_1st.png')
    source = targets[missing]
    backup = root/'temporarily_removed'
    source.rename(backup)
    sync.synchronize(root)
    catalog = MapCatalog(root/sync.CATALOG_PATH)
    payload = read_bundle(root)
    if missing in ('map', 'difficulty'):
        assert ('a_bringer_of_doom', 'easy') not in catalog.profiles
        assert key not in payload['template']['groups']
        assert 'easy' not in json.loads(scale_path.read_bytes())['maps'].get('a_bringer_of_doom', {})
        if missing == 'map':
            assert catalog.map_names == {} and not catalog.profiles
            assert json.loads((root/sync.CATALOG_PATH).read_bytes())['maps'] == []
            assert json.loads(scale_path.read_bytes()) == {'maps': {}}
            assert not payload['recognition'] and not payload['template']['images']
    elif missing in ('route', 'route_tree'):
        profile = catalog.profile('a_bringer_of_doom', 'easy')
        assert profile.asset_status == 'ready' and profile.template_versions == ('clean',)
        assert set(profile.layout_floors) == {'layout_01', 'layout_02'}
        assert set(payload['template']['groups'][key]['layout_01']) == {'clean'}
        assert any('/easy/' in path for path in payload['recognition'])
        assert scale_path.read_bytes() == scales
    else:
        profile = catalog.profile('a_bringer_of_doom', 'easy')
        assert profile.asset_status == 'pending' and profile.floors == () and not profile.layout_floors
        assert not profile.template_versions and not profile.asset_versions and profile.clean_root is None
        definition = next(d for m in json.loads((root/sync.CATALOG_PATH).read_bytes())['maps']
                          for d in m['difficulties'] if d['id']=='easy')
        assert 'clean_root' not in definition
        assert payload['template']['groups'][key] == {}
        assert not any('/easy/' in path for path in payload['recognition'])
        assert scale_path.read_bytes() == scales
    assert (root/'runtime/settings.json').read_bytes() == personal
    runtime_files(root)
    saved = json.loads((root/sync.SYNC_DIRECTORY/'previous.json').read_bytes())
    assert set(saved) == set(sync.CONFIG_FILES)
    backup.rename(source)
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert profile.asset_status == 'ready' and set(profile.layout_floors) == {'layout_01', 'layout_02'}
    assert profile.minimum_scale_for_layout('layout_01') == (1.5 if missing in ('map', 'difficulty') else 1.6)
    assert profile.minimum_scale_for_layout('layout_02') == (1.5 if missing in ('map', 'difficulty') else 2.0)
    assert profile.template_versions == ('route_lhp_solo', 'clean')
    if missing not in ('map', 'difficulty'):
        assert scale_path.read_bytes() == scales
    assert (root/'runtime/settings.json').read_bytes() == personal
    runtime_files(root)


def test_map_with_its_last_difficulty_deleted_is_removed(package, monkeypatch, capsys):
    root, _, _ = package
    difficulty_path = root/'assets/maps/a_bringer_of_doom/hard'
    difficulty_path.rename(root/'removed_difficulty')
    sync.synchronize(root)
    assert not MapCatalog(root/sync.CATALOG_PATH).map_names
    assert json.loads((root/sync.SCALE_PATH).read_bytes()) == {'maps': {}}
    assert not read_bundle(root)['template']['groups']
    runtime_files(root)
    from src import __main__ as entry
    from tests.support.aligned_runtime import activate
    activate(root, monkeypatch)
    monkeypatch.setattr(entry.sys, 'argv', ['src', '--check'])
    entry.main()
    assert json.loads(capsys.readouterr().out)['profiles'] == []
    monkeypatch.setattr(entry.sys, 'argv', ['src', '--project-once', 'unused.png'])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2 and '暂无地图素材' in capsys.readouterr().err


def test_inaccessible_map_is_not_mistaken_for_deleted_directory(package, monkeypatch):
    root, _, _ = package
    map_path = root/'assets/maps/a_bringer_of_doom'
    before = config_snapshot(root)
    original = Path.stat
    def inaccessible(path, *args, **kwargs):
        if path == map_path:
            raise PermissionError('test permission failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', inaccessible)
    with pytest.raises(PermissionError):
        sync.synchronize(root)
    assert config_snapshot(root) == before


def test_deleted_layout_preserves_dormant_scale_and_removed_route_is_pruned(package):
    root, _, _ = package
    base = add_assets(root, route_name='路线甲')
    add_assets(root, route_name='路线乙')
    sync.synchronize(root)
    path = root/sync.SCALE_PATH
    data = json.loads(path.read_bytes())
    data['maps']['a_bringer_of_doom']['easy']['layout_minimum_scales'] = {'layout_02': 2.0}
    path.write_text(json.dumps(data), encoding='utf-8')
    sync.synchronize(root)
    scales = path.read_bytes()
    moved = []
    for index, directory in enumerate((base/'clean', base/'clean-with-route/路线甲', base/'clean-with-route/路线乙')):
        source = directory/'layout_02'
        backup = root/f'layout_backup_{index}'
        source.rename(backup)
        moved.append((backup, source))
    (base/'clean-with-route/路线乙').rename(root/'route_backup')
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert profile.asset_status == 'ready' and set(profile.layout_floors) == {'layout_01'}
    assert profile.template_versions == ('路线甲', 'clean')
    assert profile.layout_minimum_scales == {'layout_02': 2.0} and path.read_bytes() == scales
    runtime_files(root)
    (root/'route_backup').rename(base/'clean-with-route/路线乙')
    for backup, source in moved:
        backup.rename(source)
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert profile.asset_status == 'ready' and set(profile.layout_floors) == {'layout_01', 'layout_02'}
    assert profile.minimum_scale_for_layout('layout_02') == 2.0 and path.read_bytes() == scales
    assert set(profile.template_versions) == {'路线甲', '路线乙', 'clean'}
    runtime_files(root)


def test_same_version_updates_missing_old_path_then_removed_version_is_pruned(package):
    root, original, image = package
    base = root/'assets/maps/a_bringer_of_doom/hard'
    old = original['maps'][0]['difficulties'][0]['asset_versions'][1]
    (root/old['source_root']/'layout_01/layout_01_1st.png').unlink()
    current = base/'clean-with-route/custom/layout_01/layout_01_1st.png'
    extra = base/'clean-with-route/extra/layout_01/layout_01_1st.png'
    write_png(current, image)
    write_png(extra, image)
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard')
    assert profile.asset_versions[1]['source_root'].endswith('clean-with-route/custom')
    extra.unlink()
    extra.parent.rmdir()
    extra.parent.parent.rmdir()
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'hard')
    assert profile.template_versions == ('custom', 'clean')
    assert {v['id'] for v in profile.asset_versions} == {'clean', 'custom'}
    runtime_files(root)


def test_added_layout_and_replaced_image_refresh_manifest(package):
    root, _, image = package
    add_assets(root)
    sync.synchronize(root)
    before = (root/sync.BUNDLE_PATH).read_bytes()
    add_assets(root, layouts=('layout_03',))
    sync.synchronize(root)
    assert (root/sync.BUNDLE_PATH).read_bytes() != before
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert MapCatalog(root/sync.CATALOG_PATH).layouts(profile) == ['layout_01', 'layout_02', 'layout_03']
    before = (root/sync.BUNDLE_PATH).read_bytes()
    write_png(profile.clean_root/'layout_03/layout_03_1st.png', image+1)
    sync.synchronize(root)
    assert (root/sync.BUNDLE_PATH).read_bytes() != before
    runtime_files(root)


def test_invalid_replacement_does_not_rewrite_configuration(package):
    root, _, _ = package
    before = config_snapshot(root)
    (root/'assets/maps/a_bringer_of_doom/hard/clean/layout_01/layout_01_1st.png').write_bytes(b'bad')
    with pytest.raises(OSError):
        sync.synchronize(root)
    assert config_snapshot(root) == before


def test_existing_unreferenced_route_preview_is_preserved(package):
    root, original, image = package
    before = config_snapshot(root)
    version = original['maps'][0]['difficulties'][0]['asset_versions'][1]
    preview = root/version['source_root']/'layout_01/layout_01_merged.png'
    write_png(preview, image)
    contents = preview.read_bytes()
    assert sync.synchronize(root) == []
    assert preview.read_bytes() == contents and config_snapshot(root) == before


def test_layouts_with_different_floor_counts_publish_and_match_only_actual_floors(package, monkeypatch):
    from tests.support.aligned_runtime import activate
    from src.registration.template_matching import floors, direct
    from src.layout_retrieval import create_layout_retriever
    root, _, image = package
    add_assets(root, 'easy', layouts=('layout_01',), floors=('1st',))
    add_assets(root, 'easy', layouts=('layout_02',), floors=('1st', '2nd'))
    sync.synchronize(root)
    catalog = MapCatalog(root/sync.CATALOG_PATH)
    profile = catalog.profile('a_bringer_of_doom', 'easy')
    assert profile.floors == ('2nd', '1st')
    assert profile.floors_for_layout('layout_01') == ('1st',)
    assert profile.floors_for_layout('layout_02') == ('2nd', '1st')
    activate(root, monkeypatch)
    assert set(floors.sources(profile.key, 'layout_01', 'clean')[0]) == {'1st'}
    assert set(floors.FrozenSourceSnapshot(profile.key, 'layout_01', 'route_lhp_solo').entries) == {'1st'}
    retriever = create_layout_retriever('pure_structure', profile.clean_root, floors=profile.floors, cache_root=root/'features')
    try:
        retriever.prepare()
        assert {(v['layout'], v['floor']) for v in retriever._targets} == {
            ('layout_01', '1st'), ('layout_02', '1st'), ('layout_02', '2nd')}
        result, _, _, _, _ = direct.project_original(image[10:50,10:60], profile.key,
            'layout_01', .5, variant_id='route_lhp_solo')
        assert result.accepted and result.validation['template']['candidate_floor'] == '1st'
        assert len(result.validation['fixed_candidates']) == 1
    finally:
        retriever.close()
        floors._close_fixed_session()


@pytest.mark.parametrize('change', ['new_layout', 'pixels', 'catalog'])
def test_input_changes_during_audit_never_publish(package, monkeypatch, change):
    root, _, _ = package
    add_assets(root)
    before = config_snapshot(root)
    generate = sync.generate
    def changing(*args, **kwargs):
        result = generate(*args, **kwargs)
        if change == 'new_layout':
            add_assets(root, layouts=('layout_03',))
        elif change == 'pixels':
            path = root/'assets/maps/a_bringer_of_doom/easy/clean/layout_01/layout_01_1st.png'
            write_png(path, np.zeros((80, 100, 3), np.uint8))
        else:
            path = root/sync.CATALOG_PATH
            path.write_bytes(path.read_bytes()+b' ')
        return result
    monkeypatch.setattr(sync, 'generate', changing)
    with pytest.raises(ValueError, match='变化'):
        sync.synchronize(root)
    for name in (sync.SCALE_PATH, sync.BUNDLE_PATH):
        assert (root/name).read_bytes() == before[name][0]


@pytest.mark.parametrize('interrupted', [False, True])
def test_failed_or_interrupted_publication_restores_complete_batch(package, monkeypatch, interrupted):
    root, _, _ = package
    original = {name: raw for name, (raw, _) in config_snapshot(root).items()}
    replace = sync.os.replace
    failed = False
    def fail_once(source, destination):
        nonlocal failed
        if not failed and Path(destination) == root/sync.SCALE_PATH:
            failed = True
            if interrupted:
                raise KeyboardInterrupt()
            raise OSError('disk full')
        return replace(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(sync.os, 'replace', fail_once)
        with pytest.raises(KeyboardInterrupt if interrupted else OSError):
            sync.publish(root, {name: b'candidate' for name in sync.CONFIG_FILES})
    if interrupted:
        assert (root/sync.SYNC_DIRECTORY/'pending.json').exists()
        sync.recover(root)
    assert {name: (root/name).read_bytes() for name in sync.CONFIG_FILES} == original
    assert not (root/sync.SYNC_DIRECTORY/'pending.json').exists()
    assert not list((root/'configs').glob('*.tmp'))


def test_corrupt_recovery_backup_cannot_replace_configs(package):
    root, _, _ = package
    before = config_snapshot(root)
    directory = root/sync.SYNC_DIRECTORY
    directory.mkdir(parents=True)
    (directory/'previous.json').write_bytes(b'bad backup')
    (directory/'pending.json').write_text(json.dumps(dict(backup_sha256='bad')), encoding='utf-8')
    with pytest.raises(ValueError, match='指纹'):
        sync.recover(root)
    assert config_snapshot(root) == before


def test_new_assets_cannot_escape_project_through_directory_link(package, tmp_path):
    from tests.support.directory_links import create_directory_link, remove_directory_link
    root, _, _ = package
    outside = tmp_path.parent/(tmp_path.name+'_outside')
    outside.mkdir()
    link = root/'assets/maps/a_bringer_of_doom/easy'
    before = config_snapshot(root)
    create_directory_link(link, outside)
    try:
        with pytest.raises(ValueError):
            sync.synchronize(root)
        assert config_snapshot(root) == before
    finally:
        remove_directory_link(link, root=root)


@pytest.mark.parametrize('mode', ['--check', '--project-once'])
def test_read_only_commands_do_not_start_synchronization(monkeypatch, mode):
    from src import __main__ as entry
    monkeypatch.setattr(sync, 'synchronize', lambda *_: pytest.fail('read-only command published assets'))
    calls = []
    monkeypatch.setattr(entry, '_run', lambda *args: calls.append(args))
    monkeypatch.setattr(entry.sys, 'argv', ['src', mode]+(['unused.png'] if mode == '--project-once' else []))
    entry.main()
    assert len(calls) == 1


def test_normal_start_synchronizes_before_loading_gui_and_existing_instance_does_not(package, monkeypatch):
    from src import __main__ as entry
    root, _, _ = package
    add_assets(root, 'novice')
    monkeypatch.setattr(entry, 'PROJECT_ROOT', root)
    from src.startup import prepare_startup_recognition
    def prepare(root, **selection):
        return prepare_startup_recognition(root, sync.synchronize(root),
            progress=lambda *_:None, cancelled=lambda:False, **selection)
    monkeypatch.setattr(entry, '_synchronize_for_gui', prepare)
    monkeypatch.setattr(entry.sys, 'argv', ['src'])
    monkeypatch.setattr(entry, '_acquire_app_mutex', lambda: (None, True))
    monkeypatch.setattr(entry, '_run', lambda *_: pytest.fail('Second instance entered GUI'))
    before = config_snapshot(root)
    entry.main()
    assert config_snapshot(root) == before
    monkeypatch.setattr(entry, '_acquire_app_mutex', lambda: (None, False))
    called = []
    def gui(_args, _parser, state):
        called.append(MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'novice'))
        assert state.messages
        runtime_files(root)
    monkeypatch.setattr(entry, '_run', gui)
    entry.main()
    assert called[0].minimum_scale == 1.5


@pytest.mark.parametrize('field', ['minimum_scale', 'layout_minimum_scales'])
def test_catalog_rejects_duplicate_scale_configuration(package, field):
    root, catalog, _ = package
    catalog['maps'][0]['difficulties'][0][field] = 1.5 if field == 'minimum_scale' else {}
    (root/sync.CATALOG_PATH).write_text(json.dumps(catalog), encoding='utf-8')
    before = config_snapshot(root)
    with pytest.raises(ValueError, match='尺度只能在 configs/scale_config.json'):
        sync.synchronize(root)
    assert config_snapshot(root) == before


def test_preconfigured_new_difficulty_keeps_its_default_and_override(package):
    root, _, _ = package
    add_assets(root)
    path = root/sync.SCALE_PATH
    scales = json.loads(path.read_text())
    scales['maps']['a_bringer_of_doom']['easy'] = dict(
        minimum_scale=1.6, layout_minimum_scales={'layout_02': 2.0})
    raw = json.dumps(scales, indent=4).encode()
    path.write_bytes(raw)
    sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('a_bringer_of_doom', 'easy')
    assert profile.minimum_scale_for_layout('layout_01') == 1.6
    assert profile.minimum_scale_for_layout('layout_02') == 2.0
    assert path.read_bytes() == raw
    assert 'minimum_scale' not in json.loads((root/sync.CATALOG_PATH).read_text())['maps'][0]['difficulties'][1]


def test_missing_existing_scale_is_not_replaced_with_default(package):
    root, _, _ = package
    path = root/sync.SCALE_PATH
    path.write_bytes(b'{"maps": {"a_bringer_of_doom": {}}}')
    before = config_snapshot(root)
    with pytest.raises(ValueError, match='缺少.*hard'):
        sync.synchronize(root)
    assert config_snapshot(root) == before


def test_scale_edit_during_sync_is_preserved_without_partial_publication(package, monkeypatch):
    root, _, _ = package
    add_assets(root)
    before = config_snapshot(root)
    generate = sync.generate
    edited = before[sync.SCALE_PATH][0]+b' '
    def changing(*args, **kwargs):
        result = generate(*args, **kwargs)
        (root/sync.SCALE_PATH).write_bytes(edited)
        return result
    monkeypatch.setattr(sync, 'generate', changing)
    with pytest.raises(ValueError, match='变化'):
        sync.synchronize(root)
    assert (root/sync.SCALE_PATH).read_bytes() == edited
    for name in (sync.CATALOG_PATH, sync.BUNDLE_PATH):
        assert (root/name).read_bytes() == before[name][0]


@pytest.mark.parametrize('empty_route_tree', [False, True])
def test_new_map_with_only_clean_is_discovered_and_later_routes_are_optional(package, empty_route_tree):
    root, _, image = package
    base = root/'assets/maps/仅底图地图/normal'
    write_png(base/'clean/layout_01/layout_01_b4.png', image)
    if empty_route_tree:
        (base/'clean-with-route').mkdir()
    personal = (root/'runtime/settings.json').read_bytes()
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('仅底图地图','normal')
    assert profile.asset_status == 'ready' and profile.minimum_scale == 1.5
    assert profile.floors == ('b4',) and profile.layout_floors == {'layout_01': ('b4',)}
    assert profile.version_names == {'clean': '原版干净底图'} and profile.template_versions == ('clean',)
    assert set(read_bundle(root)['template']['groups'][profile.key]['layout_01']) == {'clean'}
    runtime_files(root)
    assert not sync.synchronize(root)
    # With no routes, every subsequently added clean layout is available by itself.
    write_png(base/'clean/layout_02/layout_02_1st.png', image)
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('仅底图地图','normal')
    assert profile.floors_for_layout('layout_02') == ('1st',)
    for layout, floor in [('layout_01','b4'),('layout_02','1st')]:
        write_png(base/f'clean-with-route/后加路线/{layout}/{layout}_{floor}.png', image)
    assert sync.synchronize(root)
    profile = MapCatalog(root/sync.CATALOG_PATH).profile('仅底图地图','normal')
    assert profile.template_versions == ('后加路线','clean') and profile.minimum_scale == 1.5
    assert (root/'runtime/settings.json').read_bytes() == personal
    runtime_files(root)
