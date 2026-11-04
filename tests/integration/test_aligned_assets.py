"""Same-canvas publishing and clean-only registration, independent of game labels."""
import json
import shutil

import cv2
import numpy as np
import pytest

from src.paths import PROJECT_ROOT
from tools.assets.imports.export_aligned_assets import generate


@pytest.fixture
def aligned_package(tmp_path):
    for name in ('configs/app_defaults.json', 'src/registration/template_matching/provenance.json',
                 'assets/app_ui/home_wallpaper.jpg', 'assets/app_ui/icons/check.svg',
                 'assets/layout_retrieval/icons/query.png', 'assets/layout_retrieval/icons/clean_09.png',
                 'assets/layout_retrieval/icons/clean_15.png'):
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/name, target)
    for path in (PROJECT_ROOT/'src/registration/template_matching').glob('*.py'):
        shutil.copyfile(path, tmp_path/'src/registration/template_matching'/path.name)
    clean = 'assets/maps/sample/hard/clean'
    route = 'assets/maps/sample/hard/routes/custom'
    definition = dict(id='hard', name='困难', floors=['1st'], asset_contract='aligned_floor_images',
        asset_status='ready', clean_root=clean, calibration_root='maintenance/sample',
        template_versions=['clean', 'custom'], asset_versions=[
            dict(id='clean', name='底图', kind='clean', source_root=clean),
            dict(id='custom', name='自定义路线', kind='route', source_root=route)])
    catalog = dict(schema_version=2, maps=[dict(id='sample', name='测试地图', difficulties=[definition])])
    (tmp_path/'configs/map_catalog.json').write_text(json.dumps(catalog), encoding='utf-8')
    (tmp_path/'configs/scale_config.json').write_text(json.dumps({'maps': {'sample': {'hard': dict(
        minimum_scale=1., layout_minimum_scales={})}}}), encoding='utf-8')
    image = np.zeros((220, 260, 3), np.uint8)
    rng = np.random.default_rng(71)
    for _ in range(22):
        x, y = rng.integers(15, 160, 2)
        cv2.rectangle(image, (int(x), int(y)), (int(x)+30, int(y)+25), (160,160,160), 3)
    overlay = image.copy()
    cv2.line(overlay, (30,30), (190,190), (0,0,255), 8)
    paths = []
    for directory, pixels in ((clean,image), (route,overlay)):
        path = tmp_path/directory/'layout_01/layout_01_1st.png'
        path.parent.mkdir(parents=True)
        cv2.imencode('.png', pixels)[1].tofile(str(path))
        paths.append(path)
    return tmp_path, paths, image


def publish(root):
    data, scales = generate(root)
    (root/'configs/runtime_resources.json').write_text(json.dumps(data), encoding='utf-8')
    (root/'configs/scale_config.json').write_bytes(scales)
    return data


def test_publish_checks_dimensions_and_one_shared_scale(aligned_package):
    root, paths, _ = aligned_package
    data = publish(root)
    scales = json.loads((root/'configs/scale_config.json').read_text())
    assert scales['maps']['sample']['hard'] == dict(minimum_scale=1., layout_minimum_scales={})
    assert all('scales' not in record for record in next(iter(data['payload']['recognition'].values()))['targets'].values())
    from src.runtime_resources import template_config
    assert template_config(root)['floor_variants']['sample/hard']['layout_01']['clean']['1st']['saved_scale'] == 1.
    cv2.imencode('.png', np.zeros((10,10,3), np.uint8))[1].tofile(str(paths[1]))
    with pytest.raises(ValueError, match='尺寸'):
        generate(root)


@pytest.mark.parametrize('overrides', [None, [], {'layout_01': 0}, {'layout_01': -1},
    {'layout_01': True}, {'layout_01': '2.0'}, {'layout_01': float('nan')},
    {'layout_01': float('inf')}, {'layout_bad': 2.0}, {'../layout_01': 2.0}])
def test_invalid_layout_minimum_scales_are_rejected(aligned_package, overrides):
    root, _, _ = aligned_package
    path = root/'configs/scale_config.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['maps']['sample']['hard']['layout_minimum_scales'] = overrides
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='layout_minimum_scales'):
        generate(root)


def test_missing_layout_override_is_preserved_without_creating_a_runtime_layout(aligned_package):
    root, _, _ = aligned_package
    path = root/'configs/scale_config.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['maps']['sample']['hard']['layout_minimum_scales'] = {'layout_99': 2.0}
    path.write_text(json.dumps(data), encoding='utf-8')
    from src.runtime_resources import template_config
    before = path.read_bytes()
    publish(root)
    assert path.read_bytes() == before
    records = template_config(root)['floor_variants']['sample/hard']
    assert set(records) == {'layout_01'}
    assert records['layout_01']['clean']['1st']['saved_scale'] == 1.


def test_retired_catalog_schema_cannot_be_published(aligned_package):
    root,_,_=aligned_package
    path=root/'configs/map_catalog.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    data['schema_version']=1
    path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='地图目录配置格式'):
        generate(root)


def test_mixed_layout_scales_are_shared_by_recognition_and_projection(tmp_path, monkeypatch):
    from tests.support.aligned_runtime import activate, difficulty, publish as publish_aligned, write_png
    from src.app_catalog import MapCatalog
    from src.layout_retrieval import create_layout_retriever
    from src.registration.template_matching import direct, floors
    from src.runtime_resources import read_bundle, template_config, recognition_profile
    definition = difficulty('hard', '困难', ('2nd', '1st'), {'clean': '底图', '路线甲': '路线甲'})
    shots = {}
    for number, scale in [(1, 1.5), (2, 2.0)]:
        layout = f'layout_{number:02}'
        for floor_index, floor in enumerate(definition['floors']):
            shot = np.full((240, 280, 3), (41, 33, 28), np.uint8)
            rng = np.random.default_rng(number*71+floor_index)
            for _ in range(20):
                x, y = (int(v) for v in rng.integers(20, 165, 2))
                cv2.rectangle(shot, (x,y), (x+45,y+35), (100,100,100), -1)
                cv2.rectangle(shot, (x,y), (x+45,y+35), (175,175,175), 3)
            image = cv2.resize(shot, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            route = image.copy()
            cv2.line(route, (25,25), (image.shape[1]-25,image.shape[0]-25), (0,0,255), 5)
            for version in definition['asset_versions']:
                write_png(tmp_path/version['source_root']/layout/f'{layout}_{floor}.png',
                          image if version['kind'] == 'clean' else route)
            shots[layout, floor] = shot
    publish_aligned(tmp_path, dict(schema_version=2, maps=[dict(
        id='a_bringer_of_doom', name='测试地图', difficulties=[definition])]), scales={
            'maps': {'a_bringer_of_doom': {'hard': dict(minimum_scale=1.5, layout_minimum_scales={'layout_02': 2.0})}}})
    activate(tmp_path, monkeypatch)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    catalog = MapCatalog(tmp_path/'configs/map_catalog.json')
    profile = catalog.profile('a_bringer_of_doom', 'hard')
    bundle = read_bundle(tmp_path)
    recognition = recognition_profile(profile.clean_root)['targets']
    assert all('scales' not in item for item in bundle['recognition'][definition['clean_root']]['targets'].values())
    records = template_config(tmp_path)['floor_variants'][profile.key]
    retriever = create_layout_retriever('pure_structure', profile.clean_root,
        floors=profile.floors, cache_root=tmp_path/'features')
    try:
        for (layout, floor), shot in shots.items():
            scale = 1.5 if layout == 'layout_01' else 2.0
            assert profile.minimum_scale_for_layout(layout) == scale
            assert recognition[f'{layout}/{layout}_{floor}.png']['scales'] == [scale]
            assert all(version[floor]['saved_scale'] == scale for version in records[layout].values())
            recognized = retriever.recognize(shot, shot.shape[1::-1])
            assert recognized.candidates[0].layout == layout
            assert recognized.candidates[0].details['scale'] == scale
            assert recognized.candidates[0].details['configured_scales'] == [scale]
            screenshot = shot[20:210,20:260].copy()
            clean = direct.project_original(screenshot, profile.key, layout, .5, variant_id='clean')
            route = direct.project_original(screenshot, profile.key, layout, .5, variant_id='路线甲')
            assert clean[0].accepted and route[0].accepted
            assert route[0].validation['template']['candidate_floor'] == floor
            assert route[0].validation['template']['saved_scale'] == scale
            np.testing.assert_allclose(route[0].screenshot_to_original,
                [[scale,0,20*scale], [0,scale,20*scale], [0,0,1]], atol=4)
            np.testing.assert_array_equal(clean[0].original_to_screen, route[0].original_to_screen)
    finally:
        retriever.close()
        floors._close_fixed_session()


def test_route_pixels_do_not_change_registration(aligned_package, monkeypatch):
    root, paths, image = aligned_package
    publish(root)
    from src import runtime_resources
    from src.registration.template_matching import floors, direct
    monkeypatch.setattr(runtime_resources, 'PROJECT_ROOT', root)
    monkeypatch.setattr(floors, 'PROJECT_ROOT', root)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    screenshot = image[20:190,20:210].copy()
    clean, _, clean_layer, _, _ = direct.project_original(screenshot,'sample/hard','layout_01',1.,variant_id='clean')
    route, _, route_layer, _, _ = direct.project_original(screenshot,'sample/hard','layout_01',1.,variant_id='custom')
    assert clean.accepted and route.accepted
    np.testing.assert_array_equal(clean.original_to_screen, route.original_to_screen)
    assert not np.array_equal(clean_layer, route_layer)
    assert route.validation['matching_source_image'].endswith('/clean/layout_01/layout_01_1st.png')
    # The process retains verified source bytes even after files change.
    paths[1].write_bytes(b'changed')
    again, _, layer, _, _ = direct.project_original(screenshot,'sample/hard','layout_01',1.,variant_id='custom')
    assert again.accepted
    np.testing.assert_array_equal(layer, route_layer)


def test_pending_profiles_have_no_runtime_layouts(aligned_package):
    from src.app_catalog import MapCatalog
    root, _, _ = aligned_package
    path = root/'configs/map_catalog.json'
    data = json.loads(path.read_bytes())
    definition = data['maps'][0]['difficulties'][0]
    definition.update(asset_status='pending', floors=[], layout_floors={})
    path.write_text(json.dumps(data), encoding='utf-8')
    publish(root)
    catalog = MapCatalog(path)
    profile = catalog.profile('sample','hard')
    assert profile.asset_status == 'pending' and profile.floors == ()
    assert catalog.layouts(profile) == []


def test_workflow_default_route_comes_from_declared_versions(aligned_package, monkeypatch):
    from tests.support.aligned_runtime import activate
    from src.app_catalog import MapCatalog
    from src.projection_workflow import process_screenshot
    from src.registration.template_matching import direct, floors
    root, _, image = aligned_package
    publish(root)
    activate(root, monkeypatch)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    catalog = MapCatalog(root/'configs/map_catalog.json')
    profile = catalog.profile('sample', 'hard')
    try:
        result = process_screenshot(image[20:190,20:210].copy(), catalog=catalog,
            profile=profile, opacity=.5, layout_override='layout_01')
        assert result.accepted, result.reasons
        assert result.variant_id == profile.template_versions[0] == 'clean'
        assert result.variant_source.startswith('底图')
    finally:
        floors._close_fixed_session()


@pytest.mark.parametrize('default', ['clean', 'custom'])
def test_shared_projection_default_uses_frozen_configuration(aligned_package, monkeypatch, default):
    from tests.support.aligned_runtime import activate
    from src.registration.template_matching import direct, floors
    root, paths, image = aligned_package
    path = root/'configs/map_catalog.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    definition = data['maps'][0]['difficulties'][0]
    definition['template_versions'] = [default, 'custom' if default == 'clean' else 'clean']
    path.write_text(json.dumps(data), encoding='utf-8')
    publish(root)
    activate(root, monkeypatch)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    shot = image[20:190,20:210].copy()
    try:
        implicit = direct.project_original(shot, 'sample/hard', 'layout_01', .5)
        explicit = direct.project_original(shot, 'sample/hard', 'layout_01', .5, variant_id=default)
        assert implicit[0].accepted and explicit[0].accepted
        assert implicit[1] == paths[0 if default == 'clean' else 1]
        np.testing.assert_array_equal(implicit[0].original_to_screen, explicit[0].original_to_screen)
        np.testing.assert_array_equal(implicit[2], explicit[2])
        # Runtime resources remain fixed until restart, including the default selection.
        definition['template_versions'].reverse()
        path.write_text(json.dumps(data), encoding='utf-8')
        retained = direct.project_original(shot, 'sample/hard', 'layout_01', .5)
        assert retained[0].accepted and retained[1] == implicit[1]
        np.testing.assert_array_equal(retained[2], implicit[2])
    finally:
        floors._close_fixed_session()


@pytest.mark.parametrize('pending', [False, True])
def test_shared_projection_unavailable_profile_does_not_use_legacy_default(aligned_package, monkeypatch, pending):
    from tests.support.aligned_runtime import activate
    from src.registration.template_matching import direct, floors
    root, _, image = aligned_package
    if pending:
        path = root/'configs/map_catalog.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['maps'][0]['difficulties'][0]['asset_status'] = 'pending'
        path.write_text(json.dumps(data), encoding='utf-8')
    publish(root)
    activate(root, monkeypatch)
    try:
        result, source, layer, _, _ = direct.project_original(
            image, 'sample/hard' if pending else 'missing/hard', 'layout_01', .5)
        assert not result.accepted and source is None and layer is None
        assert any(('资产待配置' if pending else '未配置地图难度') in reason for reason in result.rejection_reasons)
    finally:
        floors._close_fixed_session()


def test_stale_code_provenance_refuses_export(aligned_package):
    root, _, _ = aligned_package
    path = root/'src/registration/template_matching/direct.py'
    path.write_bytes(path.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError, match='源码指纹'):
        generate(root)


def test_offline_scale_trial_replays_known_crop(aligned_package):
    from tools.assets.calibration.calibrate_minimum_scale import trial
    _, paths, image = aligned_package
    candidate, preview = trial(paths[0], image[20:190,20:210].copy(), [1.])
    assert candidate['scale'] == 1.
    assert candidate['location'] == [20,20]
    assert candidate['verified'] is False
    assert preview.shape == (170,190,3)


def test_offline_scale_trial_excludes_dark_gray_canvas(aligned_package):
    from tools.assets.calibration.calibrate_minimum_scale import trial
    _, paths, image = aligned_package
    image[np.all(image == 0, axis=2)] = (41,33,28)
    cv2.imencode('.png', image)[1].tofile(str(paths[0]))
    candidate, _ = trial(paths[0], image[20:190,20:210].copy(), [1.])
    assert candidate['location'] == [20,20]
    assert candidate['map_threshold'] == 56


def test_new_recognition_uses_only_minimum_and_ignores_old_index(aligned_package, monkeypatch):
    root, paths, image = aligned_package
    publish(root)
    from src import runtime_resources
    from src.layout_retrieval import create_layout_retriever
    monkeypatch.setattr(runtime_resources, 'PROJECT_ROOT', root)
    clean_root = paths[0].parent.parent
    (clean_root.parent/'recognition').mkdir()
    retriever = create_layout_retriever('pure_structure', clean_root, floors=('1st',), workers=1)
    try:
        assert retriever.fixed_resources
        result = retriever.recognize(image, image.shape[1::-1])
        assert result.candidates[0].layout == 'layout_01'
        assert len(retriever._targets) == 1
        assert tuple(retriever._targets[0]['scales']) == (1.,)
    finally:
        retriever.close()
