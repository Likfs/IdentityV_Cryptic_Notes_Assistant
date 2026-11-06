"""A route covers whole layouts, while recognition always covers every clean layout."""
import shutil

import cv2
import numpy as np
import pytest

from src import asset_sync
from src.aligned_assets import generate
from src.app_catalog import MapCatalog, effective_template_variant
from src.layout_retrieval import create_layout_retriever
from src.projection_workflow import process_screenshot
from src.registration.template_matching import direct, floors
from src.runtime_resources import read_bundle, runtime_files
from tests.support.aligned_runtime import activate, build_multifloor_package, write_png


@pytest.fixture
def partial_package(tmp_path, monkeypatch):
    root = build_multifloor_package(tmp_path/'package', layout_floors={
        'layout_01': ('1st',), 'layout_02': ('b1', 'b2', 'b3')})
    activate(root, monkeypatch)
    monkeypatch.setattr(direct, 'PROJECT_ROOT', root)
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    yield root
    floors._close_fixed_session()


def profile():
    return MapCatalog().profile('phosphorescence_disease', 'hard')


def route_root(root):
    return root/'assets/maps/phosphorescence_disease/hard/clean-with-route/测试路线'


def shot(item, layout, floor):
    image = cv2.imdecode(np.fromfile(str(item.clean_root/layout/f'{layout}_{floor}.png'), np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(image[45:315, 60:390], (220, 180), interpolation=cv2.INTER_AREA)


def test_partial_routes_publish_all_clean_targets_and_register_with_identical_geometry(partial_package):
    root = partial_package
    item = profile()
    covered_shot, uncovered_shot = shot(item, 'layout_01', '1st'), shot(item, 'layout_02', 'b3')
    covered_before = direct.project_original(covered_shot, item.key, 'layout_01', .5, variant_id='测试路线')
    clean_before = direct.project_original(uncovered_shot, item.key, 'layout_02', .5, variant_id='clean')
    assert covered_before[0].accepted and clean_before[0].accepted
    floors._close_fixed_session()
    scales = (root/'configs/scale_config.json').read_bytes()
    settings = root/'runtime/settings.json'
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_bytes(b'{"personal":"keep"}')
    shutil.rmtree(route_root(root)/'layout_02')
    assert asset_sync.synchronize(root)
    assert profile().asset_status == 'ready'
    assert profile().template_versions == ('测试路线', 'clean')
    assert direct.available_variants(item.key, 'layout_01') == ['测试路线', 'clean']
    assert direct.available_variants(item.key, 'layout_02') == ['clean']
    bundle = read_bundle(root)
    assert set(bundle['template']['groups'][item.key]) == {'layout_01', 'layout_02'}
    assert len(bundle['recognition'][item.clean_root.relative_to(root).as_posix()]['targets']) == 4
    assert len(bundle['template']['images']) == 5
    assert {v['variant_id'] for v in bundle['route_status'][item.key]['layout_02']} == {'clean'}
    summary = direct.check_original_resources(item.key, ['layout_01', 'layout_02'])
    assert summary['测试路线']['layouts'] == 1 and summary['clean']['layouts'] == 2
    assert not any(record['errors'] for record in summary.values())
    runtime_files(root)
    assert not asset_sync.synchronize(root)
    assert (root/'configs/scale_config.json').read_bytes() == scales
    assert settings.read_bytes() == b'{"personal":"keep"}'

    covered_after = direct.project_original(covered_shot, item.key, 'layout_01', .5)
    clean_after = direct.project_original(uncovered_shot, item.key, 'layout_02', .5)
    for before, after in [(covered_before, covered_after), (clean_before, clean_after)]:
        assert after[0].accepted, after[0].rejection_reasons
        assert before[0].validation == after[0].validation
        np.testing.assert_array_equal(before[0].original_to_screen, after[0].original_to_screen)
        np.testing.assert_array_equal(before[2], after[2])
    result = process_screenshot(uncovered_shot, catalog=MapCatalog(), profile=profile(),
        layout_override='layout_02', opacity=.5, registration_backend='template')
    assert result.accepted and result.variant_id == 'clean' and result.floor == 'b3'
    refused = direct.project_original(uncovered_shot, item.key, 'layout_02', .5, variant_id='测试路线')[0]
    assert not refused.accepted and '不包含路线版本' in refused.rejection_reasons[0]
    selected = MapCatalog().selected_variant(profile(), 'layout_02', {item.key: '测试路线'})
    assert selected.variant_id == 'clean'
    assert effective_template_variant('测试路线', ['另一条路线', 'clean']) == 'clean'
    clean_path = item.clean_root/'layout_02/layout_02_b3.png'
    full_clean = cv2.imdecode(np.fromfile(str(clean_path), np.uint8), cv2.IMREAD_COLOR)
    query = cv2.resize(full_clean, (300,240), interpolation=cv2.INTER_AREA)
    retriever = create_layout_retriever('pure_structure', item.clean_root, floors=item.floors,
                                      cache_root=root/'features')
    try:
        identified = retriever.recognize(query, query.shape[1::-1])
        assert identified.candidates, identified.diagnostics()
        assert identified.candidates[0].layout == 'layout_02'
        assert identified.candidates[0].details['floor'] == 'b3'
    finally:
        retriever.close()


@pytest.mark.parametrize('problem', ['missing_floor', 'empty_layout', 'extra_floor', 'unknown_layout', 'dimensions'])
def test_provided_layouts_must_have_exact_clean_floors(partial_package, problem):
    root = partial_package
    path = route_root(root)/'layout_02/layout_02_b3.png'
    if problem == 'missing_floor':
        path.unlink()
    elif problem == 'empty_layout':
        for image in path.parent.iterdir():
            image.unlink()
    elif problem == 'extra_floor':
        shutil.copyfile(path, path.with_name('layout_02_2nd.png'))
    elif problem == 'unknown_layout':
        write_png(route_root(root)/'layout_03/layout_03_b3.png', np.full((360,450,3), 100, np.uint8))
    else:
        write_png(path, np.full((20,30,3), 100, np.uint8))
    with pytest.raises(ValueError):
        generate(root)
    before = {name: (root/name).read_bytes() for name in asset_sync.CONFIG_FILES}
    if problem in ('missing_floor', 'empty_layout', 'extra_floor'):
        asset_sync.synchronize(root)
        assert profile().asset_status == 'pending'
        assert not read_bundle(root)['template']['groups'][profile().key]
    else:
        with pytest.raises(ValueError):
            asset_sync.synchronize(root)
        assert {name: (root/name).read_bytes() for name in asset_sync.CONFIG_FILES} == before


def test_layout_coverage_refreshes_after_addition_deletion_and_restore(partial_package):
    root = partial_package
    assert direct.available_variants(profile().key, 'layout_02') == ['测试路线', 'clean']
    route = route_root(root)/'layout_02'
    saved = {p.name: p.read_bytes() for p in route.iterdir()}
    shutil.rmtree(route)
    asset_sync.synchronize(root)
    # Offline audit reads current publication even while runtime retains its old session.
    assert direct.check_original_resources(profile().key, ['layout_01', 'layout_02'])['测试路线']['layouts'] == 1
    assert direct.available_variants(profile().key, 'layout_02') == ['测试路线', 'clean']
    floors._close_fixed_session()
    assert direct.available_variants(profile().key, 'layout_02') == ['clean']
    floors._close_fixed_session()
    for name, raw in saved.items():
        route.mkdir(exist_ok=True)
        (route/name).write_bytes(raw)
    asset_sync.synchronize(root)
    assert direct.available_variants(profile().key, 'layout_02') == ['测试路线', 'clean']
    floors._close_fixed_session()
    clean = profile().clean_root
    shutil.copytree(clean/'layout_01', clean/'layout_03')
    (clean/'layout_03/layout_01_1st.png').rename(clean/'layout_03/layout_03_1st.png')
    asset_sync.synchronize(root)
    assert direct.available_variants(profile().key, 'layout_03') == ['clean']
    assert len(MapCatalog().layouts(profile())) == 3
    floors._close_fixed_session()
    shutil.rmtree(clean/'layout_03')
    asset_sync.synchronize(root)
    assert set(read_bundle(root)['template']['groups'][profile().key]) == {'layout_01', 'layout_02'}
