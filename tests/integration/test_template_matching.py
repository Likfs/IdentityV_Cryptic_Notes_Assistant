from tests.support.published_records import write_template
import hashlib
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import projection_workflow as workflow
from src.registration.template_matching import direct, matcher, floors
from tests.support.frozen_resources import restart_template_resources


def write_image(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


@pytest.fixture
def originals(tmp_path, monkeypatch, request):
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    floors._images.clear()
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    matcher._map_cache.clear()
    image = np.full((720, 300, 3), (39, 32, 24), np.uint8)
    rng = np.random.default_rng(58)
    screenshots = []
    candidates = []
    for i, floor in enumerate(('2nd', '1st', 'b1')):
        scale = getattr(request, 'param', 1.5)
        shot = np.zeros((60, 70, 3), np.uint8)
        for _ in range(12):
            x, y = rng.integers(5, 47, 2)
            cv2.rectangle(shot, (int(x), int(y)), (int(x)+11, int(y)+9), (140, 140, 140), -1)
        scaled = cv2.resize(shot, (int(70*scale), int(60*scale)), interpolation=cv2.INTER_LINEAR)
        x, y = 75, 55+i*240
        image[y:y+scaled.shape[0], x:x+scaled.shape[1]] = scaled
        screenshots.append((shot, scale, (x, y)))
        candidates.append(dict(floor=floor, saved_scale=scale, search_region=[0, i*240, 300, 240]))
    path = tmp_path/'original.png'
    write_image(path, image)
    parts = []
    for i, floor in enumerate(('2nd', '1st', 'b1')):
        tile = tmp_path/f'{floor}.png'
        write_image(tile, image[i*240:(i+1)*240])
        parts.append(dict(floor=floor, path=tile.name, sha256=hashlib.sha256(tile.read_bytes()).hexdigest(),
                          region=[0, i*240, 300, 240]))
    config = dict(perf_mode=0)
    floor_records = {floor: dict(source_image=f'{floor}.png', source_sha256=parts[i]['sha256'],
        size=[300, 240], saved_scale=candidates[i]['saved_scale'], map_threshold=48)
        for i, floor in enumerate(floors.DEFAULT_FLOORS)}
    config['floor_variants'] = {'map/diff': {'layout_01': {
        variant: json.loads(json.dumps(floor_records)) for variant in ('route_lhp_solo', 'clean')}}}
    config_path = write_template(tmp_path, config)
    yield config, config_path, path, screenshots
    matcher._map_cache.clear()


@pytest.mark.parametrize('originals', [1., 1.5, 2.], indirect=True)
@pytest.mark.parametrize('floor_index', [0, 1, 2])
@pytest.mark.parametrize('perf_mode', [0, 2])
@pytest.mark.parametrize('variant_id', ['route_lhp_solo', 'clean'])
def test_three_fixed_candidates_recover_known_original_pose(originals, monkeypatch, floor_index, perf_mode, variant_id):
    config, config_path, _, screenshots = originals
    config['perf_mode'] = perf_mode
    write_template(config_path.parent.parent, config)
    shot, scale, origin = screenshots[floor_index]
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('continuous scale search'))
    profile = SimpleNamespace(key='map/diff', version_names={variant_id: variant_id})
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01', opacity=0.6, registration_backend='template', template_variant_id=variant_id)
    assert result.accepted, result.reasons
    assert result.floor == floors.DEFAULT_FLOORS[floor_index] and result.variant_id == variant_id
    template = result.registration.validation['template']
    assert template['candidate_floor'] == ('2nd', '1st', 'b1')[floor_index]
    assert len(result.registration.validation['fixed_candidates']) == 3
    expected = np.float64([[scale, 0, origin[0]], [0, scale, 55], [0, 0, 1]])
    np.testing.assert_allclose(result.registration.screenshot_to_original, expected,
                               atol=1/matcher.PERF_MODES[perf_mode]['ds'])
    np.testing.assert_allclose(result.registration.original_to_screen,
                               np.linalg.inv(result.registration.screenshot_to_original))
    rgba = floors.load('map/diff', 'layout_01', variant_id, result.floor)[4]
    layer = cv2.warpPerspective(rgba, result.registration.original_to_screen, (shot.shape[1], shot.shape[0]))
    layer[:,:,3] = np.rint(layer[:,:,3]*.6).astype(np.uint8)
    np.testing.assert_array_equal(result.layer, layer)


def test_region_and_threshold_have_separate_caches_and_never_resize_to_fit(originals):
    _, _, path, screenshots = originals
    matcher.TEMPLATE_DS = 1.
    shot, scale, _ = screenshots[1]
    good = matcher._match_template(shot, str(path), scale, map_threshold=48, search_region=[0, 240, 300, 240])
    other = matcher._match_template(shot, str(path), scale, map_threshold=48, search_region=[0, 0, 300, 240])
    assert good['score'] > other['score']
    assert len(matcher._map_cache) == 2
    matcher._match_template(shot, str(path), scale, map_threshold=3, search_region=[0, 240, 300, 240])
    assert len(matcher._map_cache) == 3
    with pytest.raises(ValueError, match='大于原图搜索区域'):
        matcher._match_template(shot, str(path), 2., map_threshold=48, search_region=[0, 0, 20, 20])


def test_invalid_candidates_stay_rejected_without_fallback(originals, monkeypatch):
    config, config_path, _, screenshots = originals
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw: (_ for _ in ()).throw(ValueError('截图可见区域大于单层图片')))
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert len(result.validation['fixed_candidates']) == 3
    assert '所有固定尺度候选' in result.rejection_reasons[0]


@pytest.mark.parametrize('scale', [None, 0, -1, float('nan'), float('inf')])
def test_missing_or_invalid_saved_scale_never_starts_search(originals, monkeypatch, scale):
    config, path, _, screenshots = originals
    for version in config['floor_variants']['map/diff']['layout_01'].values():
        for candidate in version.values():
            if scale is None:
                candidate.pop('saved_scale')
            else:
                candidate['saved_scale'] = scale
    write_template(path.parent.parent, config)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid saved scale triggered a matching search')
    monkeypatch.setattr(matcher, '_match_template', forbidden)
    monkeypatch.setattr(matcher, 'find_best_scale', forbidden)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert result.original_to_screen is None and result.screenshot_to_original is None
    assert 'minimum_scale' in result.rejection_reasons[0]


def test_unmatchable_candidates_do_not_hide_a_valid_fixed_candidate(originals, monkeypatch):
    config, path, _, screenshots = originals
    match = matcher._match_template
    def limited(image, path, scale, **options):
        if not str(path).endswith('b1.png'):
            raise ValueError('截图可见区域大于单层图片')
        return match(image, path, scale, **options)
    monkeypatch.setattr(matcher, '_match_template', limited)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('Continuous search'))
    result, _, layer, _, _ = direct.project_original(screenshots[2][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert result.accepted and layer is not None
    assert result.validation['template']['candidate_floor'] == 'b1'
    assert all('reason' in attempt for attempt in result.validation['fixed_candidates'][:2])
    scale, origin = screenshots[2][1:]
    np.testing.assert_allclose(result.screenshot_to_original,
                               [[scale, 0, origin[0]], [0, scale, 55], [0, 0, 1]], atol=1)


def test_only_saved_candidates_are_compared_and_highest_score_is_displayed(originals, monkeypatch):
    """Low correlation remains a candidate; never reintroduce multiscale acceptance thresholds."""
    _, _, _, screenshots = originals
    scales = []
    def match(image, path, scale, **kwargs):
        scales.append(scale)
        return dict(score={'2nd.png': -.8, '1st.png': -.1, 'b1.png': -.4}[str(path).replace('\\', '/').rsplit('/', 1)[-1]], location=(11, 23))
    monkeypatch.setattr(matcher, '_match_template', match)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('Continuous search'))
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert scales == [1.5]*3
    assert result.accepted and layer is not None
    assert result.validation['template']['score'] == -.1
    np.testing.assert_allclose(result.screenshot_to_original, [[1.5, 0, 11], [0, 1.5, 23], [0, 0, 1]])


def test_scale_edit_during_matching_keeps_fixed_configuration_until_restart(originals, monkeypatch):
    config, path, _, screenshots = originals
    def match(*args, **kwargs):
        for version in config['floor_variants']['map/diff']['layout_01'].values():
            for item in version.values():
                item['saved_scale'] = 1.1
        write_template(path.parent.parent, config)
        return dict(score=10., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert result.accepted and layer is not None
    assert [a['saved_scale'] for a in result.validation['fixed_candidates']] == [1.5]*3
    restart_template_resources()
    fresh = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')[0]
    assert fresh.accepted
    assert fresh.validation['fixed_candidates'][0]['saved_scale'] == 1.1


def test_projection_reads_config_only_once_per_process(originals, monkeypatch):
    """All requests and candidate loads share the fixed process configuration."""
    _, path, _, screenshots = originals
    reads = []
    open_file = type(path).open
    def counted(file, *args, **kwargs):
        if file == path:
            reads.append(file)
        return open_file(file, *args, **kwargs)
    monkeypatch.setattr(type(path), 'open', counted)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert result.accepted and layer is not None
    assert len(result.validation['fixed_candidates']) == 3
    assert len(reads) == 1
    assert direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')[0].accepted
    assert len(reads) == 1


def test_same_stat_scale_table_edit_keeps_snapshot_but_is_rejected_on_restart(originals, monkeypatch):
    import os
    config, path, source, screenshots = originals
    dependency = path.parent/'scale_config.json'
    stamp = dependency.stat()
    def match(*args, **kwargs):
        dependency.write_bytes(dependency.read_bytes().replace(b'1.5', b'1.1'))
        os.utime(dependency, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return dict(score=10., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert result.accepted and layer is not None
    restart_template_resources()
    fresh, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert not fresh.accepted and layer is None
    assert '尺度配置与已发布资源不一致' in fresh.rejection_reasons[0]


@pytest.mark.parametrize('floor_index', [0, 1])
def test_two_floor_profile_recovers_pose_without_a_basement(originals, monkeypatch, floor_index):
    config, path, _, screenshots = originals
    catalog_path = path.parent/'map_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    definition = catalog['maps'][0]['difficulties'][0]
    definition['floors'] = ['2nd', '1st']
    definition['template_versions'] = ['route_zhanshi_full', 'clean']
    definition['asset_versions'].append(dict(id='route_zhanshi_full', name='路线', kind='route', source_root='clean'))
    catalog_path.write_text(json.dumps(catalog), encoding='utf-8')
    records = config['floor_variants']['map/diff']['layout_01']['clean']
    del records['b1']
    config['floor_variants']['map/diff']['layout_01'] = {'route_zhanshi_full': records, 'clean': records}
    config['perf_mode'] = 0
    write_template(path.parent.parent, config)
    shot, scale, origin = screenshots[floor_index]
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('实时流程不得搜索尺度'))
    result, _, layer, _, _ = direct.project_original(shot, 'map/diff', 'layout_01', .6, variant_id='route_zhanshi_full')
    assert result.accepted, result.rejection_reasons
    assert result.validation['template']['candidate_floor'] == ('2nd', '1st')[floor_index]
    assert [a['candidate_floor'] for a in result.validation['fixed_candidates']] == ['2nd', '1st']
    np.testing.assert_allclose(result.screenshot_to_original, [[scale, 0, origin[0]], [0, scale, 55], [0, 0, 1]], atol=1)
    assert layer is not None


@pytest.mark.parametrize('mutation', ['scale', 'capture', 'invalid_config', 'scale_table', 'image', 'deleted_image'])
def test_external_writer_cannot_change_fixed_projection_until_restart(originals, monkeypatch, mutation):
    """Current bytes stay pinned; restarting reloads and checks the edited resources."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    config, path, source, screenshots = originals
    dependency = path.parent/'scale_config.json'
    entries = config['floor_variants']['map/diff']['layout_01']['route_lhp_solo']
    shot = screenshots[0][0]
    # Populate caches so a stale cache hit, not merely an initial read, is exercised.
    before, _, before_layer, before_preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert before.accepted
    ready, changed = Event(), Event()
    def write():
        assert ready.wait(10), 'Match did not reach mutation barrier'
        try:
            if mutation in ('scale', 'capture', 'invalid_config'):
                stamp = path.stat()
                if mutation == 'scale':
                    for version in config['floor_variants']['map/diff']['layout_01'].values():
                        for item in version.values():
                            item['saved_scale'] = 1.1
                elif mutation == 'capture':
                    config['capture'] = dict(screen_name='changed')
                if mutation != 'invalid_config':
                    write_template(path.parent.parent, config)
                    payload = path.read_bytes()
                else:
                    payload = b'{broken'
                if mutation == 'scale':
                    assert len(payload) == stamp.st_size
                replacement = path.with_suffix('.replacement')
                replacement.write_bytes(payload)
                os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                replacement.replace(path)
            elif mutation == 'scale_table':
                stamp = dependency.stat()
                dependency.write_bytes(dependency.read_bytes().replace(b'1.5', b'1.1'))
                os.utime(dependency, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            elif mutation == 'image':
                image = source.parent/'2nd.png'
                stamp = image.stat()
                image.write_bytes(b'changed')
                os.utime(image, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1_000_000_000))
            else:
                (source.parent/'2nd.png').unlink()
        finally:
            changed.set()
    match = matcher._match_template
    first = True
    def paused(*args, **kwargs):
        nonlocal first
        answer = match(*args, **kwargs)
        if first:
            first = False
            ready.set()
            assert changed.wait(10), 'External write did not finish'
        return answer
    monkeypatch.setattr(matcher, '_match_template', paused)
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write)
        result, _, layer, preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
        writer.result(timeout=10)
    assert result.accepted and result.validation == before.validation
    np.testing.assert_array_equal(result.screenshot_to_original, before.screenshot_to_original)
    np.testing.assert_array_equal(layer, before_layer)
    np.testing.assert_array_equal(preview, before_preview)
    # Re-decoding after eviction must still use the frozen bytes, not the edited files.
    floors._images.clear()
    matcher._map_cache.clear()
    again, _, layer, preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert again.accepted and again.validation == before.validation
    np.testing.assert_array_equal(layer, before_layer)
    np.testing.assert_array_equal(preview, before_preview)
    restart_template_resources()
    fresh = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')[0]
    if mutation in ('scale', 'capture'):
        assert fresh.accepted
        if mutation == 'scale':
            assert fresh.validation['fixed_candidates'][0]['saved_scale'] == 1.1
        else:
            assert floors.fixed_session().config['capture']['screen_name'] == 'changed'
    else:
        assert not fresh.accepted and fresh.rejection_reasons


@pytest.mark.parametrize('outside', [False, True])
def test_directory_retarget_keeps_fixed_projection_and_is_checked_on_restart(originals, monkeypatch, tmp_path_factory, outside):
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from tests.support.directory_links import create_directory_link, remove_directory_link
    config, path, source, screenshots = originals
    root = source.parent
    first = root/'first'
    first.mkdir()
    second = tmp_path_factory.mktemp('external_source') if outside else root/'second'
    second.mkdir(exist_ok=True)
    for floor in floors.DEFAULT_FLOORS:
        for directory in (first, second):
            shutil.copy2(root/f'{floor}.png', directory/f'{floor}.png')
    if not outside:
        (second/'2nd.png').write_bytes(b'changed')
    entries = config['floor_variants']['map/diff']['layout_01']['route_lhp_solo']
    for floor, entry in entries.items():
        entry['source_image'] = f'linked/{floor}.png'
    write_template(path.parent.parent, config)
    link = root/'linked'
    create_directory_link(link, first)
    ready, changed = Event(), Event()
    def retarget():
        assert ready.wait(10)
        try:
            remove_directory_link(link, root=root)
            create_directory_link(link, second)
        finally:
            changed.set()
    try:
        shot = screenshots[0][0]
        before, _, before_layer, before_preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
        assert before.accepted
        match = matcher._match_template
        first_call = True
        def paused(*args, **kwargs):
            nonlocal first_call
            answer = match(*args, **kwargs)
            if first_call:
                first_call = False
                ready.set()
                assert changed.wait(10)
            return answer
        monkeypatch.setattr(matcher, '_match_template', paused)
        with ThreadPoolExecutor(max_workers=1) as executor:
            writer = executor.submit(retarget)
            result, _, layer, preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
            writer.result(timeout=10)
        assert result.accepted and result.validation == before.validation
        np.testing.assert_array_equal(layer, before_layer)
        np.testing.assert_array_equal(preview, before_preview)
        (first/'2nd.png').write_bytes(b'replaced original target')
        floors._images.clear()
        matcher._map_cache.clear()
        again, _, layer, preview, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
        assert again.accepted and again.validation == before.validation
        np.testing.assert_array_equal(layer, before_layer)
        np.testing.assert_array_equal(preview, before_preview)
        restart_template_resources()
        fresh = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')[0]
        assert not fresh.accepted and fresh.rejection_reasons
    finally:
        remove_directory_link(link, root=root)
    assert (first/'2nd.png').exists() and (second/'2nd.png').exists()
