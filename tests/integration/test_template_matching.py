from tests.support.published_records import write_template
import hashlib
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import projection_workflow as workflow
from src.registration.template_matching import direct, matcher, floors, walls
from tests.support.frozen_resources import restart_template_resources
from src.floor_indicator import FloorDetection
from src.floor_indicator import detect_floor


def write_image(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


@pytest.fixture
def originals(tmp_path, monkeypatch, request):
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    floors._images.clear()
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


@pytest.mark.parametrize('floor_index', [0, 1, 2])
def test_ui_floor_matches_only_its_clean_image_and_keeps_geometry(originals, monkeypatch, floor_index):
    _, _, _, screenshots = originals
    selected = floors.DEFAULT_FLOORS[floor_index]
    shot, scale, origin = screenshots[floor_index]
    profile = SimpleNamespace(key='map/diff', version_names={'clean': 'clean'},
                              floors_for_layout=lambda _: floors.DEFAULT_FLOORS)
    called = []
    real = walls.match
    def match(game, path, *args, **kwargs):
        called.append(path)
        assert path.replace('\\', '/').endswith('/'+selected+'.png')
        assert kwargs['expand_search'] is True
        return real(game, path, *args, **kwargs)
    monkeypatch.setattr(walls, 'match', match)
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw: pytest.fail('binary fallback'))
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, template_variant_id='clean', floor_detection=FloorDetection(selected))
    assert result.accepted and result.floor == selected
    assert len(called) == 1
    assert [a['candidate_floor'] for a in result.registration.validation['fixed_candidates']] == [selected]
    assert result.registration.validation['floor_detection_enabled']
    assert result.registration.validation['search_policy'] == 'direct_expanded'
    assert result.registration.validation['matching_method'] == 'directional_walls'
    assert result.registration.validation['template']['score_kind'] == 'mean_directional_wall_support'
    assert result.diagnostics()['floor_detection']['floor'] == selected
    np.testing.assert_allclose(result.registration.screenshot_to_original,
                               [[scale, 0, origin[0]], [0, scale, 55], [0, 0, 1]], atol=1)


@pytest.mark.parametrize('units', [11, 14, 26])
def test_ui_floor_registration_keeps_tracked_zoom_without_scale_search(originals, monkeypatch, units):
    selected = '1st'
    shot, scale, origin = originals[3][1]
    zoom = units / 7
    shot = cv2.resize(shot, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    profile = SimpleNamespace(key='map/diff', version_names={'clean':'clean'},
                              floors_for_layout=lambda _:floors.DEFAULT_FLOORS)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a:pytest.fail('Scale search must not run'))
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw:pytest.fail('UI floor cannot fall back to another matcher'))
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, template_variant_id='clean', floor_detection=FloorDetection(selected), map_zoom_units=units)
    assert result.accepted and result.floor == selected, result.reasons
    assert result.registration.validation['map_zoom_units'] == units
    assert result.registration.validation['template']['saved_scale'] == 1.5
    np.testing.assert_allclose(result.registration.screenshot_to_original,
                               [[scale / zoom, 0, origin[0]], [0, scale / zoom, 55], [0, 0, 1]], atol=2)


@pytest.mark.parametrize('selection', [FloorDetection(None, reason='uncertain UI'), FloorDetection('b4')])
def test_unknown_or_unavailable_ui_floor_never_runs_map_matching(originals, monkeypatch, selection):
    shot = originals[3][0][0]
    profile = SimpleNamespace(key='map/diff', version_names={'clean': 'clean'},
                              floors_for_layout=lambda _: floors.DEFAULT_FLOORS)
    monkeypatch.setattr(direct, 'project_original', lambda *a, **kw: pytest.fail('fallback map matching'))
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, template_variant_id='clean', floor_detection=selection)
    assert not result.accepted and result.layer is None and result.registration is None
    assert result.reasons and result.diagnostics()['floor_detection']['floor'] == selection.floor


def test_single_floor_selection_runs_real_matching_without_claiming_ui_confidence(originals):
    shot = originals[3][1][0]
    profile = SimpleNamespace(key='map/diff', version_names={'clean':'clean'},
                              floors_for_layout=lambda _:('1st',))
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, template_variant_id='clean', floor_detection=FloorDetection('1st', selection_source='single_floor_layout'))
    assert result.accepted and result.floor == '1st', result.reasons
    assert not result.registration.validation['floor_detection_enabled']
    assert result.registration.validation['floor_selection_source'] == 'single_floor_layout'
    assert [item['candidate_floor'] for item in result.registration.validation['fixed_candidates']] == ['1st']
    assert result.floor_detection['score'] == 0 and not result.floor_detection['candidates']


def test_match_crop_preserves_full_viewport_matrices_and_routes(originals, monkeypatch):
    shot, scale, _ = originals[3][1]
    monkeypatch.setattr(walls, 'match', lambda *a, **kw:dict(score=8., location=(75.,55.)))
    baseline, _, full_layer, _, _ = direct.project_original(shot, 'map/diff', 'layout_01', .6,
        variant_id='clean', selected_floor='1st')
    assert baseline.accepted
    roi = (11, 7, 45, 45)
    polluted = shot.copy()
    polluted[:, :11] = (255,0,255)
    polluted[:, 56:] = (0,255,255)
    seen = []
    def cropped_match(image, *a, **kw):
        seen.append(image.copy())
        return dict(score=8., location=(75.+scale*roi[0],55.+scale*roi[1]))
    monkeypatch.setattr(walls, 'match', cropped_match)
    result, _, layer, preview, _ = direct.project_original(polluted, 'map/diff', 'layout_01', .6,
        variant_id='clean', selected_floor='1st', matching_roi=roi)
    assert result.accepted, result.rejection_reasons
    np.testing.assert_array_equal(seen[0], shot[7:52,11:56])
    np.testing.assert_allclose(result.original_to_screen, baseline.original_to_screen, atol=1e-12)
    np.testing.assert_allclose(result.screenshot_to_original@result.original_to_screen, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(layer, full_layer)
    assert layer.shape[:2] == preview.shape[:2] == shot.shape[:2]
    assert layer[:, :roi[0], 3].any()  # Routes outside the matching crop were genuinely rendered.
    assert result.validation['template']['location'] == [75.,55.]


@pytest.mark.parametrize('roi', [(-1,0,2,2), (0,0,0,4), (0,0,1000,4), (1.5,0,2,2)])
def test_invalid_match_crop_rejects_before_map_loading(roi, monkeypatch):
    monkeypatch.setattr(direct, 'available_variants', lambda *a:pytest.fail('Invalid ROI loaded resources'))
    result, _, layer, _, _ = direct.project_original(np.full((40,50,3),100,np.uint8),
        'map/diff','layout_01',.6,matching_roi=roi)
    assert not result.accepted and layer is None and '匹配区域' in result.rejection_reasons[0]


def test_ui_pixels_outside_match_crop_cannot_make_empty_map_matchable(monkeypatch):
    monkeypatch.setattr(direct, 'available_variants', lambda *a:pytest.fail('UI-only frame loaded maps'))
    image = np.zeros((40,60,3), np.uint8)
    image[:, :10] = 255
    result, _, layer, _, _ = direct.project_original(image, 'map/diff', 'layout_01', .6,
        matching_roi=(10,0,50,40))
    assert not result.accepted and layer is None and '没有可匹配' in result.rejection_reasons[0]


def test_floor_rejection_also_saves_cropped_matching_input(tmp_path):
    image = np.full((40,60,3), 100, np.uint8)
    profile = SimpleNamespace(key='map/diff')
    result = workflow.process_screenshot(image, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, floor_detection=FloorDetection(None, reason='no selected floor'), matching_roi=(10,0,40,40))
    assert not result.accepted and result.registration is None
    directory = workflow.save_workflow_result(result, tmp_path/'failure')
    np.testing.assert_array_equal(cv2.imread(str(directory/'04_matching_input.png')), image[:,10:50])


def test_recognition_uses_only_cropped_pixels_and_saves_both_inputs(tmp_path):
    image = np.random.default_rng(90).integers(0, 256, (80,120,3), np.uint8)
    seen = []
    def recognize(query, reference_size):
        seen.append((query.copy(), reference_size))
        return SimpleNamespace(decision=lambda:dict(best_label='layout_01', layout=None),
            timings_ms={'total':1.}, reasons=[], diagnostics=lambda:{'stub':True})
    result = workflow.process_screenshot(image, catalog=object(), profile=SimpleNamespace(key='map/diff'),
        opacity=.6, recognize_only=True, layout_retriever=SimpleNamespace(recognize=recognize),
        recognition_reference_size=(160,160), matching_roi=(20,0,80,80))
    assert result.accepted
    np.testing.assert_array_equal(seen[0][0], image[:,20:100])
    assert seen[0][1] == (160,160)
    path = workflow.save_workflow_result(result, tmp_path/'diagnostic')
    np.testing.assert_array_equal(cv2.imread(str(path/'01_screenshot.png')), image)
    np.testing.assert_array_equal(cv2.imread(str(path/'04_matching_input.png')), image[:,20:100])
    data = json.loads((path/'result.json').read_text(encoding='utf-8'))
    assert data['matching_roi'] == [20,0,80,80]


def test_direct_selected_floor_must_exist_in_published_images(originals, monkeypatch):
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw: pytest.fail('matched another floor'))
    monkeypatch.setattr(walls, 'match', lambda *a, **kw: pytest.fail('matched another floor'))
    result, _, layer, _, _ = direct.project_original(originals[3][0][0], 'map/diff', 'layout_01', .6,
                                                    variant_id='clean', selected_floor='b4')
    assert not result.accepted and layer is None
    assert 'b4' in result.rejection_reasons[0]


def test_ui_floor_without_continuous_walls_rejects_without_binary_fallback(originals, monkeypatch):
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw: pytest.fail('binary fallback'))
    result, _, layer, _, _ = direct.project_original(np.full((60,70,3),100,np.uint8),
        'map/diff', 'layout_01', .6, variant_id='clean', selected_floor='b1')
    assert not result.accepted and layer is None
    assert '连续墙线' in result.rejection_reasons[0]
    assert [a['candidate_floor'] for a in result.validation['fixed_candidates']] == ['b1']


@pytest.mark.parametrize('valid', [True, False])
def test_floor_ui_diagnostics_are_saved_on_success_and_rejection(originals, tmp_path, valid):
    from tests.support.floor_ui import recorded_control
    control = recorded_control()
    if not valid:
        control = np.full((383, 104, 3), 100, np.uint8)
    detection = detect_floor(control)
    shot = originals[3][1][0]
    profile = SimpleNamespace(key='map/diff', version_names={'clean':'clean'},
                              floors_for_layout=lambda _:floors.DEFAULT_FLOORS)
    size = (2560, 1440) if valid else (1920, 1080)
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_01',
        opacity=.6, template_variant_id='clean', floor_detection=detection,
        capture_client_size=size, defer_projection_preview=True)
    assert result.accepted == valid, result.reasons
    completed = workflow.complete_projection_preview(result)
    assert completed.floor_ui_image is result.floor_ui_image
    directory = workflow.save_workflow_result(completed, tmp_path/'diagnostic')
    data = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    assert data['capture_context']['game_client_size'] == list(size)
    assert len(data['floor_detection']['candidates']) == 7
    np.testing.assert_array_equal(cv2.imread(str(directory/'02_floor_ui_raw.png')), control)
    np.testing.assert_array_equal(cv2.imread(str(directory/'03_floor_ui_normalized.png'), 0), detection.normalized_image)
    assert data['floor_ui_images']['raw'] == '02_floor_ui_raw.png'
    if not valid:
        assert data['registration'] is None and data['floor'] == 'unknown'
        assert not (directory/'06_overlay_layer.png').exists()


@pytest.mark.parametrize('case_id', ['text_prompt', 'no_text_control'])
def test_directional_walls_replay_text_and_normal_with_one_verified_floor(tmp_path, monkeypatch, case_id):
    from pathlib import Path
    import shutil
    fixture = Path(__file__).resolve().parents[1]/'fixtures/registration/directional_walls'
    cases = json.loads((fixture/'cases.json').read_text(encoding='utf-8'))
    for name, digest in cases['files'].items():
        assert hashlib.sha256((fixture/name).read_bytes()).hexdigest() == digest
    case = next(c for c in cases['cases'] if c['id'] == case_id)
    shutil.copyfile(fixture/'clean_b3.png', tmp_path/'clean.png')
    entry = dict(source_image='clean.png', source_sha256=cases['files']['clean_b3.png'],
                 size=cases['size'], saved_scale=1.5, map_threshold=56)
    write_template(tmp_path,dict(perf_mode=2,floor_variants={'map/diff':{'layout_02':{'clean':{'b3':entry}}}}))
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    shot = matcher.read_image(fixture/(case_id+'.png'))
    binary = matcher._match_template(shot,str(tmp_path/'clean.png'),1.5,ds=.25,expand_search=True,
        map_threshold=56,map_image=matcher.read_image(tmp_path/'clean.png'),image_fingerprint='binary-control')
    assert list(binary['location']) == case['old_binary']
    if case_id == 'text_prompt':
        assert np.linalg.norm(np.array(binary['location'])-case['wall_origin']) > 1000
    monkeypatch.setattr(matcher,'_match_template',lambda *a, **kw:pytest.fail('binary fallback'))
    profile = SimpleNamespace(key='map/diff',version_names={'clean':'clean'},floors_for_layout=lambda _:('b3',))
    result = workflow.process_screenshot(shot,catalog=object(),profile=profile,layout_override='layout_02',
        opacity=.34,template_variant_id='clean',floor_detection=FloorDetection('b3'))
    assert result.accepted, result.reasons
    v = result.registration.validation
    assert v['matching_method'] == 'directional_walls'
    assert [c['candidate_floor'] for c in v['fixed_candidates']] == ['b3']
    assert v['template']['location'] == case['wall_origin']
    assert v['template']['scale'] == 1.5 and v['template']['ds'] == .25
    np.testing.assert_allclose(result.registration.original_to_screen@result.registration.screenshot_to_original,
                               np.eye(3),atol=1e-10)
    assert v['template']['wall_match']['wall_pixels'] > 0 and result.layer is not None


@pytest.mark.parametrize('case_name', ['text_prompt', 'no_text_control'])
def test_ui_direct_expansion_replays_fixed_capture_without_changing_scale(tmp_path, monkeypatch, case_name):
    from pathlib import Path
    import shutil
    fixture = Path(__file__).resolve().parents[1]/'fixtures/registration/ui_direct_expansion'
    cases = json.loads((fixture/'cases.json').read_text(encoding='utf-8'))
    for name, digest in cases['files'].items():
        assert hashlib.sha256((fixture/name).read_bytes()).hexdigest() == digest
    shutil.copyfile(fixture/'clean_b3.png', tmp_path/'clean.png')
    entry = dict(source_image='clean.png', source_sha256=cases['files']['clean_b3.png'],
                 size=cases['source_size'], saved_scale=1.5, map_threshold=56)
    write_template(tmp_path, dict(perf_mode=2, floor_variants={'map/diff': {'layout_12': {'clean': {'b3': entry}}}}))
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    shot = matcher.read_image(fixture/f'{case_name}.png')
    if case_name == 'text_prompt':
        limited = matcher._match_template(shot, str(tmp_path/'clean.png'), 1.5, ds=.25,
            map_threshold=56, map_image=matcher.read_image(tmp_path/'clean.png'), image_fingerprint='limited-control')
        assert not limited['search']['boundary_edges']
        np.testing.assert_array_equal(limited['location'], cases[case_name]['old_origin'])
        assert limited['search']['origin_bounds_xyxy'][1] > cases[case_name]['expected_origin'][1]
    profile = SimpleNamespace(key='map/diff', version_names={'clean': 'clean'}, floors_for_layout=lambda _: ('b3',))
    result = workflow.process_screenshot(shot, catalog=object(), profile=profile, layout_override='layout_12',
        opacity=.34, template_variant_id='clean', floor_detection=FloorDetection('b3'), capture_client_size=(2560,1440))
    assert result.accepted, result.reasons
    validation = result.registration.validation
    assert validation['search_policy'] == 'direct_expanded'
    assert validation['template']['search']['expanded']
    assert validation['template']['ds'] == .25
    assert len(validation['fixed_candidates']) == 1
    assert 'initial_attempt' not in validation['fixed_candidates'][0]
    expected = [[1.5, 0, cases[case_name]['expected_origin'][0]],
                [0, 1.5, cases[case_name]['expected_origin'][1]], [0, 0, 1]]
    np.testing.assert_allclose(result.registration.screenshot_to_original, expected, atol=0)
    np.testing.assert_allclose(result.registration.original_to_screen@result.registration.screenshot_to_original,
                               np.eye(3), atol=1e-10)
    assert result.layer.shape == (*shot.shape[:2], 4)
