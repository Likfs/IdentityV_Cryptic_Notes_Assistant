import hashlib
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import projection_workflow as workflow
from src.registration.template_matching import direct, matcher, floors


def write_image(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


@pytest.fixture
def originals(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    floors._images.clear()
    monkeypatch.setattr(direct, 'screen_info', lambda _: dict(factor=1))
    direct._images.clear()
    matcher._map_cache.clear()
    image = np.full((720, 300, 3), (39, 32, 24), np.uint8)
    rng = np.random.default_rng(58)
    screenshots = []
    candidates = []
    for i, (floor, scale) in enumerate(zip(('2nd', '1st', 'b1'), (1., 1.5, 2.))):
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
    entry = dict(source_image=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 map_threshold=48, scale_provenance='derived_from_route_calibrations', candidates=candidates)
    parts = []
    for i, floor in enumerate(('2nd', '1st', 'b1')):
        tile = tmp_path/f'{floor}.png'
        write_image(tile, image[i*240:(i+1)*240])
        parts.append(dict(floor=floor, path=tile.name, sha256=hashlib.sha256(tile.read_bytes()).hexdigest(),
                          region=[0, i*240, 300, 240]))
    clean = dict(source_image='.', source_sha256=direct.entry_fingerprint(parts), source_parts=parts,
                 map_threshold=48, candidates=json.loads(json.dumps(candidates)))
    config = dict(entries={}, perf_mode=0, original_variants={'map/diff': {'layout_01': {
        'route_lhp_solo': entry, 'route_clean': clean}}})
    floor_records = {floor: dict(source_image=f'{floor}.png', source_sha256=parts[i]['sha256'],
        size=[300, 240], saved_scale=candidates[i]['saved_scale'], map_threshold=48)
        for i, floor in enumerate(floors.FLOORS)}
    config['floor_variants'] = {'map/diff': {'layout_01': {
        variant: json.loads(json.dumps(floor_records)) for variant in ('route_lhp_solo', 'route_clean')}}}
    (tmp_path/'configs').mkdir()
    config_path = tmp_path/'configs/template_matching.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    yield config, config_path, path, screenshots
    direct._images.clear()
    matcher._map_cache.clear()


@pytest.mark.parametrize('floor_index', [0, 1, 2])
@pytest.mark.parametrize('perf_mode', [0, 2])
@pytest.mark.parametrize('variant_id', ['route_lhp_solo', 'route_clean'])
def test_three_fixed_candidates_recover_known_original_pose(originals, monkeypatch, floor_index, perf_mode, variant_id):
    config, config_path, _, screenshots = originals
    config['perf_mode'] = perf_mode
    config_path.write_text(json.dumps(config))
    shot, scale, origin = screenshots[floor_index]
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('continuous scale search'))
    result = workflow.process_screenshot(shot, catalog=object(), profile=SimpleNamespace(key='map/diff'), layout_override='layout_01', opacity=0.6, registration_backend='template', template_variant_id=variant_id)
    assert result.accepted, result.reasons
    assert result.floor == floors.FLOORS[floor_index] and result.variant_id == variant_id
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


@pytest.mark.parametrize('change', ['tile', 'placement'])
def test_clean_canvas_rejects_changed_parts_after_cache(originals, change):
    config, config_path, source, _ = originals
    entry = config['original_variants']['map/diff']['layout_01']['route_clean']
    loaded = direct.load_original('map/diff', 'layout_01', 'route_clean')
    np.testing.assert_array_equal(loaded[3], matcher.read_image(source))
    if change == 'tile':
        (source.parent/entry['source_parts'][0]['path']).write_bytes(b'replaced')
    else:
        entry['source_parts'][0]['region'][1] = 20
        config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='改变'):
        direct.load_original('map/diff', 'layout_01', 'route_clean')


def test_dark_background_transparent_but_blue_route_preserved(originals):
    config, config_path, path, _ = originals
    image = matcher.read_image(path)
    image[1, 1] = (255, 0, 0)
    write_image(path, image)
    config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    rgba = direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')[4]
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 1, 3] == 255


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


@pytest.mark.parametrize('change', ['source', 'dependency', 'reference_scale'])
def test_changed_derivation_inputs_reject_even_after_image_cached(originals, change):
    config, config_path, source, _ = originals
    dependency = source.parent/'calibration.json'
    dependency.write_text('old calibration')
    reference = dict(saved_scale=2., source_image='reference.png')
    config['entries']['clean.png'] = reference
    for candidate in config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['candidates']:
        candidate.update(dependencies=[dict(path=dependency.name, sha256=hashlib.sha256(dependency.read_bytes()).hexdigest())],
                         reference_entry_key='clean.png', reference_entry_sha256=direct.entry_fingerprint(reference))
    config_path.write_text(json.dumps(config))
    direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')
    if change == 'source':
        source.write_bytes(b'replaced')
    elif change == 'dependency':
        dependency.write_text('new calibration')
    else:
        config['entries']['clean.png']['saved_scale'] = 3.
        config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='改变'):
        direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')


def test_invalid_candidates_stay_rejected_without_fallback(originals, monkeypatch):
    config, config_path, _, screenshots = originals
    monkeypatch.setattr(matcher, '_match_template', lambda *a, **kw: (_ for _ in ()).throw(ValueError('截图可见区域大于单层图片')))
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert len(result.validation['fixed_candidates']) == 3
    assert '所有固定尺度候选' in result.rejection_reasons[0]


def test_reimported_source_rebuilds_matching_and_alpha_caches(originals):
    config, config_path, path, screenshots = originals
    shot, _, _ = screenshots[0]
    first, _, _, _, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert first.accepted and matcher._map_cache
    matcher._map_cache.clear()
    matcher._match_template(shot, str(path), 1., map_threshold=48)
    image = matcher.read_image(path)
    image[0, 0] = (200, 200, 200)
    write_image(path, image)
    config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    assert direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')[4][0, 0, 3] == 255
    assert not matcher._map_cache


@pytest.mark.parametrize('scale', [None, 0, -1, float('nan'), float('inf')])
def test_missing_or_invalid_saved_scale_never_starts_search(originals, monkeypatch, scale):
    config, path, _, screenshots = originals
    candidate = config['floor_variants']['map/diff']['layout_01']['route_lhp_solo']['2nd']
    if scale is None:
        candidate.pop('saved_scale')
    else:
        candidate['saved_scale'] = scale
    path.write_text(json.dumps(config))
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid saved scale triggered a matching search')
    monkeypatch.setattr(matcher, '_match_template', forbidden)
    monkeypatch.setattr(matcher, 'find_best_scale', forbidden)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert result.original_to_screen is None and result.screenshot_to_original is None
    assert '有效尺度' in result.rejection_reasons[0]


def test_unmatchable_candidates_do_not_hide_a_valid_fixed_candidate(originals, monkeypatch):
    config, path, _, screenshots = originals
    match = matcher._match_template
    def limited(image, path, scale, **options):
        if scale < 2:
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
        return dict(score={1.: -.8, 1.5: -.1, 2.: -.4}[scale], location=(11, 23))
    monkeypatch.setattr(matcher, '_match_template', match)
    monkeypatch.setattr(matcher, 'find_best_scale', lambda *a: pytest.fail('Continuous search'))
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert scales == [1., 1.5, 2.]
    assert result.accepted and layer is not None
    assert result.validation['template']['score'] == -.1
    np.testing.assert_allclose(result.screenshot_to_original, [[1.5, 0, 11], [0, 1.5, 23], [0, 0, 1]])


def test_scale_edit_during_matching_cannot_publish_a_mixed_configuration(originals, monkeypatch):
    config, path, _, screenshots = originals
    def match(*args, **kwargs):
        config['floor_variants']['map/diff']['layout_01']['route_lhp_solo']['2nd']['saved_scale'] = 1.1
        path.write_text(json.dumps(config))
        return dict(score=10., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert '配置已改变' in result.rejection_reasons[0]


def test_projection_reads_config_only_at_request_boundaries(originals, monkeypatch):
    """Three candidate loads must share one verified configuration snapshot."""
    _, path, _, screenshots = originals
    reads = []
    read_text = type(path).read_text
    def counted(file, *args, **kwargs):
        if file == path:
            reads.append(file)
        return read_text(file, *args, **kwargs)
    monkeypatch.setattr(type(path), 'read_text', counted)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert result.accepted and layer is not None
    assert len(result.validation['fixed_candidates']) == 3
    assert len(reads) == 2


def test_dependency_edit_during_matching_rejects_even_with_unchanged_stat(originals, monkeypatch):
    import os
    config, path, source, screenshots = originals
    dependency = source.parent/'calibration.json'
    dependency.write_bytes(b'{"scale":1}')
    stamp = dependency.stat()
    for item in config['floor_variants']['map/diff']['layout_01']['route_lhp_solo'].values():
        item['dependencies'] = [dict(path=dependency.name,
            sha256=hashlib.sha256(dependency.read_bytes()).hexdigest())]
    path.write_text(json.dumps(config))
    def match(*args, **kwargs):
        dependency.write_bytes(b'{"scale":2}')
        os.utime(dependency, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return dict(score=10., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5,
                                                    variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert '资源已改变' in result.rejection_reasons[0]
