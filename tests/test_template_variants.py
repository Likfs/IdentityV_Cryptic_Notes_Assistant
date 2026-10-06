"""Version-specific originals, fixed candidates and stale Qt delivery regression."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import projection_workflow as workflow
from src import route_projection_app as app
from src.app_catalog import MapCatalog, load_settings, save_settings, selected_template_variant
from src.registration.template_matching import direct, matcher
from tools.common.route_calibration import fit_fixed_scale
from test_reference_map_ui import window


def write_image(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


@pytest.fixture
def originals(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(direct.runtime, 'screen_info', lambda _: dict(factor=1))
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
    monkeypatch.setattr(workflow, 'register_screenshot', lambda *a, **kw: pytest.fail('other backend'))
    result = workflow.process_screenshot(shot, catalog=object(), profile=SimpleNamespace(key='map/diff'),
        model_bundle=None, floor='unknown', layout_override='layout_01', route_versions={}, opacity=.6,
        prepared_cache={}, asset_cache={}, registration_backend='template', template_variant_id=variant_id)
    assert result.accepted, result.reasons
    assert result.floor == 'all' and result.variant_id == variant_id
    template = result.registration.validation['template']
    assert template['candidate_floor'] == ('2nd', '1st', 'b1')[floor_index]
    assert len(result.registration.validation['fixed_candidates']) == 3
    expected = np.float64([[scale, 0, origin[0]], [0, scale, origin[1]], [0, 0, 1]])
    np.testing.assert_allclose(result.registration.screenshot_to_original, expected,
                               atol=1/matcher.PERF_MODES[perf_mode]['ds'])
    np.testing.assert_allclose(result.registration.original_to_screen,
                               np.linalg.inv(result.registration.screenshot_to_original))
    rgba = direct.load_original('map/diff', 'layout_01', variant_id)[4]
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


def test_invalid_candidates_stay_rejected_without_fallback(originals):
    config, config_path, _, screenshots = originals
    entry = config['original_variants']['map/diff']['layout_01']['route_lhp_solo']
    for candidate in entry['candidates']:
        candidate['search_region'] = [0, 0, 10, 10]
    config_path.write_text(json.dumps(config))
    result, _, layer, _, _ = direct.project_original(screenshots[0][0], 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert not result.accepted and layer is None
    assert len(result.validation['fixed_candidates']) == 3
    assert '所有固定尺度候选' in result.rejection_reasons[0]


def test_reimported_source_rebuilds_matching_and_alpha_caches(originals):
    config, config_path, path, screenshots = originals
    shot, _, _ = screenshots[0]
    first, _, _, _, _ = direct.project_original(shot, 'map/diff', 'layout_01', .5, variant_id='route_lhp_solo')
    assert first.accepted and matcher._map_cache
    image = matcher.read_image(path)
    image[0, 0] = (200, 200, 200)
    write_image(path, image)
    config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    assert direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')[4][0, 0, 3] == 255
    assert not matcher._map_cache


def test_fixed_scale_fit_recovers_scale_translation_and_reports_residuals():
    points = np.float64([[0, 0], [100, 0], [0, 100], [100, 100]])
    scale, diagnostics = fit_fixed_scale(points, points*.73+[27, 405])
    assert scale == pytest.approx(.73)
    assert diagnostics['translation'] == pytest.approx([27, 405])
    assert diagnostics['residual_rms_original_pixels'] < 1e-12
    with pytest.raises(ValueError, match='空间范围不足'):
        fit_fixed_scale(np.zeros((3, 2)), np.ones((3, 2)))


def test_old_settings_default_to_zhanshi_and_preferences_remain_independent(tmp_path):
    catalog = MapCatalog()
    path = tmp_path/'settings.json'
    profile = next(iter(catalog.profiles.values()))
    path.write_text(json.dumps(dict(route_versions={profile.key: {'layout_01': 'route_clean'}})))
    settings = load_settings(catalog, path)
    assert selected_template_variant(settings, profile.key) == 'route_zhanshi_fast'
    settings['template_route_versions'][profile.key] = 'route_lhp_solo'
    save_settings(settings, path)
    reloaded = load_settings(catalog, path)
    assert selected_template_variant(reloaded, profile.key) == 'route_lhp_solo'
    assert reloaded['route_versions'] == settings['route_versions']


@pytest.mark.parametrize('state', ['open', 'closed', 'background', 'busy'])
@pytest.mark.parametrize('variant_id', ['route_lhp_solo', 'route_clean', 'route_lhp_solo_clean', 'route_zhanshi_fast_clean'])
def test_qt_version_switch_clears_old_result_and_rematches_only_current_state(window, monkeypatch, state, variant_id):
    monkeypatch.setattr(window, '_start_registration_backend', lambda: None)
    calls = []
    monkeypatch.setattr(window, 'request_work', lambda mode, **kw: calls.append((mode, kw)))
    versions = json.loads(json.dumps(window.settings['route_versions']))
    window.apply_settings({**window.settings, 'registration_backend': 'template'})
    assert [window.route_combo.itemData(i) for i in range(window.route_combo.count())] == [
        'route_zhanshi_fast', 'route_lhp_solo', 'route_clean', 'route_lhp_solo_clean', 'route_zhanshi_fast_clean']
    generation = window.map_generation
    window.last_result = object()
    window.settings['auto_project'] = False  # Explicit version changes still rematch.
    window.map_state.is_open = state != 'closed'
    window.busy = state == 'busy'
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: state != 'background')
    window.route_combo.setCurrentIndex(window.route_combo.findData(variant_id))
    assert window.map_generation == generation+1 and window.last_result is None and window.live_projection is None
    assert window.settings['route_versions'] == versions
    assert load_settings(window.catalog, window.settings_path)['template_route_versions'][window.profile.key] == variant_id
    assert bool(calls) == (state == 'open')
    if state == 'busy':
        window.busy = False
        window._drain_pending_projection()
        assert calls == [('project', {'auto': False})]
    old = SimpleNamespace(accepted=True, floor='all', layer=np.ones((5, 5, 4), np.uint8))
    window._apply_result('project', True, generation, [0, 0, 5, 5], old, None)
    assert window.last_result is None


def test_reference_and_late_preview_follow_selected_original(window, monkeypatch):
    monkeypatch.setattr(window, '_start_registration_backend', lambda: None)
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: None)
    window.apply_settings({**window.settings, 'registration_backend': 'template'})
    ready = SimpleNamespace()
    window.last_result = ready
    generation, session = window.map_generation, window.session_id
    window.route_combo.setCurrentIndex(window.route_combo.findData('route_lhp_solo'))
    monkeypatch.setattr(window, '_set_preview_kind', lambda *a: pytest.fail('late preview'))
    window._receive_work_event(('preview', generation, session, ready, SimpleNamespace()))
    assert window.last_result is None
    seen = []
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: seen.append(a))
    window.reference_override = True
    window._apply_reference_ready(window.map_generation, [0, 0, 1180, 1050], 'all', window.profile.key, window.locked_layout)
    assert seen and '凉哈皮' in seen[-1][2]
    assert seen[-1][0].shape == direct.load_original(window.profile.key, window.locked_layout, 'route_lhp_solo')[3].shape


def test_all_production_originals_are_verified_and_in_release_manifest():
    catalog = MapCatalog()
    profile = catalog.profile('a_bringer_of_doom', 'nightmare')
    layouts = catalog.layouts(profile)
    summary = direct.check_original_resources(profile.key, layouts)
    assert summary['route_lhp_solo']['layouts'] == len(layouts)
    assert summary['route_lhp_solo']['fixed_candidates'] == len(layouts)*3
    assert summary['route_zhanshi_fast']['fixed_candidates'] == len(layouts)
    assert summary['route_clean']['layouts'] == len(layouts)
    assert summary['route_clean']['fixed_candidates'] == len(layouts)*3
    assert summary['route_lhp_solo_clean']['fixed_candidates'] == len(layouts)*3
    assert summary['route_zhanshi_fast_clean']['fixed_candidates'] == len(layouts)
    assert all(not item['errors'] for item in summary.values())
    from tools.release.build_windows_exe import runtime_files, ROOT
    manifest = set(runtime_files())
    for layout in layouts:
        entry, source, _ = direct._source(profile.key, layout, 'route_lhp_solo')
        assert source.resolve() in manifest
        for candidate in entry['candidates']:
            assert all((ROOT/d['path']).resolve() in manifest for d in candidate['dependencies'])
        clean, _, _ = direct._source(profile.key, layout, 'route_clean')
        assert all((ROOT/part['path']).resolve() in manifest for part in clean['source_parts'])
        for variant_id in ('route_lhp_solo_clean', 'route_zhanshi_fast_clean'):
            source_variant, path, _ = direct._source(profile.key, layout, variant_id)
            assert path.resolve() in manifest
            parent, _, _ = direct._source(profile.key, layout, source_variant['reference_variant_id'])
            assert [c['saved_scale'] for c in source_variant['candidates']] == [c['saved_scale'] for c in parent['candidates']]
