import hashlib
import json
import os
import cv2
import numpy as np
import pytest
from src.registration.template_matching import direct, matcher
from tools.assets.imports.import_template_source_variants import import_source_variants
from tools.common.route_calibration import verify_identity_geometry


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', image)[1].tofile(path)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path/'assets/maps/map/diff'
    image = np.random.default_rng(94).integers(0, 256, (660, 320), dtype=np.uint8)
    image[:5, :5] = 0
    regions = {floor: [0, i*220, 320, 220] for i, floor in enumerate(('2nd', '1st', 'b1'))}
    variants = {}
    config = dict(entries={'untouched': {'value': 17}}, perf_mode=0,
                  original_variants={'map/diff': {'layout_01': variants}})
    for parent in ('route_lhp_solo', 'route_zhanshi_fast'):
        path = root/'sources'/parent/'1.png'
        write_image(path, image)
        write_image(root/'sources'/(parent+'_clean')/'1.png', image)
        relative = path.relative_to(tmp_path).as_posix()
        candidates = [dict(saved_scale=.7)] if parent == 'route_zhanshi_fast' else [
            dict(floor=floor, saved_scale=scale, search_region=region)
            for (floor, region), scale in zip(regions.items(), (.7, .71, .9))]
        variants[parent] = dict(source_image=relative, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                candidates=candidates)
        if parent == 'route_lhp_solo':
            variants[parent]['map_threshold'] = 48
        cal = root/'routes'/parent/'layout_01/calibration.json'
        cal.parent.mkdir(parents=True, exist_ok=True)
        cal.write_text(json.dumps(dict(layout='layout_01', route=relative, floors={
            floor: dict(status='PASS', source_crop_xywh=region) for floor, region in regions.items()})))
    monkeypatch.setattr(direct, 'PROJECT_ROOT', tmp_path)
    direct._images.clear(); matcher._map_cache.clear()
    yield tmp_path, config, image, regions
    direct._images.clear(); matcher._map_cache.clear()


def save_config(root, config):
    path = root/'configs/template_matching.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(config))


def test_import_preserves_original_records_scales_and_regions(sources):
    root, old, _, _ = sources
    config, count = import_source_variants(old, root)
    assert count == 2 and config['entries'] == old['entries']
    for variant, reference in [('route_lhp_solo_clean', 'route_lhp_solo'), ('route_zhanshi_fast_clean', 'route_zhanshi_fast')]:
        parent = old['original_variants']['map/diff']['layout_01'][reference]
        assert config['original_variants']['map/diff']['layout_01'][reference] == parent
        entry = config['original_variants']['map/diff']['layout_01'][variant]
        assert len(entry['geometry_verification']) == 3
        assert [(c['saved_scale'], c.get('search_region')) for c in entry['candidates']] == [
            (c['saved_scale'], c.get('search_region')) for c in parent['candidates']]
    assert len(old['original_variants']['map/diff']['layout_01']) == 2  # Input is untouched.


@pytest.mark.parametrize('change', ['source', 'reference_source', 'reference_scale', 'calibration'])
def test_source_version_dependencies_invalidate_after_cached_load(sources, change):
    root, old, _, _ = sources
    config, _ = import_source_variants(old, root)
    save_config(root, config)
    entry, path, _, _, _ = direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')
    if change == 'source':
        path.write_bytes(b'changed clean image')
    elif change == 'reference_source':
        (root/config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['source_image']).write_bytes(b'changed parent')
    elif change == 'reference_scale':
        config['original_variants']['map/diff']['layout_01']['route_lhp_solo']['candidates'][0]['saved_scale'] = 1.2
        save_config(root, config)
    else:
        calibration = next(d['path'] for d in entry['candidates'][0]['dependencies'] if d['path'].endswith('calibration.json'))
        (root/calibration).write_text('changed calibration')
    with pytest.raises(ValueError, match='改变'):
        direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')


def test_new_source_and_color_alpha_have_distinct_caches(sources):
    root, old, image, _ = sources
    replacement = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    replacement[1, 1] = (255, 0, 0)
    write_image(root/'assets/maps/map/diff/sources/route_lhp_solo_clean/1.png', replacement)
    config, _ = import_source_variants(old, root)
    save_config(root, config)
    direct.load_original('map/diff', 'layout_01', 'route_lhp_solo')
    first_key = next(iter(direct._images))
    entry, path, _, _, rgba = direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')
    assert rgba[0, 0, 3] == 0 and rgba[1, 1, 3] == 255
    assert 'route_lhp_solo_clean' in path.parts and next(iter(direct._images)) != first_key
    assert entry['preserve_colored_routes']
    config['original_variants']['map/diff']['layout_01']['route_lhp_solo_clean']['preserve_colored_routes'] = False
    save_config(root, config)
    assert direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')[4][1, 1, 3] == 0


def test_same_size_calibration_change_invalidates_even_with_unchanged_timestamp(sources):
    root, old, _, _ = sources
    config, _ = import_source_variants(old, root)
    save_config(root, config)
    entry = direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')[0]
    dependency = next(d for d in entry['candidates'][0]['dependencies'] if d['path'].endswith('.json'))
    path = root/dependency['path']
    stat = path.stat()
    contents = path.read_bytes()
    path.write_bytes(contents.replace(b'PASS', b'FAIL'))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(ValueError, match='改变'):
        direct.load_original('map/diff', 'layout_01', 'route_lhp_solo_clean')


def test_geometry_rejects_shifted_or_resized_replacement(sources):
    _, _, image, regions = sources
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 0]]), (image.shape[1], image.shape[0]))
    with pytest.raises(ValueError, match='结构位置有变化'):
        verify_identity_geometry(image, shifted, regions)
    with pytest.raises(ValueError, match='尺寸不同'):
        verify_identity_geometry(image, image[:-1], regions)


def test_missing_replacement_fails_without_partial_import(sources):
    root, old, _, _ = sources
    (root/'assets/maps/map/diff/sources/route_zhanshi_fast_clean/1.png').unlink()
    with pytest.raises(ValueError, match='缺少与原图同名'):
        import_source_variants(old, root)
    assert len(old['original_variants']['map/diff']['layout_01']) == 2
