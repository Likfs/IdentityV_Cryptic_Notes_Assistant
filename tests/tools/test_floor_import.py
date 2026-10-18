import cv2
import hashlib
import json
import shutil
import numpy as np
import pytest
from tools.assets.imports.import_template_floors import verify_padded_clean, clean_scale_overrides, import_floors


def scene():
    image = np.zeros((360, 400, 3), np.uint8)
    rng = np.random.default_rng(90)
    for _ in range(30):
        x, y = map(int, rng.integers(10, 300, 2))
        cv2.rectangle(image, (x, y), (x+36, y+30), (90, 110, 130), -1)
    return image


def test_legacy_floor_import_preserves_profiles_without_montage_records(tmp_path):
    pending = {f: dict(saved_scale=None, scale_status='pending') for f in ('2nd', '1st')}
    config = dict(original_variants={}, floor_variants={'map/hard': {'layout_01': {'route_zhanshi_full': pending}}})
    result = import_floors(config, root=tmp_path)
    assert result['floor_variants'] == config['floor_variants']
    result['floor_variants']['map/hard']['layout_01']['route_zhanshi_full']['2nd']['saved_scale'] = 2
    assert config['floor_variants']['map/hard']['layout_01']['route_zhanshi_full']['2nd']['saved_scale'] is None


@pytest.mark.parametrize('left', [68, 128])
def test_padding_translation_does_not_change_scale(left):
    clean = scene()
    image = cv2.copyMakeBorder(clean, 128, 128, left, 128, cv2.BORDER_CONSTANT)
    result = verify_padded_clean(clean, image)
    assert result['clean_to_source'] == [[1, 0, left], [0, 1, 128], [0, 0, 1]]


@pytest.mark.parametrize('transform', ['shift', 'scale'])
def test_changed_structure_cannot_inherit_scale_by_hash_only(transform):
    clean = scene()
    image = cv2.copyMakeBorder(clean, 128, 128, 128, 128, cv2.BORDER_CONSTANT)
    matrix = np.float64([[1, 0, 35], [0, 1, 20]]) if transform == 'shift' else np.float64([[1.1, 0, 0], [0, 1.1, 0]])
    image = cv2.warpAffine(image, matrix, image.shape[1::-1])
    with pytest.raises(ValueError):
        verify_padded_clean(clean, image)


def test_reviewed_clean_scale_is_converted_to_reference_units():
    config = dict(entries={'clean/b1.png': dict(saved_scale=1.8)}, original_variants={
        'map/diff': {'layout_01': {'route_clean': {'candidates': [dict(
            floor='b1', saved_scale=1.6, reference_entry_key='clean/b1.png')]}}}})
    key = ('map/diff', 'layout_01', 'b1')
    result = clean_scale_overrides(config, {key: 1.48})
    assert result[key] == pytest.approx(1.665)
    # Route pixels at a different scale inherit a ratio, not a copied clean value.
    assert .96 * result[key] / 1.8 == pytest.approx(.888)
    assert config['entries']['clean/b1.png']['saved_scale'] == 1.8


@pytest.mark.parametrize('failure', [None, 'missing', 'shift', 'reference_changed'])
def test_two_floor_clean_variant_requires_verified_geometry(tmp_path, failure):
    from types import SimpleNamespace
    from tools.assets.imports.import_clean_floor_variant import import_variant
    profile = SimpleNamespace(key='map/hard', floors=('2nd', '1st'))
    base = tmp_path/'assets/maps/map/hard'
    sources = base/'routes/route_full_clean_single_floor'
    records = {}
    clean = scene()
    for floor in profile.floors:
        path = base/f'clean/layout_05/layout_05_{floor}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', clean)[1].tofile(path)
        records[floor] = dict(source_image=path.relative_to(tmp_path).as_posix(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=[400, 360],
            saved_scale=3.178 if floor=='1st' else 3.174, scale_status='verified_capture')
        path = sources/f'layout_05/layout_05_{floor}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        padded = cv2.copyMakeBorder(clean, 256, 256, 256, 256, cv2.BORDER_CONSTANT)
        if failure == 'shift' and floor == '1st':
            padded = cv2.warpAffine(padded, np.float32([[1,0,40],[0,1,20]]), padded.shape[1::-1])
        cv2.imencode('.png', padded)[1].tofile(path)
    config = dict(floor_variants={profile.key: {'layout_05': {'route_clean': records}},
                                 'map/other': {'preserve': {}}})
    if failure == 'missing':
        (sources/'layout_05/layout_05_1st.png').unlink()
    if failure == 'reference_changed':
        records['1st']['source_sha256'] = 'stale'
    if failure:
        with pytest.raises(ValueError):
            import_variant(config, profile, sources, 'route_full_clean', border=256, root=tmp_path)
        return
    result = import_variant(config, profile, sources, 'route_full_clean', border=256, root=tmp_path)
    assert 'route_full_clean' not in config['floor_variants'][profile.key]['layout_05']
    assert result['floor_variants']['map/other'] == config['floor_variants']['map/other']
    for floor, item in result['floor_variants'][profile.key]['layout_05']['route_full_clean'].items():
        assert item['saved_scale'] == records[floor]['saved_scale']
        assert item['geometry_verification']['clean_to_source'] == [[1,0,256],[0,1,256],[0,0,1]]
        assert item['scale_status'] == 'derived_geometry'
        assert item['reference_floor_variant']['floor'] == floor


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_clean_scale_cannot_be_published(value):
    with pytest.raises(ValueError, match='有限正数'):
        clean_scale_overrides({}, {('map/diff', 'layout_01', 'b1'): value})


@pytest.mark.parametrize('rectified', [False, True])
def test_clean_scale_updates_all_versions_and_survives_reimport(tmp_path, rectified):
    versions = ('route_clean', 'route_zhanshi_fast', 'route_lhp_solo',
                'route_zhanshi_fast_clean', 'route_lhp_solo_clean')
    base = tmp_path/'assets/maps/map/diff'
    offline = tmp_path/'maintenance/maps/map/diff'
    clean = scene()
    def image(path, pixels):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', pixels)[1].tofile(path)
        return path.relative_to(tmp_path).as_posix()
    clean_records, lhp_records, entries = [], [], {}
    calibrations = {v: {'floors': {}} for v in ('route_zhanshi_fast', 'route_lhp_solo')}
    for floor in ('2nd', '1st', 'b1'):
        clean_path = base/f'clean/layout_01/layout_01_{floor}.png'
        key = image(clean_path, clean)
        entries[key] = dict(saved_scale=1.8)
        clean_records.append(dict(floor=floor, saved_scale=1.6, reference_entry_key=key,
            dependencies=[dict(path=key, sha256=hashlib.sha256(clean_path.read_bytes()).hexdigest())]))
        lhp_records.append(dict(floor=floor, saved_scale=.96))
        for version in calibrations:
            asset = image(base/f'routes/{version}/layout_01/{floor}.png',
                          np.dstack((clean, np.full(clean.shape[:2], 255, np.uint8))))
            calibrations[version]['floors'][floor] = dict(status='PASS', asset=asset,
                asset_to_clean_homography=np.eye(3).tolist(), source_crop_xywh=[0, 0, 400, 360])
        for version in ('route_zhanshi_fast_clean', 'route_lhp_solo_clean'):
            image(base/f'routes/{version}_single_floor/01/01_{floor}.png',
                  cv2.copyMakeBorder(clean, 128, 128, 128, 128, cv2.BORDER_CONSTANT))
    for version, data in calibrations.items():
        calibration = offline/f'routes/{version}/layout_01/calibration.json'
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text(json.dumps(data))
    variants = {v: {} for v in versions}
    variants['route_clean']['candidates'] = clean_records
    variants['route_lhp_solo']['candidates'] = lhp_records
    initial = import_floors(dict(entries=entries, original_variants={'map/diff': {'layout_01': variants}}),
                            root=tmp_path)
    if rectified:
        from tools.assets.calibration.rectify_single_floor_routes import stage
        manifests = stage(initial, tmp_path/'staging', root=tmp_path)
        relative = 'assets/maps/map/diff/routes/route_lhp_solo_rectified'
        shutil.copytree(tmp_path/'staging'/relative, tmp_path/relative)
        metadata = tmp_path/manifests['map/diff']
        metadata.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path/'staging'/manifests['map/diff'], metadata)
        initial = import_floors(initial, root=tmp_path)
    before = initial['floor_variants']['map/diff']['layout_01']
    result = import_floors(initial, root=tmp_path, clean_scales={('map/diff', 'layout_01', 'b1'): 1.48})
    after = result['floor_variants']['map/diff']['layout_01']
    for version in versions:
        expected = .888 if version == 'route_lhp_solo' and not rectified else 1.48
        assert after[version]['b1']['saved_scale'] == pytest.approx(expected)
        assert after[version]['1st'] == before[version]['1st']
        assert after[version]['2nd'] == before[version]['2nd']
    assert import_floors(result, root=tmp_path) == result
    if rectified:
        # Merely updating a hash must not bless an image inconsistent with calibration.
        target = tmp_path/after['route_lhp_solo']['b1']['source_image']
        pixels = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_UNCHANGED)
        pixels[150, 150] = (255, 255, 255, 255)
        cv2.imencode('.png', pixels)[1].tofile(target)
        manifest_path = tmp_path/manifests['map/diff']
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['layouts']['layout_01']['b1']['source_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with pytest.raises(ValueError, match='不符合原始标定几何'):
            import_floors(result, root=tmp_path)
