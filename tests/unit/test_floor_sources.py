import hashlib
import json

import cv2
import numpy as np
import pytest

from src.registration.template_matching import floors
from tests.support.frozen_resources import restart_template_resources


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    entries = {}
    for i, floor in enumerate(floors.FLOORS):
        image = np.full((40, 50, 4), (100+i, 120, 130, 255), np.uint8)
        image[0, 0] = (255, 0, 0, 0)
        image[0, 1] = (255, 0, 0, 128)
        path = tmp_path/f'{floor}.png'
        cv2.imencode('.png', image)[1].tofile(path)
        entries[floor] = dict(source_image=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              saved_scale=1+i*.2, size=[50, 40], map_threshold=48)
    config = dict(floor_variants={'map/diff': {'layout_01': {'route_clean': entries}}})
    (tmp_path/'maintenance/configs').mkdir(parents=True)
    path = tmp_path/'maintenance/configs/template_matching.json'
    path.write_text(json.dumps(config))
    return config, path


def test_scale_change_takes_effect_only_after_restart_and_invalidates_old_pose(records):
    config, path = records
    old = floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[0]
    config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']['saved_scale'] = 1.83
    path.write_text(json.dumps(config))
    assert floors.load('map/diff', 'layout_01', 'route_clean', '2nd')[0]['saved_scale'] == 1
    assert floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[0]['saved_scale'] == 1.4
    restart_template_resources()
    assert floors.load('map/diff', 'layout_01', 'route_clean', '2nd')[0]['saved_scale'] == 1
    assert floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[0]['saved_scale'] == 1.83
    with pytest.raises(ValueError, match='重新截图'):
        floors.load('map/diff', 'layout_01', 'route_clean', 'b1', expected_fingerprint=floors.fingerprint(old))


def test_alpha_is_preserved_and_invisible_pixels_cannot_match(records):
    _, _, _, image, rgba = floors.load('map/diff', 'layout_01', 'route_clean', 'b1')
    assert not image[0, 0].any() and rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 128


def test_clean_backed_version_rejects_changed_reference_scale(records):
    from copy import deepcopy
    config, path = records
    variants = config['floor_variants']['map/diff']['layout_01']
    derived = deepcopy(variants['route_clean'])
    for floor, item in derived.items():
        item['reference_floor_variant'] = dict(variant_id='route_clean', floor=floor,
            sha256=floors.fingerprint(variants['route_clean'][floor]))
    variants['route_zhanshi_full_clean'] = derived
    path.write_text(json.dumps(config))
    snapshot = floors.SourceSnapshot('map/diff', 'layout_01', 'route_zhanshi_full_clean')
    variants['route_clean']['1st']['saved_scale'] += .01
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='原底图尺度记录已改变'):
        snapshot.validate()


def test_changed_file_preserves_fixed_image_and_rejects_on_restart(records):
    _, path = records
    before = floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[3].copy()
    (path.parents[2]/'b1.png').write_bytes(b'changed')
    floors._images.clear()  # Force decoding again, not just a decoded-cache hit.
    np.testing.assert_array_equal(floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[3], before)
    restart_template_resources()
    with pytest.raises(ValueError, match='资源已改变'):
        floors.load('map/diff', 'layout_01', 'route_clean', 'b1')


def test_missing_floor_never_falls_back_to_merged_original(records):
    config, path = records
    del config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='完整的单层'):
        floors.sources('map/diff', 'layout_01', 'route_clean')


def test_declared_two_floor_profile_rejects_missing_first_floor_and_basement(records):
    config, path = records
    catalog = dict(schema_version=1, maps=[dict(id='map', name='地图', difficulties=[dict(
        id='diff', name='困难', clean_root='clean', calibration_root='routes',
        floors=['2nd', '1st'], template_versions=['route_clean'])])])
    (path.parents[2]/'configs').mkdir(exist_ok=True)
    (path.parents[2]/'configs/map_catalog.json').write_text(json.dumps(catalog), encoding='utf-8')
    entry = config['floor_variants']['map/diff']['layout_01']['route_clean']
    del entry['b1']
    path.write_text(json.dumps(config))
    snapshot = floors.SourceSnapshot('map/diff', 'layout_01', 'route_clean')
    assert list(snapshot.entries) == ['2nd', '1st']
    with pytest.raises(ValueError, match='单层匹配候选'):
        snapshot.load('b1')
    assert floors.check('map/diff', 'layout_01', 'route_clean') == 2
    del entry['1st']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='完整的单层'):
        snapshot.validate()


def test_pending_scale_allows_asset_audit_but_never_projection_or_stale_images(records):
    config, path = records
    entry = config['floor_variants']['map/diff']['layout_01']['route_clean']['2nd']
    entry.update(saved_scale=None, scale_status='pending')
    path.write_text(json.dumps(config))
    assert floors.check('map/diff', 'layout_01', 'route_clean', require_scale=False) == 3
    with pytest.raises(ValueError, match='屏幕尺度待标定'):
        floors.SourceSnapshot('map/diff', 'layout_01', 'route_clean')
    (path.parents[2]/'2nd.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='资源已改变'):
        floors.check('map/diff', 'layout_01', 'route_clean', require_scale=False)


def test_partial_snapshot_excludes_pending_but_still_validates_it(records):
    config, path = records
    entries = config['floor_variants']['map/diff']['layout_01']['route_clean']
    entries['2nd'].update(saved_scale=None, scale_status='pending')
    path.write_text(json.dumps(config))
    snapshot = floors.SourceSnapshot('map/diff', 'layout_01', 'route_clean', allow_pending=True)
    assert list(snapshot.entries) == ['1st', 'b1']
    assert snapshot.pending == ['2nd']
    with pytest.raises(ValueError, match='单层匹配候选'):
        snapshot.load('2nd')
    entries['2nd'].update(saved_scale=1, scale_status='verified_capture')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='配置已改变'):
        snapshot.validate()
    for entry in entries.values():
        entry.update(saved_scale=None, scale_status='pending')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='屏幕尺度待标定'):
        floors.SourceSnapshot('map/diff', 'layout_01', 'route_clean', allow_pending=True)


def test_parsed_config_reuses_identical_content_but_detects_same_stat_edits(records, monkeypatch):
    import os
    config,path=records
    calls=[]
    parse=json.loads
    def counted(*a,**kw):
        calls.append(1)
        return parse(*a,**kw)
    monkeypatch.setattr(json,'loads',counted)
    first,settings=floors.sources('map/diff','layout_01','route_clean')
    first['b1']['saved_scale']=99  # Returned snapshots cannot poison the parse cache.
    second,_=floors.sources('map/diff','layout_01','route_clean')
    assert second['b1']['saved_scale']==1.4
    assert len(calls)==1
    stamp=path.stat()
    config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']['saved_scale']=1.8
    path.write_text(json.dumps(config))
    os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    third,_=floors.sources('map/diff','layout_01','route_clean')
    assert third['b1']['saved_scale']==1.8 and len(calls)==2


def test_prewarm_populates_three_bounded_images_and_match_templates(records, monkeypatch):
    config,path=records
    config['perf_mode']=2
    path.write_text(json.dumps(config))
    assert floors.prewarm('map/diff','layout_01','route_clean')
    assert len(floors._images)==3
    cached=[floors.load('map/diff','layout_01','route_clean',f)[3] for f in floors.FLOORS]
    monkeypatch.setattr(floors.Image,'open',lambda *a:pytest.fail('Already warmed pixels decoded again'))
    assert floors.prewarm('map/diff','layout_01','route_clean')
    assert all(floors.load('map/diff','layout_01','route_clean',f)[3] is im for f,im in zip(floors.FLOORS,cached))


def test_cancelled_prewarm_never_publishes_decoded_pixels(records, monkeypatch):
    from threading import Event
    cancelled=Event()
    original=floors.cv2.cvtColor
    def cancel_during_decode(*a,**kw):
        cancelled.set()
        return original(*a,**kw)
    monkeypatch.setattr(floors.cv2,'cvtColor',cancel_during_decode)
    assert not floors.prewarm('map/diff','layout_01','route_clean',cancelled=cancelled.is_set)
    assert not floors._images


def test_prewarm_keeps_fixed_config_until_restart(records, monkeypatch):
    config,path=records
    prepare=floors.matcher.prepare_map
    def changed(*a,**kw):
        config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']['saved_scale']=1.8
        path.write_text(json.dumps(config))
        return prepare(*a,**kw)
    monkeypatch.setattr(floors.matcher,'prepare_map',changed)
    assert floors.prewarm('map/diff','layout_01','route_clean')
    assert floors.load('map/diff','layout_01','route_clean','b1')[0]['saved_scale']==1.4
    restart_template_resources()
    assert floors.load('map/diff','layout_01','route_clean','b1')[0]['saved_scale']==1.8
    assert len(floors._images)<=3


@pytest.mark.parametrize('payload,error', [
    (b'{invalid}', json.JSONDecodeError),
    (b'\xef\xbb\xbf{}', json.JSONDecodeError),
    (b'\xff', UnicodeDecodeError),
])
def test_invalid_config_never_falls_back_to_cached_document(records, payload, error):
    _, path = records
    before = path.read_bytes()
    floors.sources('map/diff', 'layout_01', 'route_clean')
    path.write_bytes(payload)
    for _ in range(2):
        with pytest.raises(error):
            floors.sources('map/diff', 'layout_01', 'route_clean')
    path.write_bytes(before)
    assert floors.sources('map/diff', 'layout_01', 'route_clean')[0]['b1']['saved_scale'] == 1.4


def test_config_cache_handles_newlines_and_same_content_at_new_root(records, monkeypatch):
    config, path = records
    for newline in ('\n', '\r\n', '\r'):
        path.write_bytes(json.dumps(config, indent=2).replace('\n', newline).encode('utf-8'))
        assert floors._read_config() == config
    other = path.parents[2]/'other'
    (other/'maintenance/configs').mkdir(parents=True)
    (other/'maintenance/configs/template_matching.json').write_bytes(path.read_bytes())
    monkeypatch.setattr(floors, 'PROJECT_ROOT', other)
    assert floors._read_config() == config
    (other/'maintenance/configs/template_matching.json').unlink()
    with pytest.raises(FileNotFoundError):
        floors._read_config()


def test_directory_link_resolution_rechecks_target_and_root_boundary(tmp_path, monkeypatch):
    from tests.support.directory_links import create_directory_link, remove_directory_link
    root = tmp_path/'root'
    root.mkdir()
    inside, outside = root/'inside', tmp_path/'outside'
    for folder in (inside, outside):
        folder.mkdir()
        (folder/'image.png').write_bytes(b'pixels')
    monkeypatch.setattr(floors, 'PROJECT_ROOT', root)
    link = root/'linked'
    create_directory_link(link, inside)
    try:
        assert floors.source_path('linked/image.png') == inside/'image.png'
        remove_directory_link(link, root=root)
        create_directory_link(link, outside)
        with pytest.raises(ValueError):
            floors.source_path('linked/image.png')
        with pytest.raises(ValueError):
            floors.source_path('../outside/image.png')
    finally:
        remove_directory_link(link, root=root)
    assert (inside/'image.png').read_bytes() == (outside/'image.png').read_bytes() == b'pixels'


def test_dangling_directory_link_rejects_missing_resource(tmp_path, monkeypatch):
    from tests.support.directory_links import create_directory_link, remove_directory_link
    target = tmp_path/'target'
    target.mkdir()
    link = tmp_path/'linked'
    create_directory_link(link, target)
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    # Both paths are confined to this fresh test directory; preserve the target.
    target.rename(tmp_path/'moved')
    try:
        with pytest.raises(FileNotFoundError):
            floors.source_path('linked/image.png')
    finally:
        remove_directory_link(link, root=tmp_path)
