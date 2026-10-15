import hashlib
import json

import cv2
import numpy as np
import pytest

from src.registration.template_matching import floors


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
    (tmp_path/'configs').mkdir()
    path = tmp_path/'configs/template_matching.json'
    path.write_text(json.dumps(config))
    return config, path


def test_scale_change_only_affects_its_floor_and_invalidates_old_pose(records):
    config, path = records
    old = floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[0]
    config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']['saved_scale'] = 1.83
    path.write_text(json.dumps(config))
    assert floors.load('map/diff', 'layout_01', 'route_clean', '2nd')[0]['saved_scale'] == 1
    assert floors.load('map/diff', 'layout_01', 'route_clean', 'b1')[0]['saved_scale'] == 1.83
    with pytest.raises(ValueError, match='重新截图'):
        floors.load('map/diff', 'layout_01', 'route_clean', 'b1', expected_fingerprint=floors.fingerprint(old))


def test_alpha_is_preserved_and_invisible_pixels_cannot_match(records):
    _, _, _, image, rgba = floors.load('map/diff', 'layout_01', 'route_clean', 'b1')
    assert not image[0, 0].any() and rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 128


def test_changed_file_rejects_cached_image(records):
    _, path = records
    floors.load('map/diff', 'layout_01', 'route_clean', 'b1')
    (path.parent.parent/'b1.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='资源已改变'):
        floors.load('map/diff', 'layout_01', 'route_clean', 'b1')


def test_missing_floor_never_falls_back_to_merged_original(records):
    config, path = records
    del config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='完整的单层'):
        floors.sources('map/diff', 'layout_01', 'route_clean')


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


def test_prewarm_config_changed_during_decode_is_rejected(records, monkeypatch):
    config,path=records
    prepare=floors.matcher.prepare_map
    def changed(*a,**kw):
        config['floor_variants']['map/diff']['layout_01']['route_clean']['b1']['saved_scale']=1.8
        path.write_text(json.dumps(config))
        return prepare(*a,**kw)
    monkeypatch.setattr(floors.matcher,'prepare_map',changed)
    with pytest.raises(ValueError,match='配置已改变'):
        floors.prewarm('map/diff','layout_01','route_clean')
    assert floors.load('map/diff','layout_01','route_clean','b1')[0]['saved_scale']==1.8
    assert len(floors._images)<=3
