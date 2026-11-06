from tests.support.published_records import write_template
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
    for i, floor in enumerate(floors.DEFAULT_FLOORS):
        image = np.full((40, 50, 4), (100+i, 120, 130, 255), np.uint8)
        image[0, 0] = (255, 0, 0, 0)
        image[0, 1] = (255, 0, 0, 128)
        path = tmp_path/f'{floor}.png'
        cv2.imencode('.png', image)[1].tofile(path)
        entries[floor] = dict(source_image=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              saved_scale=1.4, size=[50, 40], map_threshold=48)
    config = dict(floor_variants={'map/diff': {'layout_01': {'clean': entries}}})
    path = write_template(tmp_path, config)
    return config, path


def test_scale_change_takes_effect_only_after_restart_and_invalidates_old_pose(records):
    config, path = records
    old = floors.load('map/diff', 'layout_01', 'clean', 'b1')[0]
    for item in config['floor_variants']['map/diff']['layout_01']['clean'].values():
        item['saved_scale'] = 1.83
    write_template(path.parent.parent, config)
    assert floors.load('map/diff', 'layout_01', 'clean', '2nd')[0]['saved_scale'] == 1.4
    assert floors.load('map/diff', 'layout_01', 'clean', 'b1')[0]['saved_scale'] == 1.4
    restart_template_resources()
    assert floors.load('map/diff', 'layout_01', 'clean', '2nd')[0]['saved_scale'] == 1.83
    assert floors.load('map/diff', 'layout_01', 'clean', 'b1')[0]['saved_scale'] == 1.83
    with pytest.raises(ValueError, match='重新截图'):
        floors.load('map/diff', 'layout_01', 'clean', 'b1', expected_fingerprint=floors.fingerprint(old))


def test_alpha_is_preserved_and_invisible_pixels_cannot_match(records):
    _, _, _, image, rgba = floors.load('map/diff', 'layout_01', 'clean', 'b1')
    assert not image[0, 0].any() and rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 128


def test_changed_file_preserves_fixed_image_and_rejects_on_restart(records):
    _, path = records
    before = floors.load('map/diff', 'layout_01', 'clean', 'b1')[3].copy()
    (path.parent.parent/'b1.png').write_bytes(b'changed')
    floors._images.clear()  # Force decoding again, not just a decoded-cache hit.
    np.testing.assert_array_equal(floors.load('map/diff', 'layout_01', 'clean', 'b1')[3], before)
    restart_template_resources()
    with pytest.raises(ValueError, match='资源已改变'):
        floors.load('map/diff', 'layout_01', 'clean', 'b1')


def test_missing_floor_never_falls_back_to_merged_original(records):
    config, path = records
    del config['floor_variants']['map/diff']['layout_01']['clean']['b1']
    write_template(path.parent.parent, config)
    with pytest.raises(ValueError, match='完整的单层'):
        floors.sources('map/diff', 'layout_01', 'clean')


def test_declared_two_floor_profile_rejects_missing_first_floor_and_basement(records):
    config, path = records
    catalog_path = path.parent/'map_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog['maps'][0]['difficulties'][0]['floors'] = ['2nd', '1st']
    catalog_path.write_text(json.dumps(catalog), encoding='utf-8')
    entry = config['floor_variants']['map/diff']['layout_01']['clean']
    del entry['b1']
    write_template(path.parent.parent, config)
    snapshot = floors.SourceSnapshot('map/diff', 'layout_01', 'clean')
    assert list(snapshot.entries) == ['2nd', '1st']
    with pytest.raises(ValueError, match='单层匹配候选'):
        snapshot.load('b1')
    assert floors.check('map/diff', 'layout_01', 'clean') == 2
    del entry['1st']
    write_template(path.parent.parent, config)
    with pytest.raises(ValueError, match='完整的单层'):
        snapshot.validate()


def test_prewarm_populates_three_bounded_images_and_match_templates(records, monkeypatch):
    config,path=records
    config['perf_mode']=2
    write_template(path.parent.parent, config)
    assert floors.prewarm('map/diff','layout_01','clean')
    assert len(floors._images)==3
    cached=[floors.load('map/diff','layout_01','clean',f)[3] for f in floors.DEFAULT_FLOORS]
    monkeypatch.setattr(floors.Image,'open',lambda *a:pytest.fail('Already warmed pixels decoded again'))
    assert floors.prewarm('map/diff','layout_01','clean')
    assert all(floors.load('map/diff','layout_01','clean',f)[3] is im for f,im in zip(floors.DEFAULT_FLOORS,cached))


def test_cancelled_prewarm_never_publishes_decoded_pixels(records, monkeypatch):
    from threading import Event
    cancelled=Event()
    original=floors.cv2.cvtColor
    def cancel_during_decode(*a,**kw):
        cancelled.set()
        return original(*a,**kw)
    monkeypatch.setattr(floors.cv2,'cvtColor',cancel_during_decode)
    assert not floors.prewarm('map/diff','layout_01','clean',cancelled=cancelled.is_set)
    assert not floors._images


def test_prewarm_keeps_fixed_config_until_restart(records, monkeypatch):
    config,path=records
    prepare=floors.matcher.prepare_map
    def changed(*a,**kw):
        for item in config['floor_variants']['map/diff']['layout_01']['clean'].values():
            item['saved_scale']=1.8
        write_template(path.parent.parent, config)
        return prepare(*a,**kw)
    monkeypatch.setattr(floors.matcher,'prepare_map',changed)
    assert floors.prewarm('map/diff','layout_01','clean')
    assert floors.load('map/diff','layout_01','clean','b1')[0]['saved_scale']==1.4
    restart_template_resources()
    assert floors.load('map/diff','layout_01','clean','b1')[0]['saved_scale']==1.8
    assert len(floors._images)<=3


@pytest.mark.parametrize('payload,error', [
    (b'{invalid}', json.JSONDecodeError),
    (b'\xef\xbb\xbf{}', json.JSONDecodeError),
    (b'\xff', UnicodeDecodeError),
])
def test_invalid_config_never_falls_back_to_cached_document(records, payload, error):
    _, path = records
    before = path.read_bytes()
    floors.sources('map/diff', 'layout_01', 'clean')
    path.write_bytes(payload)
    for _ in range(2):
        with pytest.raises(error):
            floors.sources('map/diff', 'layout_01', 'clean')
    path.write_bytes(before)
    assert floors.sources('map/diff', 'layout_01', 'clean')[0]['b1']['saved_scale'] == 1.4


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


def test_working_set_budget_uses_decoded_pixels_and_unique_sources(monkeypatch):
    entry = dict(source_image='same.png', source_sha256='verified', size=[100, 100])
    monkeypatch.setattr(floors, 'DECODED_IMAGE_BYTES', 100 * 100 * 7)
    assert floors.working_set_fits([entry, dict(entry)])
    assert not floors.working_set_fits([entry, dict(entry, source_image='another.png')])
    monkeypatch.setattr(floors, 'DECODED_IMAGE_BYTES', 1024 * 1024)
    assert floors.working_set_fits([dict(entry, source_image=str(i)) for i in range(8)])
    assert not floors.working_set_fits([dict(entry, source_image=str(i)) for i in range(9)])
