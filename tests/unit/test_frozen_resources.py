import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pytest

from src.registration.template_matching import direct, floors, matcher
from tests.support.frozen_resources import restart_template_resources


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(floors, 'PROJECT_ROOT', tmp_path)
    entries = {}
    for index, floor in enumerate(floors.FLOORS):
        pixels = np.full((40,50,4), (100+index,130,170,255), np.uint8)
        path = tmp_path/f'{floor}.png'
        cv2.imencode('.png',pixels)[1].tofile(path)
        entries[floor] = dict(source_image=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              size=[50,40], saved_scale=1., map_threshold=48)
    config = dict(perf_mode=2, floor_variants={'map/diff':{'layout_01':{
        'route_clean':entries, 'route_lhp_solo':json.loads(json.dumps(entries))}}})
    (tmp_path/'maintenance/configs').mkdir(parents=True)
    path = tmp_path/'maintenance/configs/template_matching.json'
    path.write_text(json.dumps(config),encoding='utf-8')
    return tmp_path,config,path


def test_concurrent_requests_and_versions_share_one_full_verification(frozen_inputs, monkeypatch):
    root,_,config_path = frozen_inputs
    calls=Counter();lock=Lock();read=Path.read_bytes
    def counted(path):
        with lock:calls[path]+=1
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',counted)
    def load(version):
        return floors.load('map/diff','layout_01',version,'1st')[3]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(load,['route_clean','route_lhp_solo']))
    np.testing.assert_array_equal(*results)
    assert calls[config_path] == 1
    assert all(calls[root/f'{floor}.png']==1 for floor in floors.FLOORS)
    before=calls.copy()
    floors._images.clear()
    matcher._map_cache.clear()
    load('route_clean')
    assert calls==before


def test_failed_initial_batch_publishes_no_partial_resources_and_can_retry(frozen_inputs):
    root,config,path=frozen_inputs
    dependency=root/'evidence.json';dependency.write_bytes(b'{"scale":1}')
    expected=hashlib.sha256(dependency.read_bytes()).hexdigest()
    for item in config['floor_variants']['map/diff']['layout_01']['route_clean'].values():
        item['dependencies']=[dict(path=dependency.name,sha256=expected)]
    path.write_text(json.dumps(config))
    dependency.write_bytes(b'{"scale":2}')
    with pytest.raises(ValueError,match='资源已改变'):
        floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean')
    session=floors.fixed_session()
    assert not session._files and not session._records
    assert session._file.seek(0,2)==0
    dependency.write_bytes(b'{"scale":1}')
    assert floors.load('map/diff','layout_01','route_clean','1st')[3].size
    assert len(session._files)==4


def test_initial_verification_reads_content_despite_warm_live_stat_cache(frozen_inputs):
    root,_,_=frozen_inputs
    floors.sources('map/diff','layout_01','route_clean')
    path=root/'1st.png';stamp=path.stat();raw=bytearray(path.read_bytes())
    raw[-1]^=1
    path.write_bytes(raw);os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    with pytest.raises(ValueError,match='资源已改变'):
        floors.load('map/diff','layout_01','route_clean','1st')


def test_verified_bytes_survive_replacement_before_first_decode(frozen_inputs):
    root,_,_=frozen_inputs
    snapshot=floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean')
    original=cv2.imread(str(root/'1st.png'))
    (root/'1st.png').unlink()
    np.testing.assert_array_equal(snapshot.load('1st')[3],original)
    restart_template_resources()
    with pytest.raises(FileNotFoundError):
        floors.load('map/diff','layout_01','route_clean','1st')


def test_frozen_metadata_cannot_be_poisoned_by_request_copy(frozen_inputs):
    snapshot=floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean')
    snapshot.entries['1st']['saved_scale']=999
    snapshot.config['perf_mode']=0
    other=floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean')
    assert other.entries['1st']['saved_scale']==1.
    assert other.config['perf_mode']==2


def test_catalog_and_version_list_change_only_after_restart(frozen_inputs):
    root,config,path=frozen_inputs
    catalog=dict(schema_version=1,maps=[dict(id='map',name='map',difficulties=[dict(
        id='diff',name='diff',clean_root='clean',calibration_root='routes',floors=list(floors.FLOORS))])])
    catalog_path=root/'configs/map_catalog.json'
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog))
    assert direct.available_variants('map/diff','layout_01')==['route_lhp_solo','route_clean']
    versions=config['floor_variants']['map/diff']['layout_01']
    versions.pop('route_lhp_solo')
    versions['route_clean'].pop('b1')
    catalog['maps'][0]['difficulties'][0]['floors']=['2nd','1st']
    path.write_text(json.dumps(config));catalog_path.write_text(json.dumps(catalog))
    assert direct.available_variants('map/diff','layout_01')==['route_lhp_solo','route_clean']
    assert list(floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean').entries)==list(floors.FLOORS)
    restart_template_resources()
    assert direct.available_variants('map/diff','layout_01')==['route_clean']
    assert list(floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean').entries)==['2nd','1st']


def test_close_releases_process_owned_snapshot_file(frozen_inputs):
    floors.FrozenSourceSnapshot('map/diff','layout_01','route_clean')
    session=floors.fixed_session();file=session._file
    assert not file.closed and file.seek(0,2)>0
    restart_template_resources()
    assert file.closed and session._file is None and not session._files


def test_new_process_reads_updated_scale(frozen_inputs):
    import subprocess
    import sys
    root,config,path=frozen_inputs
    script=('from pathlib import Path; import sys; '
            'from src.registration.template_matching import floors; '
            'floors.PROJECT_ROOT=Path(sys.argv[1]); '
            "print(floors.load('map/diff','layout_01','route_clean','1st')[0]['saved_scale'])")
    def load():
        result=subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True,text=True,check=True)
        return float(result.stdout.strip())
    assert load()==1.
    config['floor_variants']['map/diff']['layout_01']['route_clean']['1st']['saved_scale']=1.75
    path.write_text(json.dumps(config))
    assert load()==1.75
