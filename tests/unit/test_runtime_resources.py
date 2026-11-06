import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import runtime_resources as runtime
from src.registration.template_matching import floors
from tests.support.published_records import write_bundle, write_template, json_bytes


@pytest.fixture
def package(tmp_path, monkeypatch):
    path = tmp_path/'assets/maps/map/diff/clean/layout_01/layout_01_1st.png'
    path.parent.mkdir(parents=True)
    cv2.imencode('.png', np.full((30,40,3),120,np.uint8))[1].tofile(path)
    name = path.relative_to(tmp_path).as_posix()
    item = dict(source_image=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=[40,30], saved_scale=1., scale_status='verified_capture')
    payload = dict(template=dict(perf_mode=2,asset_contract='aligned_floor_images',floor_variants={'map/diff':{'layout_01':{
        'clean':{f:dict(item) for f in floors.DEFAULT_FLOORS}}}}),
        files={name:dict(bytes=path.stat().st_size,sha256=item['source_sha256'])},
        recognition={'assets/maps/map/diff/clean':dict(source_images=True,profile_key='map/diff',targets={
            'layout_01/layout_01_1st.png':dict(source_sha256=item['source_sha256'])})})
    write_template(tmp_path,payload['template'])
    write_bundle(tmp_path,payload)
    monkeypatch.setattr(runtime,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(floors,'PROJECT_ROOT',tmp_path)
    return tmp_path,path,payload


def test_package_matches_without_any_maintenance_document(package):
    root,path,_=package
    assert not (root/'maintenance/configs/template_matching.json').exists()
    assert not (path.parents[2]/'recognition_scales.json').exists()
    assert runtime.runtime_files()==sorted([path,root/runtime.BUNDLE_PATH,root/runtime.SCALE_PATH])
    assert floors.load('map/diff','layout_01','clean','1st')[3].shape==(30,40,3)
    from src.structure_layout.scales import read_scale_overrides
    assert read_scale_overrides(path.parents[1])['layout_01/layout_01_1st.png']['scales']==[1.]


@pytest.mark.parametrize('change',['image','missing','bundle','escape'])
def test_runtime_package_rejects_broken_actual_dependencies(package,change):
    root,path,payload=package
    if change=='image':path.write_bytes(path.read_bytes()+b'changed')
    elif change=='missing':path.unlink()
    elif change=='bundle':
        bundle=root/runtime.BUNDLE_PATH
        bundle.write_text(bundle.read_text().replace('"perf_mode": 2','"perf_mode": 3'))
    else:
        outside=root.parent/(root.name+'_outside.txt');outside.write_bytes(b'data')
        payload['files']={'../'+outside.name:dict(bytes=4,sha256=hashlib.sha256(b'data').hexdigest())}
        write_bundle(root,payload)
    with pytest.raises((ValueError,OSError)):
        runtime.runtime_files()


def test_offline_edits_do_not_change_runtime_but_explicit_audit_reads_them(package):
    root,_,_=package
    (root/'maintenance/configs').mkdir(parents=True)
    (root/'maintenance/configs/template_matching.json').write_text('{}')
    assert floors.load('map/diff','layout_01','clean','1st')[0]['saved_scale']==1.
    assert floors.sources('map/diff','layout_01','clean')[0]['1st']['saved_scale']==1.
    (root/runtime.BUNDLE_PATH).write_bytes(b'{}')
    with pytest.raises(ValueError,match='运行资源配置'):
        floors.sources('map/diff','layout_01','clean')


def test_runtime_rejects_rehashed_inconsistent_zoom_scales(package):
    root,path,payload=package
    profile=payload['recognition']['assets/maps/map/diff/clean']
    profile['factor_to_minimum']=.4
    profile['targets']['layout_01/layout_01_1st.png'].update(scales=[1.,.5],scale_modes={'minimum':1.,'50_percent':.5})
    write_bundle(root,payload)
    from src.structure_layout.scales import read_scale_overrides
    with pytest.raises(ValueError,match='不能重复保存尺度'):
        read_scale_overrides(path.parents[1])


def test_runtime_rejects_wrong_dimensions_even_with_valid_source_hash(package):
    root,_,payload=package
    for item in payload['template']['floor_variants']['map/diff']['layout_01']['clean'].values():
        item['size']=[41,30]
    write_bundle(root,payload)
    with pytest.raises(ValueError,match='尺寸'):
        floors.load('map/diff','layout_01','clean','1st')


def test_formal_entry_requires_published_bundle_even_when_maintenance_exists(package,monkeypatch,capsys):
    import sys
    from src.__main__ import main
    root,_,payload=package
    (root/runtime.BUNDLE_PATH).unlink()
    (root/'maintenance/configs').mkdir(parents=True)
    (root/'maintenance/configs/template_matching.json').write_text(json.dumps(payload['template']))
    monkeypatch.setattr(sys,'argv',['assistant','--check'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '缺少运行资源清单' in capsys.readouterr().err


def test_grouped_scales_changes_only_on_restart_and_preserves_matcher_records(package):
    root,_,payload=package
    expected=payload['template']
    assert runtime.template_config()==expected
    assert floors.load('map/diff','layout_01','clean','1st')[0]['saved_scale']==1.
    for record in payload['template']['floor_variants']['map/diff']['layout_01']['clean'].values():
        record['saved_scale']=1.2345678912345
    write_bundle(root,payload)
    assert floors.load('map/diff','layout_01','clean','1st')[0]['saved_scale']==1.
    floors._close_fixed_session()
    assert floors.load('map/diff','layout_01','clean','1st')[0]['saved_scale']==1.2345678912345


@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf'),'1.0',None])
def test_grouped_scales_rejects_invalid_scale_after_manifest_rehashed(package,value):
    root,_,_=package
    bundle=json.loads((root/runtime.BUNDLE_PATH).read_text())
    scales=json.loads((root/runtime.SCALE_PATH).read_text())
    scales['maps']['map']['diff']['minimum_scale']=value
    raw=json_bytes(scales);(root/runtime.SCALE_PATH).write_bytes(raw)
    bundle['payload']['files'][runtime.SCALE_PATH]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    bundle['payload_sha256']=runtime.digest(bundle['payload'])
    (root/runtime.BUNDLE_PATH).write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='scale'):
        runtime.template_config()


@pytest.mark.parametrize('change',['missing','changed','duplicate','extra','missing_entry'])
def test_grouped_scales_integrity_and_exact_image_coverage(package,change):
    root,_,_=package
    path=root/runtime.SCALE_PATH
    if change=='missing':
        path.unlink()
    else:
        raw=path.read_bytes()
        if change=='duplicate':raw=raw.replace(b'"minimum_scale": 1.0',b'"minimum_scale": 1.0, "minimum_scale": 2.0')
        elif change=='changed':raw=raw.replace(b'1.0',b'2.0')
        else:
            data=json.loads(raw)
            if change=='extra':data['maps']['extra']={'diff':{'minimum_scale':1.}}
            else:data['maps'].clear()
            raw=json_bytes(data)
        path.write_bytes(raw)
        if change!='changed':
            bundle=json.loads((root/runtime.BUNDLE_PATH).read_text())
            bundle['payload']['files'][runtime.SCALE_PATH]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            bundle['payload_sha256']=runtime.digest(bundle['payload'])
            (root/runtime.BUNDLE_PATH).write_text(json.dumps(bundle))
    with pytest.raises((ValueError,OSError)):
        runtime.template_config()


def test_pending_default_scale_still_refuses_projection(package):
    root,_,payload=package
    for record in payload['template']['floor_variants']['map/diff']['layout_01']['clean'].values():
        record.update(saved_scale=None,scale_status='pending')
    write_bundle(root,payload)
    with pytest.raises(ValueError,match='scale'):
        floors.load('map/diff','layout_01','clean','1st')


def test_pair_publish_failure_restores_previous_scale_table(tmp_path,monkeypatch):
    from tools.assets.imports import export_runtime_resources as exporter
    bundle=tmp_path/'runtime_resources.json';sidecar=tmp_path/'scale_config.json'
    bundle.write_bytes(b'old manifest');sidecar.write_bytes(b'old scales')
    original=exporter.os.replace
    def fail_manifest(src,dst):
        if Path(dst)==bundle:raise OSError('simulated publish failure')
        return original(src,dst)
    monkeypatch.setattr(exporter.os,'replace',fail_manifest)
    with pytest.raises(OSError,match='simulated'):
        exporter.publish_package(bundle,{'new':'manifest'},b'new scales')
    assert bundle.read_bytes()==b'old manifest'
    assert sidecar.read_bytes()==b'old scales'
    assert not list(tmp_path.glob('*.tmp'))


def test_unpublished_scale_edit_rejects_fresh_recognition_and_projection(package):
    root, path, _ = package
    scales = root/runtime.SCALE_PATH
    scales.write_bytes(scales.read_bytes().replace(b'1.0', b'2.0'))
    for read in (runtime.template_config, lambda: runtime.recognition_profile(path.parents[1])):
        with pytest.raises(ValueError, match='尺度配置与已发布资源不一致'):
            read()


def test_manifest_publish_preserves_editable_scale_file(package, monkeypatch):
    from tools.assets.imports import export_runtime_resources as exporter
    root, _, _ = package
    sidecar = root/runtime.SCALE_PATH
    before = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
    replace = exporter.os.replace
    def guarded(source, destination):
        assert Path(destination) != sidecar
        return replace(source, destination)
    monkeypatch.setattr(exporter.os, 'replace', guarded)
    exporter.publish_package(root/runtime.BUNDLE_PATH, {'new': 'manifest'}, before[0], preserve_scales=True)
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before
    sidecar.write_bytes(before[0]+b' ')
    with pytest.raises(ValueError, match='尺度配置发生变化'):
        exporter.publish_package(root/runtime.BUNDLE_PATH, {}, before[0], preserve_scales=True)
    assert sidecar.read_bytes() == before[0]+b' '
