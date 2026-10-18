import json
import hashlib
import cv2
import numpy as np
import pytest
from src.structure_layout.normalized import read_normalized_index, original_pose
from src.structure_layout.scales import read_scale_overrides, check_scale_overrides
from src.structure_layout.retriever import PureStructureRetriever
from tests.support.normalized_index import make_profile
from tools.assets.imports import build_recognition_index as builder


@pytest.fixture
def profile(tmp_path, monkeypatch):
    clean = make_profile(tmp_path)
    monkeypatch.setattr(builder, 'project_relative', lambda p:'assets/test/clean')
    return clean


def test_normalized_assets_preserve_originals_and_share_two_scales(profile):
    paths = list(profile.glob('layout_*/*.png'))+[profile.parent/'recognition_scales.json']
    before = {p:p.read_bytes() for p in paths}
    staged = profile.parent/'staged'
    data = builder.build(profile, staged)
    assert not (profile.parent/'recognition').exists()
    builder.publish(profile, staged)
    assert {p:p.read_bytes() for p in paths} == before
    assert not check_scale_overrides(profile)['errors']
    retriever = PureStructureRetriever(profile, floors=('1st','2nd'), workers=2)
    try:
        retriever.prepare()
        assert {t['scales'] for t in retriever._targets} == {tuple(data['scale_modes'].values())}
        assert len(set(s for t in retriever._targets for s in t['scales'])) == 2
        assert all(t['full'][0].shape[0] < 110 for t in retriever._targets)
        for target in retriever._targets:
            matrix = np.array(target['original_to_index'])
            source_xy = np.array([123.25, 87.75, 1.])
            index_xy = matrix @ source_xy
            pose = original_pose(dict(scale=1.5,x=index_xy[0],y=index_xy[1]), target)
            np.testing.assert_allclose([pose['x'],pose['y']],source_xy[:2],atol=1e-10)
            assert pose['scale'] == pytest.approx(read_scale_overrides(profile)[target['path']]['scales'][0])
            assert target['path'] in data['targets'] and target['sha256']==data['targets'][target['path']]['source_sha256']
    finally:retriever.close()


@pytest.mark.parametrize('change',['geometry','size','image','source','scales','missing'])
def test_normalized_index_rejects_changed_dependencies_and_fake_geometry(profile,change):
    root = profile.parent/'recognition';data=builder.build(profile,root)
    name=next(iter(data['targets']));manifest=root/'index.json'
    if change=='geometry':data['targets'][name]['original_to_index'][0][2]+=1
    elif change=='size':data['targets'][name]['size'][0]+=1
    elif change=='image':(root/name).write_bytes((root/name).read_bytes()+b'changed')
    elif change=='source':(profile/name).write_bytes((profile/name).read_bytes()+b'changed')
    elif change=='scales':data['scale_modes']['minimum']+=.01
    else:del data['targets'][name]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):read_normalized_index(profile,read_scale_overrides(profile))
    assert check_scale_overrides(profile)['errors']


@pytest.mark.parametrize('change',['index','source'])
def test_index_change_during_search_cannot_publish_candidates(profile,monkeypatch,change):
    from src.structure_layout import retriever as module
    root=profile.parent/'recognition';builder.build(profile,root)
    def changed(*args,**kwargs):
        manifest=root/'index.json';data=json.loads(manifest.read_text());name=next(iter(data['targets']))
        if change=='source':
            path=profile/name;path.write_bytes(path.read_bytes()+b'changed during search')
        else:
            data['targets'][name]['original_to_index'][0][2]+=1
            manifest.write_text(json.dumps(data))
        return {'floor_candidates':[]}
    monkeypatch.setattr(module,'search',changed)
    retriever=PureStructureRetriever(profile,floors=('1st','2nd'))
    try:
        result=retriever.recognize(np.full((80,90,3),100,np.uint8),(90,80))
        assert not result.candidates and '识别小图' in result.reasons[0]
    finally:retriever.close()


def test_source_snapshot_is_not_reused_across_requests(profile):
    root=profile.parent/'recognition';builder.build(profile,root)
    retriever=PureStructureRetriever(profile,floors=('1st','2nd'))
    try:
        retriever.prepare()
        source=next(profile.glob('layout_*/*.png'))
        source.write_bytes(source.read_bytes()+b'changed after preparation')
        with pytest.raises(ValueError,match='指纹已改变'):
            retriever.prepare()
    finally:retriever.close()


def test_publish_rejects_rehashed_wrong_pixels_and_never_overwrites(profile):
    staged=profile.parent/'staged';data=builder.build(profile,staged)
    name=next(iter(data['targets']));path=staged/name
    image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    image[30:40,30:40]=0
    content=cv2.imencode('.png',image)[1].tobytes();path.write_bytes(content)
    data['targets'][name]['sha256']=hashlib.sha256(content).hexdigest()
    (staged/'index.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='像素与原图几何'):
        builder.publish(profile,staged)
    assert not (profile.parent/'recognition').exists()
    (profile.parent/'recognition').mkdir()
    with pytest.raises(ValueError,match='已存在'):
        builder.publish(profile,staged)
