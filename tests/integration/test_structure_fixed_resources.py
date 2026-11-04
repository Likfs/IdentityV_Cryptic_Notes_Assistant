"""Published recognition resources stay fixed; failed batches never replace an index."""
from pathlib import Path
from threading import Event
import json
import shutil

import cv2
import numpy as np
import pytest

from src.layout_retrieval import create_layout_retriever
from src.recognition_control import RecognitionCancelled
from src.structure_layout import retriever as module
from src.structure_layout.scales import check_scale_overrides
from src.runtime_resources import BUNDLE_PATH, digest
from tests.support.aligned_runtime import activate
from tests.support.structure_scene import publish_structure
from tests.support.directory_links import create_directory_link, remove_directory_link


@pytest.fixture
def fixed_scene(tmp_path,monkeypatch):
    source=tmp_path/'input/layout_05';source.mkdir(parents=True)
    image=np.full((240,280,3),25,np.uint8)
    cv2.rectangle(image,(30,30),(220,205),(100,100,100),-1)
    cv2.rectangle(image,(30,30),(220,205),(175,175,175),3)
    cv2.line(image,(90,30),(90,170),(175,175,175),3)
    for floor in ('2nd','1st'):
        cv2.imencode('.png',cv2.resize(image,None,fx=3.178,fy=3.178))[1].tofile(str(source/f'layout_05_{floor}.png'))
    clean,_=publish_structure(tmp_path,source.parent,scale=3.178,floors=('2nd','1st'))
    activate(tmp_path,monkeypatch)
    return clean,image


def fixed(clean):
    return create_layout_retriever('pure_structure',clean,floors=('2nd','1st'),cache_root=clean.parents[4]/'cache')


def mutate(clean,change):
    source=clean/'layout_05/layout_05_1st.png'
    path=clean.parents[4]/BUNDLE_PATH
    if change=='source':source.write_bytes(b'changed source')
    elif change=='bundle':path.write_bytes(b'changed bundle')
    else:
        data=json.loads(path.read_text(encoding='utf-8'))
        record=data['payload']['recognition'][clean.relative_to(clean.parents[4]).as_posix()]['targets']['layout_05/layout_05_1st.png']
        if change=='record':record['source_sha256']='0'*64
        else:record['scales']=[3.178,1.2]
        data['payload_sha256']=digest(data['payload'])
        path.write_text(json.dumps(data),encoding='utf-8')


def test_factory_keeps_published_resources_fixed_and_rejects_unpublished_inputs(fixed_scene,tmp_path):
    clean,_=fixed_scene
    retriever=fixed(clean)
    try:
        assert retriever.fixed_resources
        assert not create_layout_retriever('pure_structure',clean,fixed_resources=False).fixed_resources
        assert module.PureStructureRetriever(clean).fixed_resources
        with pytest.raises(ValueError,match='已发布'):
            create_layout_retriever('pure_structure',tmp_path/'other/clean').prepare()
    finally:retriever.close()


def test_retired_local_scale_document_cannot_replace_a_published_bundle(fixed_scene,tmp_path):
    import hashlib
    _,image=fixed_scene
    clean=tmp_path/'unpublished/clean';path=clean/'layout_01/layout_01_1st.png'
    path.parent.mkdir(parents=True)
    path.write_bytes(cv2.imencode('.png',image)[1].tobytes())
    (clean.parent/'recognition_scales.json').write_text(json.dumps(dict(schema_version=1,targets={
        'layout_01/layout_01_1st.png':dict(scales=[1.5],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())})))
    retriever=create_layout_retriever('pure_structure',clean,floors=('1st',),cache_root=tmp_path/'unpublished_cache')
    try:
        with pytest.raises(ValueError,match='已发布'):retriever.prepare()
        assert not retriever._targets and not retriever._fingerprint
    finally:retriever.close()


@pytest.mark.parametrize('change',['source','bundle','record','scales'])
def test_verified_index_ignores_later_edits_but_fresh_check_rejects(fixed_scene,monkeypatch,change):
    clean,shot=fixed_scene;retriever=fixed(clean)
    try:
        before=retriever.recognize(shot,shot.shape[1::-1])
        assert before.candidates and not before.decision()['accepted']
        mutate(clean,change)
        with monkeypatch.context() as patch:
            def forbidden(*args,**kwargs):raise AssertionError('Hot recognition must not reread source paths')
            patch.setattr(Path,'read_bytes',forbidden);patch.setattr(Path,'read_text',forbidden);patch.setattr(Path,'glob',forbidden)
            after=retriever.recognize(shot,shot.shape[1::-1])
            assert after.candidates==before.candidates
            assert after.resource_fingerprint==before.resource_fingerprint
        assert check_scale_overrides(clean)['errors']
        retriever.close()
        with pytest.raises((ValueError,OSError)):retriever.prepare()
        assert retriever._targets==[] and not retriever._fingerprint
    finally:retriever.close()


@pytest.mark.parametrize('change',['source','bundle','record','scales','add','remove','cancel'])
def test_change_during_first_build_never_publishes_partial_index(fixed_scene,monkeypatch,change):
    clean,_=fixed_scene;source=clean/'layout_05/layout_05_1st.png';saved=source.read_bytes()
    original=module.FeatureStore.ensure;cancel=Event();first=True
    def changed(*args,**kwargs):
        nonlocal first
        result=original(*args,**kwargs)
        if first:
            first=False
            if change in ('source','bundle','record','scales'):mutate(clean,change)
            elif change=='add':
                path=clean/'layout_06/layout_06_1st.png';path.parent.mkdir();path.write_bytes(saved)
            elif change=='remove':source.unlink()
            else:cancel.set()
        return result
    monkeypatch.setattr(module.FeatureStore,'ensure',changed)
    retriever=fixed(clean)
    try:
        with pytest.raises(RecognitionCancelled if change=='cancel' else (ValueError,OSError)):
            retriever.prepare(cancelled=cancel.is_set)
        assert retriever._targets==[] and not retriever._fingerprint
    finally:retriever.close()


def test_failed_build_can_retry_after_repair_and_hot_cancel_is_honored(fixed_scene):
    clean,shot=fixed_scene;path=clean.parents[4]/BUNDLE_PATH;saved=path.read_bytes();path.write_bytes(b'broken')
    retriever=fixed(clean)
    try:
        with pytest.raises(ValueError):retriever.prepare()
        assert not retriever._fingerprint
        path.write_bytes(saved)
        assert retriever.recognize(shot,shot.shape[1::-1]).candidates
        with pytest.raises(RecognitionCancelled):retriever.prepare(cancelled=lambda:True)
        with pytest.raises(RecognitionCancelled):retriever.recognize(shot,cancelled=lambda:True)
    finally:retriever.close()


@pytest.mark.parametrize('during_build',[False,True])
def test_real_directory_link_retarget_cannot_mix_snapshots(fixed_scene,tmp_path,monkeypatch,during_build):
    clean,shot=fixed_scene;target=clean/'layout_05';alternate=clean/'alternate';shutil.copytree(target,alternate)
    stored=clean/'stored';target.rename(stored);create_directory_link(target,stored)
    retriever=fixed(clean)
    def switch():
        remove_directory_link(target,root=tmp_path);create_directory_link(target,alternate)
    try:
        if during_build:
            original=module.FeatureStore.ensure;first=True
            def changed(*args,**kwargs):
                nonlocal first
                result=original(*args,**kwargs)
                if first:first=False;switch()
                return result
            monkeypatch.setattr(module.FeatureStore,'ensure',changed)
            with pytest.raises(ValueError,match='特征准备期间变化'):retriever.prepare()
            assert not retriever._fingerprint and retriever._targets==[]
        else:
            before=retriever.recognize(shot,shot.shape[1::-1]);assert before.candidates
            switch();(alternate/'layout_05_1st.png').write_bytes(b'bad new target')
            assert retriever.recognize(shot,shot.shape[1::-1]).candidates==before.candidates
            assert check_scale_overrides(clean)['errors']
            retriever.close()
            with pytest.raises(ValueError):retriever.prepare()
    finally:
        retriever.close();remove_directory_link(target,root=tmp_path)
