"""Fixed recognition resources: publish one verified batch, then perform no disk reads."""
from pathlib import Path
from threading import Event
import shutil

import cv2
import numpy as np
import pytest

from src.layout_retrieval import create_layout_retriever
from src.recognition_control import RecognitionCancelled
from src.structure_layout import retriever as module
from src.structure_layout.scales import check_scale_overrides
from tests.support.normalized_index import make_profile
from tests.support.directory_links import create_directory_link, remove_directory_link
from tools.assets.imports import build_recognition_index as builder


@pytest.fixture
def fixed_scene(tmp_path, monkeypatch):
    clean = make_profile(tmp_path)
    monkeypatch.setattr(builder, 'project_relative', lambda p: 'assets/test/clean')
    builder.build(clean, clean.parent/'recognition')
    source = clean/'layout_05/layout_05_1st.png'
    image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    shot = cv2.resize(image, None, fx=1/3.178, fy=1/3.178)
    return clean, shot


def fixed(clean):
    return create_layout_retriever('pure_structure', clean, floors=('2nd','1st'), workers=2)


def test_factory_limits_fixed_policy_to_normalized_profiles(fixed_scene, tmp_path):
    clean, _ = fixed_scene
    assert fixed(clean).fixed_resources
    assert not create_layout_retriever('pure_structure', tmp_path/'other/clean').fixed_resources
    assert not create_layout_retriever('pure_structure', clean, fixed_resources=False).fixed_resources
    assert not module.PureStructureRetriever(clean).fixed_resources  # maintenance default


@pytest.mark.parametrize('change', ['source','index','image','scales'])
def test_verified_index_ignores_later_edits_but_fresh_check_rejects(fixed_scene, monkeypatch, change):
    clean, shot = fixed_scene
    paths = {'source':clean/'layout_05/layout_05_1st.png',
             'index':clean.parent/'recognition/index.json',
             'image':clean.parent/'recognition/layout_05/layout_05_1st.png',
             'scales':clean.parent/'recognition_scales.json'}
    retriever = fixed(clean)
    try:
        before = retriever.recognize(shot, shot.shape[1::-1])
        assert before.candidates and not before.decision()['accepted']
        paths[change].write_bytes(b'changed after complete publication')
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('hot recognition must not read or enumerate disk resources')
            patch.setattr(Path, 'read_bytes', forbidden)
            patch.setattr(Path, 'read_text', forbidden)
            patch.setattr(Path, 'glob', forbidden)
            after = retriever.recognize(shot, shot.shape[1::-1])
            assert after.candidates == before.candidates
            assert after.resource_fingerprint == before.resource_fingerprint
        assert check_scale_overrides(clean)['errors']
        retriever.close()
        with pytest.raises((ValueError, OSError)):
            retriever.prepare()
        assert retriever._targets == [] and not retriever._fingerprint
    finally:
        retriever.close()


@pytest.mark.parametrize('change', ['source','index','image','scales','add','remove','cancel'])
def test_change_during_first_build_never_publishes_partial_index(fixed_scene, monkeypatch, change):
    clean, _ = fixed_scene
    source = clean/'layout_05/layout_05_1st.png'
    source_bytes = source.read_bytes()
    old = module.prepare_response_cache
    cancel = Event()
    def changed(targets, **kwargs):
        result = old(targets, **kwargs)
        paths = {'source':source, 'index':clean.parent/'recognition/index.json',
                 'image':clean.parent/'recognition/layout_05/layout_05_1st.png',
                 'scales':clean.parent/'recognition_scales.json'}
        if change in paths:
            paths[change].write_bytes(b'changed while building')
        elif change == 'add':
            path=clean/'layout_06/layout_06_1st.png';path.parent.mkdir();path.write_bytes(source_bytes)
        elif change == 'remove':
            source.unlink()
        else:
            cancel.set()
        return result
    monkeypatch.setattr(module, 'prepare_response_cache', changed)
    retriever=fixed(clean)
    try:
        with pytest.raises(RecognitionCancelled if change=='cancel' else (ValueError,OSError)):
            retriever.prepare(cancelled=cancel.is_set)
        assert retriever._targets == [] and not retriever._fingerprint
    finally:
        retriever.close()


def test_failed_build_can_retry_after_repair_and_hot_cancel_is_honored(fixed_scene):
    clean, shot=fixed_scene
    index=clean.parent/'recognition/index.json';saved=index.read_bytes();index.write_bytes(b'broken')
    retriever=fixed(clean)
    try:
        with pytest.raises(ValueError):retriever.prepare()
        assert not retriever._fingerprint
        index.write_bytes(saved)
        assert retriever.recognize(shot,shot.shape[1::-1]).candidates
        with pytest.raises(RecognitionCancelled):retriever.prepare(cancelled=lambda:True)
        with pytest.raises(RecognitionCancelled):retriever.recognize(shot,cancelled=lambda:True)
    finally:retriever.close()


@pytest.mark.parametrize('during_build', [False,True])
def test_real_directory_link_retarget_cannot_mix_snapshots(fixed_scene, tmp_path, monkeypatch, during_build):
    clean,shot=fixed_scene
    target=clean/'layout_05';alternate=tmp_path/'alternate';shutil.copytree(target,alternate)
    # Keep the same bytes: a directory-link change during the batch is still detected.
    stored=tmp_path/'stored';target.rename(stored);create_directory_link(target,stored)
    retriever=fixed(clean)
    def switch():
        remove_directory_link(target,root=tmp_path)
        create_directory_link(target,alternate)
    try:
        if during_build:
            old=module.prepare_response_cache
            def changed(*args,**kwargs):
                result=old(*args,**kwargs);switch();return result
            monkeypatch.setattr(module,'prepare_response_cache',changed)
            with pytest.raises(ValueError,match='目录链接已改变'):retriever.prepare()
            assert not retriever._fingerprint and retriever._targets==[]
        else:
            before=retriever.recognize(shot,shot.shape[1::-1]);assert before.candidates
            switch();(alternate/'layout_05_1st.png').write_bytes(b'bad new target')
            after=retriever.recognize(shot,shot.shape[1::-1])
            assert after.candidates==before.candidates
            assert check_scale_overrides(clean)['errors']
            retriever.close()
            with pytest.raises(ValueError):retriever.prepare()
    finally:
        retriever.close();remove_directory_link(target,root=tmp_path)
