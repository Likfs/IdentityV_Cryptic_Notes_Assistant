"""Incremental, bounded recognition features against real published floor inputs."""
import json
import os
from pathlib import Path
import shutil
from threading import Event

import cv2
import numpy as np
import pytest

from src.layout_retrieval import create_layout_retriever
from src.recognition_control import RecognitionCancelled
from src.structure_layout import retriever as module
from src.structure_layout import feature_store
from tests.support.aligned_runtime import activate, difficulty, publish, write_png


@pytest.fixture
def scene(tmp_path, monkeypatch):
    definition = difficulty('hard', '困难', ('1st',), {'clean':'底图', 'custom':'路线'})
    catalog = dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女', difficulties=[definition])])
    clean = tmp_path/definition['clean_root']
    route = tmp_path/definition['asset_versions'][1]['source_root']
    for index in (1, 2):
        image = np.full((240,280,3), 100, np.uint8)
        cv2.rectangle(image,(20,25),(240,210),(175,175,175),3)
        cv2.line(image,(40+index*30,40),(40+index*30,185),(175,175,175),3)
        for root in (clean, route):
            write_png(root/f'layout_{index:02}/layout_{index:02}_1st.png', image)
    publish(tmp_path, catalog)
    activate(tmp_path, monkeypatch)
    def factory():
        return create_layout_retriever('pure_structure', clean, floors=('1st',), cache_root=tmp_path/'cache')
    shot = cv2.resize(image, None, fx=1/1.5, fy=1/1.5, interpolation=cv2.INTER_AREA)
    return tmp_path, clean, route, catalog, factory, shot


def test_disk_candidates_equal_original_memory_search_and_only_one_floor_maps(scene, monkeypatch):
    root, clean, _, _, factory, shot = scene
    from src.structure_layout.scales import read_scale_overrides
    from src.structure_layout.search import search, merge
    overrides = read_scale_overrides(clean)
    targets = []
    for path in sorted(clean.glob('layout_*/*.png')):
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        target = module.prepare_target(image,path.parent.name,'1st')
        target['scales'] = overrides[path.relative_to(clean).as_posix()]['scales']
        targets.append(target)
    raw = search(shot,targets)
    query = module.evidence(shot,raw['query_edges'])
    lookup = {(t['layout'],t['floor']):t for t in targets}
    expected = merge([module.review_candidate(fit,lookup,query,raw['box'],None)
                      for fit in raw['floor_candidates']])
    disk = factory()
    active = []
    original = feature_store.FeatureFile.mapped
    from contextlib import contextmanager
    @contextmanager
    def mapped(self):
        active.append(self)
        assert len(active) == 1
        try:
            with original(self) as target:
                yield target
        finally:
            active.pop()
    monkeypatch.setattr(feature_store.FeatureFile, 'mapped', mapped)
    try:
        actual = disk.recognize(shot, shot.shape[1::-1])
        assert expected and [item.layout for item in actual.candidates] == [fit['layout'] for fit in expected]
        for item,fit in zip(actual.candidates,expected):
            assert item.score == fit['score']
            assert all(item.details[key] == value for key,value in fit.items())
        assert actual.parameters['feature_storage'] == 'disk_mapped_floor_v1'
        assert all(not isinstance(v, np.ndarray) for t in disk._targets for v in t.values())
        assert disk.feature_cache_stats == dict(total=2, built=2, reused=0)
        assert not active
    finally:
        disk.close()


def test_layout_scale_override_rebuilds_only_affected_layout_features(scene):
    root, _, _, catalog, factory, _ = scene
    first = factory()
    try:
        first.prepare()
        before = {target['layout']: target['cache_key'] for target in first._targets}
    finally:
        first.close()
    scale_path = root/'configs/scale_config.json'
    scales = json.loads(scale_path.read_text())
    scales['maps']['a_bringer_of_doom']['hard']['layout_minimum_scales'] = {'layout_02': 2.0}
    scale_path.write_text(json.dumps(scales), encoding='utf-8')
    publish(root, catalog)
    second = factory()
    try:
        second.prepare()
        after = {target['layout']: target['cache_key'] for target in second._targets}
        assert second.feature_cache_stats == dict(total=2, built=1, reused=1)
        assert before['layout_01'] == after['layout_01']
        assert before['layout_02'] != after['layout_02']
        assert {target['layout']: target['scales'] for target in second._targets} == {
            'layout_01': [1.5], 'layout_02': [2.0]}
    finally:
        second.close()


def test_restart_reuses_all_features_and_hot_recognition_ignores_source_edits(scene, monkeypatch):
    _, clean, _, _, factory, shot = scene
    first = factory()
    try:
        before = first.recognize(shot, shot.shape[1::-1])
    finally:
        first.close()
    second = factory()
    try:
        second.prepare()
        assert second.feature_cache_stats == dict(total=2, built=0, reused=2)
        source = clean/'layout_01/layout_01_1st.png'
        saved = source.read_bytes()
        source.write_bytes(b'changed after the session was verified')
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                pytest.fail('Hot recognition reread/enumerated source or cache paths')
            patch.setattr(Path, 'read_bytes', forbidden)
            patch.setattr(Path, 'read_text', forbidden)
            patch.setattr(Path, 'glob', forbidden)
            assert second.recognize(shot, shot.shape[1::-1]).candidates == before.candidates
        second.close()
        with pytest.raises(ValueError, match='发布'):
            second.prepare()
        source.write_bytes(saved)
    finally:
        second.close()


def test_updates_only_changed_or_added_floors_and_prunes_removed_layout(scene):
    root, clean, route, catalog, factory, _ = scene
    first = factory();first.prepare();first.close()
    # Route-only publication must not invalidate the clean-map feature package.
    route_image = np.full((240,280,3), 125, np.uint8)
    write_png(route/'layout_01/layout_01_1st.png', route_image)
    publish(root, catalog)
    second = factory();second.prepare()
    assert second.feature_cache_stats == dict(total=2, built=0, reused=2)
    second.close()
    source = clean/'layout_01/layout_01_1st.png'
    image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    image[0,0] = 101
    write_png(source, image)
    unexported = factory()
    with pytest.raises(ValueError, match='发布'):unexported.prepare()
    assert not unexported._targets and not unexported._fingerprint
    publish(root, catalog)
    changed = factory();changed.prepare()
    assert changed.feature_cache_stats == dict(total=2, built=1, reused=1)
    changed.close()
    for folder in (clean, route):
        write_png(folder/'layout_03/layout_03_1st.png', image)
    publish(root, catalog)
    added = factory();added.prepare()
    assert added.feature_cache_stats == dict(total=3, built=1, reused=2)
    added.close()
    for folder in (clean, route):
        (folder/'layout_02/layout_02_1st.png').unlink()
        (folder/'layout_02').rmdir()
    publish(root, catalog)
    unrelated = root/'cache/keep.txt';unrelated.write_text('user data')
    removed = factory()
    try:
        removed.prepare()
        assert removed.feature_cache_stats == dict(total=2, built=0, reused=2)
        assert {t['layout'] for t in removed._targets} == {'layout_01','layout_03'}
        assert len(list((root/'cache').glob('*.feat'))) == 2
        assert unrelated.read_text() == 'user data'
    finally:
        removed.close()


def test_corrupt_cache_rebuilds_only_that_floor_and_algorithm_change_invalidates(scene, monkeypatch):
    root, _, _, _, factory, _ = scene
    first=factory();first.prepare();first.close()
    next((root/'cache').glob('*.feat')).write_bytes(b'broken feature')
    repaired=factory();repaired.prepare()
    assert repaired.feature_cache_stats == dict(total=2,built=1,reused=1)
    repaired.close()
    monkeypatch.setitem(module.PARAMETERS,'algorithm_version','test-new-feature-version')
    updated=factory()
    try:
        updated.prepare()
        assert updated.feature_cache_stats == dict(total=2,built=2,reused=0)
    finally:
        updated.close()


@pytest.mark.parametrize('action', ['cancel','source','add'])
def test_cancel_or_source_change_never_publishes_partial_index(scene, action):
    _, clean, _, _, factory, _ = scene
    cancel=Event();retriever=factory()
    def progress(done,total):
        if done != 1:return
        if action=='cancel':cancel.set()
        elif action=='source':(clean/'layout_01/layout_01_1st.png').write_bytes(b'changed during build')
        else:write_png(clean/'layout_03/layout_03_1st.png',np.full((240,280,3),100,np.uint8))
    with pytest.raises(RecognitionCancelled if action=='cancel' else ValueError):
        retriever.prepare(cancelled=cancel.is_set,progress=progress)
    assert retriever._targets==[] and not retriever._fingerprint and retriever._feature_store is None
    assert not list(retriever.cache_root.glob('*.tmp'))
    if action=='cancel':
        retriever.prepare()
        assert retriever.feature_cache_stats == dict(total=2,built=1,reused=1)
    retriever.close()


def test_disk_write_failure_is_reported_without_memory_fallback(scene, monkeypatch):
    _, _, _, _, factory, shot = scene
    def full_disk(*args, **kwargs):raise OSError('feature disk full')
    monkeypatch.setattr(feature_store.tempfile,'NamedTemporaryFile',full_disk)
    retriever=factory()
    try:
        result=retriever.recognize(shot,shot.shape[1::-1])
        assert result.reasons == ['feature disk full'] and not result.candidates
        assert retriever._targets==[] and not retriever._fingerprint
    finally:
        retriever.close()


def test_cancelled_search_releases_mapping_and_keeps_verified_session(scene, monkeypatch):
    _, _, _, _, factory, shot = scene
    retriever=factory();cancel=Event()
    try:
        before=retriever.recognize(shot,shot.shape[1::-1])
        old=module.review_candidate
        def cancelled_review(*args, **kwargs):
            cancel.set()
            return old(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(module,'review_candidate',cancelled_review)
            with pytest.raises(RecognitionCancelled):
                retriever.recognize(shot,shot.shape[1::-1],cancelled=cancel.is_set)
        assert retriever.recognize(shot,shot.shape[1::-1]).candidates==before.candidates
    finally:
        retriever.close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows cache pinning contract')
def test_verified_feature_files_cannot_be_rewritten_or_deleted_during_session(scene):
    _, _, _, _, factory, _ = scene
    retriever=factory()
    try:
        retriever.prepare()
        path=next(iter(retriever._feature_store.files.values())).path
        with pytest.raises(PermissionError):path.write_bytes(b'changed')
        with pytest.raises(PermissionError):path.unlink()
    finally:
        retriever.close()
    path.unlink()
