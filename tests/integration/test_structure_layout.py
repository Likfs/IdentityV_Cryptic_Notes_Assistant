"""Minimum-zoom recognition: geometry, index lifecycle and visible conflicts."""
import sys
import cv2
import numpy as np
from src.layout_retrieval import create_layout_retriever
from src.structure_layout.retriever import PureStructureRetriever
from src.structure_layout.search import MINIMUM_SCALES
from src.structure_layout.local import evidence, review


def scene():
    image = np.full((240, 280, 3), 25, np.uint8)
    cv2.rectangle(image, (45, 45), (235, 180), (100, 100, 100), -1)
    cv2.rectangle(image, (45, 45), (235, 180), (175, 175, 175), 2)
    cv2.line(image, (105, 45), (105, 125), (175, 175, 175), 2)
    return image


def test_minimum_recognition_preserves_pose_manual_confirmation_and_index(tmp_path):
    image = scene()
    path = tmp_path/'layout_29/layout_29_b1.png'
    path.parent.mkdir()
    scaled = cv2.resize(image, None, fx=1.49, fy=1.49)
    cv2.imencode('.png', scaled)[1].tofile(str(path))
    retriever = create_layout_retriever('pure_structure', tmp_path)
    try:
        first = retriever.prepare()
        result = retriever.recognize(image, (280, 240))
        assert result.candidates and result.candidates[0].layout == 'layout_29'
        assert result.candidates[0].details['floor'] == 'b1'
        assert result.parameters['coarse_scales'] == list(MINIMUM_SCALES)
        assert result.parameters['execution_backend'] == 'cpu'
        assert not result.parameters['pose_refinement']
        assert all(c.details['evaluations'] == 1 and c.details['scale'] in MINIMUM_SCALES for c in result.candidates)
        assert not result.decision()['accepted'] and result.decision()['layout'] is None
        retriever.close()
        assert retriever.recognize(image, (280, 240)).candidates == result.candidates
        scaled[0, 0] = 42
        cv2.imencode('.png', scaled)[1].tofile(str(path))
        assert retriever.prepare() != first
        other = tmp_path/'layout_30/layout_30_1st.png'
        other.parent.mkdir();other.write_bytes(path.read_bytes())
        assert {c.layout for c in retriever.recognize(image, (280, 240)).candidates} == {'layout_29','layout_30'}
    finally:
        retriever.close()


def test_extra_wall_only_penalizes_visible_region():
    image = np.full((80, 100, 3), 100, np.uint8)
    query = evidence(image, np.zeros((2, 80, 100), np.uint8))
    points = np.array([[50, y] for y in range(15, 65)], np.float32)
    target = dict(rawdist=[np.zeros((80,100),np.uint8)]*2,
                  reliable_points=[points,np.empty((0,2),np.float32)])
    candidate = dict(scale=1.,x=0.,y=0.,score=.8)
    visible = review(target,candidate,query,[0,0,100,80])
    query['known'][:] = 0
    fog = review(target,candidate,query,[0,0,100,80])
    assert visible['extra_local_penalty'] > 0
    assert fog['extra_local_penalty'] == 0


def test_two_floor_index_never_loads_a_basement(tmp_path):
    directory = tmp_path/'layout_01'
    directory.mkdir()
    for floor in ('2nd', '1st'):
        cv2.imencode('.png', scene())[1].tofile(directory/f'layout_01_{floor}.png')
    (directory/'layout_01_b1.png').write_bytes(b'not a supported floor for this profile')
    retriever = create_layout_retriever('pure_structure', tmp_path, floors=('2nd', '1st'))
    try:
        retriever.prepare()
        assert {t['floor'] for t in retriever._targets} == {'2nd', '1st'}
    finally:
        retriever.close()


def test_structure_import_has_no_optional_accelerator_or_gui(tmp_path):
    import subprocess
    from src.paths import PROJECT_ROOT
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  assert not fullname.startswith(('torch','PyQt6','retired_features','tools','src.registration.multiscale_structure')),fullname
sys.meta_path.insert(0,Block())
from src.structure_layout.retriever import PureStructureRetriever,PARAMETERS
assert not PARAMETERS['pose_refinement'] and PARAMETERS['execution_backend']=='cpu'
'''
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=PROJECT_ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_per_source_scales_are_isolated_and_invalidate_index(tmp_path):
    import hashlib
    import json
    import pytest
    root = tmp_path/'clean'
    image = scene()
    for layout, scale in [('layout_05', 3.18), ('layout_06', 1.49)]:
        path = root/layout/f'{layout}_1st.png'
        path.parent.mkdir(parents=True)
        cv2.imencode('.png', cv2.resize(image, None, fx=scale, fy=scale))[1].tofile(path)
    path = root/'layout_05/layout_05_1st.png'
    record = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scales=[3.18])
    config = tmp_path/'recognition_scales.json'
    def save():
        config.write_text(json.dumps(dict(schema_version=1, targets={'layout_05/layout_05_1st.png':record})))
    save()
    retriever = PureStructureRetriever(root, floors=('1st',))
    try:
        result = retriever.recognize(image, (280, 240))
        by_layout = {c.layout:c for c in result.candidates}
        assert by_layout['layout_05'].details['scale'] == 3.18
        assert by_layout['layout_06'].details['scale'] in MINIMUM_SCALES
        first = result.resource_fingerprint
        record['scales'] = [3.17]
        save()
        assert retriever.prepare() != first
        path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError, match='底图已改变'):
            retriever.prepare()
    finally:
        retriever.close()


def test_scale_change_during_search_cannot_publish_old_candidates(tmp_path, monkeypatch):
    import hashlib
    import json
    import src.structure_layout.retriever as module
    root = tmp_path/'clean'
    path = root/'layout_05/layout_05_1st.png'
    path.parent.mkdir(parents=True)
    cv2.imencode('.png', scene())[1].tofile(path)
    config = tmp_path/'recognition_scales.json'
    record = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scales=[1.])
    def save():
        config.write_text(json.dumps(dict(schema_version=1, targets={'layout_05/layout_05_1st.png':record})))
    save()
    def changed(*args, **kwargs):
        record['scales'] = [1.1]
        save()
        return {'floor_candidates': []}
    monkeypatch.setattr(module, 'search', changed)
    retriever = PureStructureRetriever(root, floors=('1st',))
    try:
        result = retriever.recognize(scene(), (280,240))
        assert not result.candidates and not result.accepted
        assert '尺度配置已改变' in result.reasons[0]
    finally:
        retriever.close()
