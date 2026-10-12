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
