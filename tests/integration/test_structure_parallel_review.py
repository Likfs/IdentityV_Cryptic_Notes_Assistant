"""Local candidate checks use the bounded pool and preserve public results."""
from threading import Barrier,Event,Lock
import cv2
import numpy as np
import pytest
from src.structure_layout import retriever as module
from src.recognition_control import RecognitionCancelled


def test_local_reviews_run_concurrently_without_changing_candidate_order(tmp_path,monkeypatch):
    image=np.full((180,200,3),25,np.uint8)
    cv2.rectangle(image,(20,20),(170,155),(100,100,100),-1)
    cv2.rectangle(image,(20,20),(170,155),(175,175,175),2)
    cv2.line(image,(90,20),(90,110),(175,175,175),2)
    for layout in ('layout_01','layout_02'):
        p=tmp_path/layout/f'{layout}_1st.png';p.parent.mkdir()
        cv2.imencode('.png',cv2.resize(image,None,fx=1.49,fy=1.49))[1].tofile(p)
    serial=module.PureStructureRetriever(tmp_path,workers=1,floors=('1st',))
    parallel=module.PureStructureRetriever(tmp_path,workers=2,floors=('1st',))
    try:
        expected=serial.recognize(image,(200,180)).diagnostics()['candidates']
        original=module.review;barrier=Barrier(2);guard=Lock();entered=[]
        def checked(*args):
            with guard:
                entered.append(args[1]['layout']);wait=len(entered)<=2
            if wait:barrier.wait(timeout=3)
            return original(*args)
        monkeypatch.setattr(module,'review',checked)
        result=parallel.recognize(image,(200,180))
        assert len(entered)>=2
        assert result.diagnostics()['candidates']==expected
        assert not result.decision()['accepted']
    finally:
        serial.close();parallel.close()


def test_index_prepares_two_sources_concurrently_and_publishes_in_path_order(tmp_path,monkeypatch):
    image=np.full((120,140,3),100,np.uint8)
    cv2.rectangle(image,(20,20),(110,95),(175,175,175),2)
    for i in range(4):
        layout=f'layout_{i+1:02}';p=tmp_path/layout/f'{layout}_1st.png';p.parent.mkdir()
        cv2.imencode('.png',image)[1].tofile(p)
    original=module.prepare_target;barrier=Barrier(2);guard=Lock()
    state={'active':0,'peak':0,'calls':0}
    def checked(*args):
        with guard:
            state['active']+=1;state['calls']+=1
            state['peak']=max(state['peak'],state['active']);wait=state['calls']<=2
        try:
            if wait:barrier.wait(timeout=3)
            return original(*args)
        finally:
            with guard:state['active']-=1
    monkeypatch.setattr(module,'prepare_target',checked)
    retriever=module.PureStructureRetriever(tmp_path,workers=8,floors=('1st',))
    try:
        first=retriever.prepare()
        assert state['peak']==2
        assert [t['layout'] for t in retriever._targets]==[f'layout_{i+1:02}' for i in range(4)]
        assert retriever.prepare()==first and state['calls']==4
    finally:retriever.close()


@pytest.mark.parametrize('failure',['cancel','decode'])
def test_interrupted_parallel_rebuild_preserves_complete_previous_index(tmp_path,monkeypatch,failure):
    image=np.full((120,140,3),100,np.uint8)
    cv2.rectangle(image,(20,20),(110,95),(175,175,175),2)
    path=tmp_path/'layout_01/layout_01_1st.png';path.parent.mkdir()
    cv2.imencode('.png',image)[1].tofile(path)
    retriever=module.PureStructureRetriever(tmp_path,workers=8,floors=('1st',))
    cancel=Event()
    try:
        fingerprint=retriever.prepare();targets=retriever._targets
        added=tmp_path/'layout_02/layout_02_1st.png';added.parent.mkdir()
        added.write_bytes(path.read_bytes() if failure=='cancel' else b'invalid image')
        if failure=='cancel':
            original=module.prepare_target
            def cancelled_build(*args):
                target=original(*args);cancel.set();return target
            monkeypatch.setattr(module,'prepare_target',cancelled_build)
        with pytest.raises(RecognitionCancelled if failure=='cancel' else ValueError):
            retriever.prepare(cancelled=cancel.is_set)
        assert retriever._fingerprint==fingerprint and retriever._targets is targets
        assert [t['layout'] for t in retriever._targets]==['layout_01']
    finally:retriever.close()
