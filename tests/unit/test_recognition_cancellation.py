from threading import Event, RLock, Thread
from concurrent.futures import ThreadPoolExecutor
import importlib
import numpy as np
import pytest
from src.recognition_control import RecognitionCancelled, recognition_lock
from src.layout_retrieval import LayoutRetriever
from src.structure_layout.retriever import PureStructureRetriever


@pytest.mark.parametrize('factory',[LayoutRetriever,PureStructureRetriever])
def test_cancelled_recognition_never_reads_assets(factory,tmp_path):
    retriever=factory(tmp_path)
    with pytest.raises(RecognitionCancelled):
        retriever.recognize(np.zeros((80,90,3),np.uint8),cancelled=lambda:True)
    assert retriever._targets==[]


def test_cancelled_waiter_leaves_lock_queue_before_owner_finishes():
    lock=RLock();entered=Event();cancel=Event();done=Event()
    def wait():
        entered.set()
        try:
            with recognition_lock(lock,cancel.is_set):pytest.fail('obsolete waiter entered lock')
        except RecognitionCancelled:done.set()
    with lock:
        thread=Thread(target=wait);thread.start()
        assert entered.wait(2)
        cancel.set()
        assert done.wait(2)
    thread.join(2)


def test_parallel_search_cancels_obsolete_targets_without_changing_completed_order(monkeypatch):
    search=importlib.import_module('src.structure_layout.search')
    image=np.full((80,90,3),100,np.uint8)
    edges=np.zeros((2,80,90),np.uint8);edges[:,20:60,25]=1;edges[:,20:60,60]=1
    monkeypatch.setattr(search,'wall_edges',lambda *a:edges)
    monkeypatch.setattr(search,'visible_interior',lambda *a:np.zeros((80,90),np.uint8))
    calls=[];cancel=Event()
    def target(item,**kwargs):
        search.check_cancelled(kwargs['cancelled'])
        calls.append(item['id']);cancel.set()
        search.check_cancelled(kwargs['cancelled'])
    monkeypatch.setattr(search,'search_target',target)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RecognitionCancelled):
            search.search(image,[{'id':i, 'scales':search.MINIMUM_SCALES} for i in range(40)],
                          executor,cancelled=cancel.is_set)
    assert calls==[0]


def test_icon_fallback_stops_at_next_scale(monkeypatch):
    from src import layout_retrieval as retrieval
    with pytest.raises(RecognitionCancelled):
        retrieval.fallback_fit([(1,None,None)],None,None,None,None,None,cancelled=lambda:True)
