"""Scratch reuse must preserve numerical results and concurrent searches."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import gc
import weakref
import cv2
import numpy as np
import pytest
from src.structure_layout.correlation import CorrelationWorkspace

def inputs(shape,query_shape,seed):
    rng=np.random.default_rng(seed)
    maps=[rng.random(shape,dtype=np.float32) for _ in range(2)]
    low=rng.random(shape,dtype=np.float32)
    channels=[rng.random(query_shape,dtype=np.float32) for _ in range(2)]
    negative=rng.random(query_shape,dtype=np.float32)
    for a in [*maps,low,*channels,negative]:a.setflags(write=False)
    return maps,low,channels,negative

def expected(data,scale):
    maps,low,channels,negative=data;total=sum(ch.sum() for ch in channels)
    support=sum(cv2.matchTemplate(m,ch,cv2.TM_CCORR) for m,ch in zip(maps,channels))/total
    extra=np.maximum(0,cv2.matchTemplate(low,negative,cv2.TM_CCORR))*scale/total
    return support,extra

@pytest.mark.parametrize('scale',[.6176283703635228,1.5])
def test_reused_buffers_preserve_all_pixels_and_peaks_across_sizes(scale):
    workspace=CorrelationWorkspace();initial=None
    for shape,query in [((121,145),(25,31)),((117,139),(31,39)),((183,197),(19,21))]:
        data=inputs(shape,query,12);wanted=expected(data,scale)
        support,extra=workspace.correlate(*data,scale,sum(ch.sum() for ch in data[2]))
        for a,b in zip((support,extra),wanted):
            np.testing.assert_array_equal(a,b)
            assert cv2.minMaxLoc(a)==cv2.minMaxLoc(b)
        if initial is None:initial=tuple(a.base for a in (support,extra))
        elif shape==(117,139):assert all(a.base is b for a,b in zip((support,extra),initial))
        assert not any(np.shares_memory(a,b) for a in (support,extra) for b in [*data[0],data[1],*data[2],data[3]])

def test_workers_never_share_mutable_correlation_results():
    workspace=CorrelationWorkspace();barrier=Barrier(2)
    def work(seed):
        data=inputs((151,163),(29,35),seed);wanted=expected(data,1.5)
        support,extra=workspace.correlate(*data,1.5,sum(ch.sum() for ch in data[2]))
        barrier.wait(timeout=5)
        for a,b in zip((support,extra),wanted):np.testing.assert_array_equal(a,b)
        return support,extra
    with ThreadPoolExecutor(max_workers=2) as executor:
        a=executor.submit(work,1);b=executor.submit(work,2);left,right=a.result(),b.result()
    assert not any(np.shares_memory(a,b) for a in left for b in right)

def test_request_workspace_releases_arrays_with_live_worker_pool():
    workspace=CorrelationWorkspace();barrier=Barrier(2)
    def work():
        data=inputs((101,113),(17,19),10)
        support,extra=workspace.correlate(*data,1.5,sum(ch.sum() for ch in data[2]))
        refs=[weakref.ref(a.base) for a in (support,extra)]
        barrier.wait(timeout=5)
        return refs
    with ThreadPoolExecutor(max_workers=2) as executor:
        a=executor.submit(work);b=executor.submit(work);refs=a.result()+b.result()
        assert all(ref() is not None for ref in refs)
        del workspace;gc.collect()
        assert all(ref() is None for ref in refs)
