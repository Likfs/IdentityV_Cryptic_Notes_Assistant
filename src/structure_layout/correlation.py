"""Request-scoped, worker-local buffers for unchanged float32 correlation."""
import threading
import cv2
import numpy as np

class CorrelationWorkspace:
    """Scratch results are valid until the next call on the same worker.

    A search owns one instance. Worker-local arrays are never part of the
    published target index or returned poses, and are released with the search.
    """
    def __init__(self):
        self._local=threading.local()

    def correlate(self,maps,low,channels,negative,scale,total):
        height=low.shape[0]-negative.shape[0]+1
        width=low.shape[1]-negative.shape[1]+1
        count=height*width
        arrays=getattr(self._local,'arrays',None)
        if arrays is None or arrays[0].size<count:
            arrays=[np.empty(count,np.float32) for _ in range(2)]
            self._local.arrays=arrays
        support,extra=[a[:count].reshape(height,width) for a in arrays[:2]]
        cv2.matchTemplate(maps[0],channels[0],cv2.TM_CCORR,result=support)
        # Preserve sum's initial zero and each float32 rounding step.
        np.add(0,support,out=support)
        cv2.matchTemplate(maps[1],channels[1],cv2.TM_CCORR,result=extra)
        np.add(support,extra,out=support)
        np.divide(support,total,out=support)
        cv2.matchTemplate(low,negative,cv2.TM_CCORR,result=extra)
        np.maximum(0,extra,out=extra)
        np.multiply(extra,scale,out=extra)
        np.divide(extra,total,out=extra)
        return support,extra

