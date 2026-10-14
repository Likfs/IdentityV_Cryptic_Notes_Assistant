"""Fixed minimum-zoom CPU search and full-resolution pose scoring.

Coordinates map the normalized screenshot ROI to each original clean map.
No labels, previous candidates or historical poses enter the search.
"""
import time
from functools import partial
import cv2
import numpy as np
from src.layout_retrieval import wall_edges
from src.recognition_control import check_cancelled

R=.3
SIGMA=3.0
PENALTY=.5
MINIMUM_SCALES=(1.46666667,1.51666667)

def visible_interior(im, edges=None):
    """Conservative observed floor interiors, never template-dependent."""
    h,s,v=cv2.split(cv2.cvtColor(im,cv2.COLOR_BGR2HSV))
    room=(h>=5)&(h<=32)&(s>=22)&(s<110)
    hall=((h>=90)&(h<=125)&(s<100))|(s<25)
    mask=((room|hall)&(v>75)&(v<150)).astype(np.uint8)
    mask=cv2.erode(mask,np.ones((9,9),np.uint8))
    if edges is None: edges=wall_edges(im,True)
    mask[cv2.dilate((edges.sum(0)>0).astype(np.uint8),np.ones((15,15),np.uint8))>0]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    keep=np.zeros(n,np.uint8)
    keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=100)
    result=keep[labels]
    yy,xx=np.where(edges.sum(0)>0)
    if not len(xx):return np.zeros_like(result)
    crop=np.zeros_like(result);crop[yy.min():yy.max()+1,xx.min():xx.max()+1]=1
    return result*crop

def sample(field, xy, outside=0):
    if not len(xy):return np.empty(0,np.float32)
    width=min(512,len(xy));height=(len(xy)+width-1)//width
    padded=np.zeros((height*width,2),np.float32);padded[:len(xy)]=xy
    return cv2.remap(field,padded[:,0].reshape(height,width),padded[:,1].reshape(height,width),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=outside).ravel()[:len(xy)]


class _PoseSampler:
    """Seed-local reusable remap storage; no buffers shared between workers."""
    def __init__(self, count):
        width=min(512,max(1,count));height=max(1,(count+width-1)//width)
        self.count=count
        self.maps=np.zeros((height,width,2),np.float32)
        self.coordinates=self.maps.reshape(-1,2)[:count]
        self.values=np.empty((height,width),np.float32)

    def sample(self, field, xy, outside=0):
        if not self.count:return self.values.ravel()[:0]
        self.coordinates[:]=xy
        cv2.remap(field,self.maps,None,cv2.INTER_LINEAR,dst=self.values,
                  borderMode=cv2.BORDER_CONSTANT,borderValue=outside)
        return self.values.ravel()[:self.count]


def score_pose(t,seed,qpoints,negative,box,query_geometry=None):
    # Query geometry is read-only and shared by all target searches.
    if query_geometry is None:
        origin=np.array(box[:2],np.float32)
        query_geometry=(origin,[p-origin for p in qpoints],sum(len(p) for p in qpoints))
    origin,rel,qcount=query_geometry
    s0=seed['scale']
    # Preserve the validated target-point crop and numeric operation order.
    # Clipping never depends on a candidate visibility mask.
    lo=np.array([seed['x'],seed['y']])-50
    hi=np.array([seed['x'],seed['y']])+(np.array(box[2:])-origin)*(s0+.055)+50
    points=t.get('all_points')
    if points is None:points=np.concatenate(t['points'])
    pts=points[((points>=lo)&(points<=hi)).all(1)]
    # Follow NumPy's scalar promotion (float64 on NumPy 2) and preserve the
    # float32 subtraction. Reuse storage without fusing or reordering operations.
    scalar=np.float64(s0)
    query_buffers=[(np.empty(p.shape,np.result_type(p,scalar)),_PoseSampler(len(p))) for p in rel]
    target_delta=np.empty_like(pts)
    target_xy=np.empty(pts.shape,np.result_type(pts,scalar))
    target_sampler=_PoseSampler(len(pts))
    offset=np.empty(2,np.float32)
    # Seed-local: a different seed can have a different target-point crop.
    def components(z):
        s,x,y=z;offset[:]=x,y
        support=0
        for d,p,(xy,sampler) in zip(t['full'],rel,query_buffers):
            np.multiply(p,s,out=xy)
            np.add(xy,offset,out=xy)
            support+=np.exp(-.5*(sampler.sample(d,xy,10000)/(s*SIGMA))**2).sum()
        support/=qcount
        if len(pts):
            np.subtract(pts,offset,out=target_delta)
            np.divide(target_delta,s,out=target_xy)
            np.add(target_xy,origin,out=target_xy)
            extra=float(target_sampler.sample(negative,target_xy).sum()/s/qcount)
        else:extra=0.
        value=(float(support),extra)
        return value
    z=np.array([s0,seed['x'],seed['y']])
    support,extra=components(z)
    return dict(layout=t['layout'],floor=t['floor'],score=support-PENALTY*extra,wall_support=support,extra_wall_ratio=extra,penalty=PENALTY*extra,scale=float(z[0]),x=float(z[1]),y=float(z[2]),evaluations=1)

def merge(candidates):
    best={}
    for c in candidates:
        if c['layout'] not in best or c['score']>best[c['layout']]['score']:best[c['layout']]=c
    return sorted(best.values(),key=lambda c:(-c['score'],c['layout']))

def search_target(t, *, templates, qpoints, negative, box, query_geometry, cancelled=None):
    """One independent floor; retain scale and seed order exactly."""
    bestfair=None;bestcombined=None
    for s,chs,neg,total in templates:
        check_cancelled(cancelled)
        if chs[0].shape[0]>=t['low'].shape[0] or chs[0].shape[1]>=t['low'].shape[1]:continue
        maps=[np.exp(-.5*(d/(SIGMA*s*R))**2) for d in t['dist']]
        support=sum(cv2.matchTemplate(m,ch,cv2.TM_CCORR) for m,ch in zip(maps,chs))/total
        # Preserve the screen-length normalization and float operation order.
        extra=np.maximum(0,cv2.matchTemplate(t['low'],neg,cv2.TM_CCORR))*s/total
        for mode,field in [('fair',support),('combined',support-PENALTY*extra)]:
            _,score,_,loc=cv2.minMaxLoc(field)
            c=dict(layout=t['layout'],floor=t['floor'],score=float(score),scale=s,x=loc[0]/R,y=loc[1]/R,wall_support=float(support[loc[1],loc[0]]),extra_wall_ratio=float(extra[loc[1],loc[0]]))
            if mode=='fair' and (bestfair is None or score>bestfair['score']):bestfair=c
            if mode=='combined' and (bestcombined is None or score>bestcombined['score']):bestcombined=c
    if bestfair is None:return None
    seeds=[bestcombined]
    if any(abs(bestfair[k]-bestcombined[k])>1e-6 for k in ['scale','x','y']):seeds.append(bestfair)
    poses=[]
    for seed in seeds:
        check_cancelled(cancelled)
        poses.append(score_pose(t,seed,qpoints,negative,box,query_geometry))
    return poses


def search(im, targets, executor=None, *, cancelled=None):
    check_cancelled(cancelled)
    start=time.perf_counter();edges=wall_edges(im,True)
    y,x=np.where(edges.sum(0)>0)
    if not len(x):return dict(floor_candidates=[],reason='no wall structure',elapsed_s=time.perf_counter()-start)
    box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    interior=visible_interior(im,edges);negative=cv2.GaussianBlur(interior.astype(np.float32),(5,5),1)
    crop=edges[:,box[1]:box[3],box[0]:box[2]]
    free=negative[box[1]:box[3],box[0]:box[2]]
    qpoints=[]
    for ch in edges:
        yy,xx=np.where(ch>0);qpoints.append(np.column_stack((xx,yy)).astype(np.float32))
    templates=[]
    for s in MINIMUM_SCALES:
        check_cancelled(cancelled)
        chs=[cv2.resize(ch.astype(np.float32),None,fx=s*R,fy=s*R,interpolation=cv2.INTER_AREA) for ch in crop]
        neg=cv2.resize(free,(chs[0].shape[1],chs[0].shape[0]),interpolation=cv2.INTER_AREA)
        templates.append((s,chs,neg,sum(ch.sum() for ch in chs)))
    origin=np.array(box[:2],np.float32)
    geometry=(origin,[p-origin for p in qpoints],sum(len(p) for p in qpoints))
    task=partial(search_target,templates=templates,qpoints=qpoints,negative=negative,
                 box=box,query_geometry=geometry, cancelled=cancelled)
    # Consume futures in input order, preserving same-score floor/seed ties.
    futures=[] if executor is None else [executor.submit(task,t) for t in targets]
    results=map(task,targets) if executor is None else (f.result() for f in futures)
    candidates=[]
    try:
        for result in results:
            check_cancelled(cancelled)
            if result is None:continue
            candidates.extend(result)
    finally:
        for future in futures:
            future.cancel()
    return dict(box=box,query_edges=edges,visible_interior_pixels=int(interior.sum()),floor_candidates=candidates,elapsed_s=time.perf_counter()-start)
