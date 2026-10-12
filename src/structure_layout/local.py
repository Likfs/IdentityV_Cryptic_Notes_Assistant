"""Bidirectional local structure evidence. No semantic door/icon detector."""
import cv2,numpy as np
from .search import sample
from src.layout_retrieval import wall_edges

PARAMETERS=dict(tolerance_screen_px=2.0,error_ramp_px=4.0,cell_screen_px=32,
 min_cell_length_px=12,tail_cells=3,missing_weight=.15,extra_weight=.15,
 visibility_value_min=75,visibility_erode=3,raw_gradient_ratio=1.5,
 feature_raw_distance_max_px=1.5,distance_quantization=4)

def raw_edges(image):
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);edge=cv2.Canny(gray,25,65)>0
 h,s,v=cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
 ignore=cv2.dilate((((h<35)|(h>170))&(s>120)&(v>110)).astype(np.uint8),np.ones((9,9),np.uint8))>0
 edge[ignore]=False
 gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0));gy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1))
 return np.stack([edge&(gx>=1.5*gy),edge&(gy>=1.5*gx)]).astype(np.uint8)

def raw_distances(image,quantized=False):
 ds=[cv2.distanceTransform(1-ch,cv2.DIST_L2,5) for ch in raw_edges(image)]
 return [np.minimum(np.rint(d*4),255).astype(np.uint8) for d in ds] if quantized else ds

def evidence(image, edges=None):
 h,s,v=cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
 room=(h>=5)&(h<=32)&(s>=22)&(s<110)
 hall=((h>=90)&(h<=125)&(s<100))|(s<25)
 # Screenshot only. The broad Hough-wall exclusion is intentionally absent.
 known=cv2.erode(((room|hall)&(v>75)&(v<150)).astype(np.uint8),np.ones((3,3),np.uint8))
 edge=wall_edges(image,True) if edges is None else edges;ds=raw_distances(image)
 points=[]
 for ch,d in zip(edge,ds):
  y,x=np.where((ch>0)&(d<=1.5));points.append(np.column_stack([x,y]).astype(np.float32))
 return dict(known=known.astype(np.float32),dist=ds,points=points,shape=image.shape[:2])

def grouped_error(points,errors,weights,shape):
 """Global mean plus local tail; two grids reduce arbitrary cell-boundary effects."""
 if not len(points) or weights.sum()<1e-6:return 0.,[]
 global_mean=float(np.dot(errors,weights)/weights.sum());allcells=[];tails=[]
 for shift in [0,16]:
  gx=np.floor((points[:,0]+shift)/32).astype(int);gy=np.floor((points[:,1]+shift)/32).astype(int)
  cols=(shape[1]+shift)//32+2;idx=gy*cols+gx
  lengths=np.bincount(idx,weights=weights);sums=np.bincount(idx,weights=weights*errors)
  valid=np.where(lengths>=12)[0]
  vals=sums[valid]/lengths[valid]
  order=np.argsort(-vals)[:3];tails.append(float(vals[order].mean()) if len(order) else global_mean)
  for j in order:
   k=int(valid[j]);x=(k%cols)*32-shift;y=(k//cols)*32-shift
   if vals[j]>.01:allcells.append(dict(box=[x,y,x+32,y+32],error=float(vals[j]),length=float(lengths[k])))
 return .5*global_mean+.5*float(np.mean(tails)),sorted(allcells,key=lambda c:-c['error'])[:6]

def review(t,c,q,box):
 s=c['scale'];off=np.array([c['x'],c['y']],np.float32);origin=np.array(box[:2],np.float32)
 missing_p=[];missing_e=[];extra_p=[];extra_e=[];extra_w=[]
 for axis in [0,1]:
  p=q['points'][axis];xy=(p-origin)*s+off
  # Raw bottom-map evidence validates a feature miss before penalizing it.
  d=sample(t['rawdist'][axis],xy,255).astype(np.float32)/4/s
  missing_p.append(p);missing_e.append(np.clip((d-2)/4,0,1))
  tp=t['reliable_points'][axis]
  # Clip in target coordinates before conversion to avoid processing whole maps.
  lo=off-origin*s;hi=lo+np.array(q['shape'][::-1])*s
  tp=tp[((tp>=lo)&(tp<hi)).all(1)];sp=(tp-off)/s+origin
  visible=sample(q['known'],sp)
  keep=visible>.5;sp=sp[keep];visible=visible[keep]
  distance=sample(q['dist'][axis],sp,10000)
  extra_p.append(sp);extra_e.append(np.clip((distance-2)/4,0,1));extra_w.append(visible/s)
 mp=np.concatenate(missing_p);me=np.concatenate(missing_e);ep=np.concatenate(extra_p);ee=np.concatenate(extra_e);ew=np.concatenate(extra_w)
 missing,mc=grouped_error(mp,me,np.ones(len(mp)),q['shape']);extra,ec=grouped_error(ep,ee,ew,q['shape'])
 result=dict(base_score=c['score'],missing_local=missing,extra_local=extra,missing_penalty=.15*missing,extra_local_penalty=.15*extra,
  local_penalty=.15*(missing+extra),missing_cells=mc,extra_cells=ec,review_query_points=len(mp),review_target_visible_length=float(ew.sum()))
 return result

def attach_raw(t,image):
 t['rawdist']=raw_distances(image,True);t['reliable_points']=[]
 for p,d in zip(t['points'],t['rawdist']):
  keep=sample(d,p).astype(np.float32)/4<=1.5;t['reliable_points'].append(p[keep])
 return t
