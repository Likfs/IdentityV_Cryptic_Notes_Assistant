"""Fixed-pose recovered geometry search, screenshot-owned edge review, grouped scale audit."""
from dataclasses import replace
from time import perf_counter
import cv2
import numpy as np
from src.registration.multiscale_structure.gpu.core.credible_geometry import score_joint, _line_fit, pose_matrix
from .search import prepare_frame
from .search_recovery import register as register_recovered, interior_evidence

REVIEW_METHOD = 'contour_evidence_shadow_scale'


def contour_evidence(frame):
    """Classify raw contours before consulting any template or candidate."""
    raw=frame.observation.edge_mask>0
    strokes=np.zeros(raw.shape,np.uint8)
    for wall in frame.walls.walls:
        a,b=np.rint(wall).astype(int).reshape(2,2)
        cv2.line(strokes,tuple(a),tuple(b),255,1)
    near=cv2.distanceTransform(255-strokes,cv2.DIST_L2,3)<=4 if np.any(strokes) else np.zeros(raw.shape,bool)
    hsv=cv2.cvtColor(frame.observation_image,cv2.COLOR_BGR2HSV)
    icons=cv2.dilate(np.uint8((hsv[:,:,1]>105)&(hsv[:,:,2]>135))*255,np.ones((5,5),np.uint8))>0
    if frame.observation.ignore_mask is not None:icons|=frame.observation.ignore_mask>0
    ignored=raw&icons
    trusted=raw&near&~icons
    return {'trusted':trusted,'ignored':ignored,'uncertain':raw&~trusted&~ignored}


def wall_groups(evidence, tolerance=4.):
    """Collinear fragments share one normal constraint; screenshot-only groups."""
    groups = []
    for direction in (0, 1):
        normal = 1-direction
        indices = np.flatnonzero(evidence.directions == direction)
        order = indices[np.argsort(evidence.walls[indices, normal], kind='stable')]
        selected = []
        for index in order:
            if selected and evidence.walls[index, normal]-evidence.walls[selected[0], normal] > tolerance:
                groups.append(selected); selected = []
            selected.append(int(index))
        if selected: groups.append(selected)
    return groups


def screen_disagreement(matrices, winner, points):
    if not len(points) or not len(matrices): return None
    clean = points @ winner[:, :2].T + winner[:, 2]
    return max(float(np.percentile(np.linalg.norm(
        (clean-m[:, 2])/m[0, 0]-points, axis=1), 95)) for m in matrices)


def grouped_audit(template, evidence, winner, shape, bbox):
    """Leave out independent normal groups, not just individual fragments."""
    pose = np.float32([*(winner[:, :2] @ evidence.center + winner[:, 2]), winner[0, 0]])
    groups = wall_groups(evidence)
    matrices = []
    for group in groups:
        keep = np.ones(len(evidence.walls), bool); keep[group] = False
        reduced = replace(evidence, walls=evidence.walls[keep], directions=evidence.directions[keep], points=evidence.points[keep])
        if min(np.sum(reduced.directions == d) for d in (0,1)) < 2:
            return {'sufficient': False, 'groups': len(groups), 'viewport_p95_px': None, 'poses': []}
        fitted = _line_fit(template, reduced, pose)
        matrices.append(pose_matrix(fitted, evidence.center))
    xx, yy = np.meshgrid(np.linspace(0, shape[1]-1, 5), np.linspace(0, shape[0]-1, 5))
    points = np.column_stack((xx.ravel(), yy.ravel()))-np.asarray(bbox[:2])
    return {'sufficient': len(groups) >= 4, 'groups': len(groups),
            'viewport_p95_px': screen_disagreement(matrices, winner, points), 'poses': [m.tolist() for m in matrices]}


def axis_audit(template, evidence, winner, shape, bbox):
    """Independent horizontal/vertical normal fits detect unequal scale evidence."""
    p=np.float32([*(winner[:,:2]@evidence.center+winner[:,2]),winner[0,0]])
    _line_fit(template,evidence,p)  # Populate immutable nearest-wall fields.
    axes=[]
    for direction in (0,1):
        xs=[];ys=[]
        for group in wall_groups(evidence):
            if evidence.directions[group[0]]!=direction:continue
            pts=evidence.points[group].reshape(-1,2)
            xy=pts@winner[:,:2].T+winner[:,2]
            target=cv2.remap(template._reliable_nearest[direction],xy[:,0][None],xy[:,1][None],cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,borderValue=(-1000,-1000))[0]
            good=np.linalg.norm(target-xy,axis=1)/p[2]<8
            if np.count_nonzero(good)<6:continue
            xs.append(float(np.median(pts[good,1-direction])));ys.append(float(np.median(target[good,1-direction])))
        x=np.asarray(xs);y=np.asarray(ys)
        if len(x)<3 or np.ptp(x)<100:
            return {'sufficient':False,'axes':axes,'disagreement_p95_px':None}
        design=np.column_stack((x,np.ones(len(x))));coef=np.linalg.lstsq(design,y,rcond=None)[0]
        for _ in range(5):
            residual=np.abs(design@coef-y)/p[2];w=np.minimum(1.,1.5/np.maximum(residual,.01))
            coef=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        axes.append({'scale':float(coef[0]),'offset':float(coef[1]),'groups':len(x),'span':float(np.ptp(x)),
                     'residual_screen_px':float(np.median(np.abs(design@coef-y))/p[2])})
    # Quantify scale conflict at the viewport, using the same observation anchor.
    matrices=[pose_matrix(np.float32([*p[:2],a['scale']]),evidence.center) for a in axes if a['scale']>0]
    xx,yy=np.meshgrid(np.linspace(0,shape[1]-1,5),np.linspace(0,shape[0]-1,5))
    points=np.column_stack((xx.ravel(),yy.ravel()))-np.asarray(bbox[:2])
    return {'sufficient':len(matrices)==2,'axes':axes,
        'disagreement_p95_px':screen_disagreement(matrices[1:],matrices[0],points) if len(matrices)==2 else None}


def projection_risk(template, winner, alternatives, shape, bbox):
    """Diagnostics only. All on-screen map structure, including unexplored parts."""
    ys, xs = np.nonzero(template.map_mask[::8, ::8])
    clean = np.column_stack((xs*8, ys*8)).astype(np.float32)
    screen = (clean-winner[:, 2])/winner[0,0] + np.asarray(bbox[:2])
    keep = (screen[:,0]>=0)&(screen[:,0]<shape[1])&(screen[:,1]>=0)&(screen[:,1]<shape[0])
    screen = screen[keep]
    if not len(screen): return {'points': 0, 'p95_px': None, 'worst_tile_p95_px': None}
    matrices = np.asarray(alternatives, np.float32).reshape(-1,2,3)
    points = screen-np.asarray(bbox[:2])
    if not len(matrices):
        return {'points':len(points),'p95_px':None,'worst_tile_p95_px':None}
    # Each alternative's point displacement is independent of tile membership.
    # Compute it once and reuse identical values for all tile percentiles.
    mapped = points @ winner[:, :2].T + winner[:, 2]
    distances = np.stack([np.linalg.norm((mapped-m[:,2])/m[0,0]-points,axis=1) for m in matrices])
    value = float(np.percentile(distances,95,axis=1).max())
    tiles = np.floor(screen/np.asarray([shape[1]/3,shape[0]/3])).astype(int)
    local = []
    for x in range(3):
        for y in range(3):
            mask = (tiles[:,0]==x)&(tiles[:,1]==y)
            if np.any(mask): local.append(float(np.percentile(distances[:,mask],95,axis=1).max()))
    return {'points':len(points),'p95_px':value,'worst_tile_p95_px':max(local) if local else None}


def evaluate_pose(template, frame, result, *, grouped=True):
    """Never alter the recovered geometry returned pose. Every metric belongs to that pose."""
    contour=contour_evidence(frame)
    g = result.validation.get('geometry', {})
    if result.clean_to_screen is None or 'selected_local_pose' not in g:
        return {'accepted': False, 'reasons': list(result.rejection_reasons) or ['no_candidate'],
                'candidate_found': result.clean_to_screen is not None}
    local = np.asarray(g['selected_local_pose'],np.float32)
    evidence = interior_evidence(frame.walls)
    if len(evidence.semantic)<40 or min(np.count_nonzero(evidence.directions==d) for d in (0,1))<2:
        return {'accepted':False,'reasons':['trusted_geometry_insufficient'],'candidate_found':True,
                'position_confirmed':False,'scale_confirmed':False,'candidate_local_pose':local.tolist()}
    p = np.float32([*(local[:,:2] @ evidence.center+local[:,2]),local[0,0]])
    _, distances, semantic, negative = score_joint(template.processed,evidence,p[None])
    medians = np.median(distances[0],axis=1)
    supported = medians <= 3.5
    normals = [evidence.walls[supported & (evidence.directions==d),1-d] for d in (0,1)]
    distributed = all(len(n)>=2 and np.ptp(n)>=25 for n in normals)
    budget = 1.5+.5/max(float(p[2]),.25)
    # These geometric/semantic thresholds are unchanged from recovered geometry's reliable path.
    checks = {
        'trusted_semantics_insufficient': (len(evidence.semantic),40,len(evidence.semantic)>=40,'points'),
        'position_ambiguous': (g.get('competitive_branches',0),0,g.get('competitive_branches',0)==0,'branches'),
        'structure_not_distinctive': (g.get('distinctive_structure',False),True,g.get('distinctive_structure',False),'bool'),
        'trusted_material_conflict': (float(semantic[0]),.98,semantic[0]>=.98,'fraction'),
        'known_background_conflict': (float(negative[0]),.12,negative[0]<=.12,'fraction'),
        'wall_distribution_insufficient': (distributed,True,distributed,'bool'),
        'trusted_edge_support': (float(np.mean(supported)) if len(supported) else 0.,.75,len(supported)>0 and np.mean(supported)>=.75,'fraction'),
        'trusted_edge_median': (float(np.median(medians)) if len(medians) else None,budget,len(medians)>0 and np.median(medians)<=budget,'screen_px'),
        'viewport_scale_uncertain': (g.get('viewport_disagreement_p95_px'),8,g.get('viewport_disagreement_p95_px') is not None and g['viewport_disagreement_p95_px']<=8,'screen_px'),
        'single_wall_holdout_uncertain': (g.get('leave_one_wall_out_p95_px'),10,g.get('leave_one_wall_out_p95_px') is not None and g['leave_one_wall_out_p95_px']<=10,'screen_px'),
        'scale_probe_boundary': (g.get('boundary',True),False,not g.get('boundary',True),'bool'),
    }
    raster = result.validation.get('raster',{})
    # Retain full positive structure/material checks; only unreliable contour veto is replaced.
    for key, threshold in [('mapped_structure_fraction',.95),('structure_precision',.90),('semantic_precision',.82)]:
        checks['raster_'+key]=(raster.get(key,0),threshold,raster.get(key,0)>=threshold,'fraction')
    old_boundary = 'scale_search_boundary' in result.rejection_reasons
    checks['scale_search_boundary']=(old_boundary,False,not old_boundary,'bool')
    group = grouped_audit(template.processed,evidence,local,frame.image.shape,frame.bbox) if grouped else None
    axis=axis_audit(template.processed,evidence,local,frame.image.shape,frame.bbox) if grouped else None
    detail = {key:{'value':float(value) if isinstance(value,np.floating) else value,
        'threshold':threshold,'passed':bool(passed),'unit':unit} for key,(value,threshold,passed,unit) in checks.items()}
    reasons = [key for key, value in detail.items() if not value['passed']]
    position_keys=['trusted_semantics_insufficient','position_ambiguous','structure_not_distinctive','trusted_material_conflict','known_background_conflict']
    scale_keys=['wall_distribution_insufficient','trusted_edge_support','trusted_edge_median','viewport_scale_uncertain','single_wall_holdout_uncertain','scale_probe_boundary','scale_search_boundary']
    return {'accepted':not reasons,'reasons':reasons,'candidate_found':True,
        'position_confirmed':all(detail[k]['passed'] for k in position_keys),
        'scale_confirmed':all(detail[k]['passed'] for k in scale_keys),
        'candidate_local_pose':local.tolist(),'checks':detail,'group_audit_shadow':group,'axis_audit_shadow':axis,
        'contour_counts':{key:int(np.count_nonzero(mask)) for key,mask in contour.items()},
        'projection_risk_shadow':projection_risk(template.processed,local,g.get('uncertainty_poses',[]),frame.image.shape,frame.bbox),
        'raw_contour_edge_median_screen_px':raster.get('edge_median_error_px')/float(p[2]) if raster.get('edge_median_error_px') is not None else None,
        'edge_evidence':'screenshot-owned reliable wall groups; shared with geometry, not independent votes',
        'scale_policy':'original recovered geometry viewport thresholds retained; projection-domain replacement shadow-only'}


def register(image, template, *, options=None, frame=None):
    started=perf_counter()
    frame=frame or prepare_frame(image)
    baseline=register_recovered(image,template,options=options,frame=frame)
    audit=evaluate_pose(template,frame,baseline)
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
        validation={**baseline.validation,'contour_acceptance':audit,'recovered_geometry_accepted':baseline.accepted},
        timings={**baseline.timings,'method':REVIEW_METHOD,'total':(perf_counter()-started)*1000})
