"""Conservative contour review: repairing contours cannot grant new geometry confirmation."""
from dataclasses import replace
from time import perf_counter
from .search import prepare_frame
from .search_recovery import register as register_recovered
from .pose_review import evaluate_pose


def require_existing_geometry(result):
    audit=dict(result.validation['contour_acceptance'])
    # Recomputed interior semantic support must not borrow a confidence estimate
    # from another evidence set. recovered geometry's own confirmation of THIS returned pose is
    # required; its uncertain positions/scales remain uncertain in this release.
    confirmed=bool(result.validation.get('geometry',{}).get('accepted'))
    checks={**audit.get('checks',{}),'existing_geometry_confirmation':{
        'value':confirmed,'threshold':True,'passed':confirmed,'unit':'bool'}}
    reasons=list(audit['reasons'])
    if not confirmed and audit.get('candidate_found'):
        reasons.append('existing_geometry_not_confirmed')
    audit.update(checks=checks,accepted=not reasons and bool(audit.get('candidate_found')),
                 reasons=reasons,display_accepted=not reasons and bool(audit.get('candidate_found')),
                 stage='fixed_pose_contour_repair',scale_policy='unchanged recovered geometry geometry acceptance; new estimators shadow-only')
    if not confirmed:
        geometry=result.validation.get('geometry',{})
        audit['position_confirmed']=geometry.get('position_status')=='confirmed'
        audit['scale_confirmed']=False
    return replace(result,accepted=audit['accepted'],rejection_reasons=reasons,
                   validation={**result.validation,'contour_acceptance':audit})


def register(image,template,*,options=None,frame=None,collect_shadow=False):
    started=perf_counter()
    frame=frame or prepare_frame(image)
    baseline=register_recovered(image,template,options=options,frame=frame)
    audit=evaluate_pose(template,frame,baseline,grouped=collect_shadow)
    result=require_existing_geometry(replace(baseline,validation={**baseline.validation,'contour_acceptance':audit,'recovered_geometry_accepted':baseline.accepted}))
    return replace(result,timings={**result.timings,'method':'fixed_geometry_contour_repair','total':(perf_counter()-started)*1000})
