"""Re-evaluate position evidence at current limits instead of a stale verdict."""
from dataclasses import dataclass,asdict,replace
from time import perf_counter
from .strict import register as strict_register
from .bounded_policy import finite
from .wide_policy import Limits as PreviousLimits,decide as previous_decide


@dataclass(frozen=True)
class Limits(PreviousLimits):
    viewport_px: float=40.
    holdout_px: float=36.
    support: float=.50
    material: float=.85
    edge_median_px: float=4.
    max_families: int=4
    debt_limit: float=7.
    raster_structure: float=.85
    raster_semantic: float=.72
    raster_mapped: float=.90


PROFILES={'status_fix':Limits(32,24,.55,.90,3.5,3,4.,.90,.82,.95),
          'component_tolerance':Limits(),
          'wider':Limits(48,40,.45,.80,4.5,5,9.,.80,.68,.85)}


def decide(validation,accepted,has_pose,limits=Limits()):
    a=previous_decide(validation,accepted,has_pose,limits)
    a.update(version='component_tolerance',component_policy=asdict(limits))
    c=a['checks'];g=validation.get('geometry',{})
    if a['tier']=='strict' or 'combined_relaxation_debt' not in c:return a
    # Old position_status already embeds semantic >= .94. It must not silently
    # override the current semantic gate. Rebuild from the actual components;
    # keep ambiguous/unknown states rejected and never change original evidence.
    position=(g.get('position_status') in ('confirmed','insufficient_evidence')
        and g.get('distinctive_structure') is True and g.get('competitive_branches')==0
        and finite(g.get('semantic_precision')) and g['semantic_precision']>=limits.material
        and finite(g.get('background_conflict')) and g['background_conflict']<=.12)
    c['position_components_at_current_limits']={'value':bool(position),'threshold':True,'passed':bool(position),'unit':'bool'}
    a['original_position_status']=g.get('position_status')
    geometry=(position and g.get('reliable_wall_groups') is True and g.get('boundary') is False
        and all(c[k]['passed'] for k in ('original_semantic_precision','original_wall_support_fraction',
            'original_viewport_disagreement_p95_px','original_leave_one_wall_out_p95_px','wide_original_edge_median')))
    c['bounded_original_geometry'].update(value=bool(geometry),passed=bool(geometry))
    # Larger scale uncertainty needs more than the minimum two supported walls
    # per direction. A narrow repeated room can otherwise look locally perfect.
    measurements=[g.get('viewport_disagreement_p95_px'),c['viewport_scale_uncertain'].get('value'),
        g.get('leave_one_wall_out_p95_px'),c['single_wall_holdout_uncertain'].get('value')]
    known=all(finite(x) for x in measurements)
    extended=known and (max(measurements[:2])>32 or max(measurements[2:])>24)
    orientations=g.get('orientation_support',[])
    support=[g.get('wall_support_fraction'),c['trusted_edge_support'].get('value')]
    residuals=[g.get('wall_median_screen_px'),c['trusted_edge_median'].get('value')]
    strong=(len(orientations)==2 and all(finite(x) and x>=3 for x in orientations)
        and all(finite(x) and x>=.75 for x in support)
        and all(finite(x) and x<=2.5 for x in residuals))
    c['extended_scale_evidence']={'value':{'needed':extended,'orientation_support':orientations,
        'support':support,'median_px':residuals},'threshold':{'per_direction':3,'support':.75,'median_px':2.5},
        'passed':known and (not extended or strong),'unit':'compound'}
    deficits=[];used=False
    for name,limit,reference in [('structure_precision',limits.raster_structure,.90),
        ('semantic_precision',limits.raster_semantic,.82),('mapped_structure_fraction',limits.raster_mapped,.95)]:
        check=c['raster_'+name];value=check.get('value')
        check.update(strict_threshold=reference,threshold=limit,passed=finite(value) and value>=limit)
        if not finite(value):deficits.append(None);used=True
        else:
            used|=value<reference
            deficits.append(max(0.,(reference-value)/max(reference-limit,1e-9)))
    if used:a['relaxed_families']=sorted(set(a['relaxed_families'])|{'raster'})
    a['relaxation_debts']['raster']=None if None in deficits else max(deficits)
    values=list(a['relaxation_debts'].values());total=None if None in values else sum(values)
    c['combined_relaxation_debt'].update(value=total,passed=total is not None and total<=limits.debt_limit+1e-9)
    c['relaxation_family_budget'].update(value=len(a['relaxed_families']),passed=len(a['relaxed_families'])<=limits.max_families)
    reasons=[r for r in a['reasons'] if r not in c]+[k for k,val in c.items() if not val.get('passed')]
    a.update(accepted=not reasons,tier='tolerant' if not reasons else 'rejected',reasons=reasons)
    return a


def apply_policy(baseline,limits=Limits()):
    audit=decide(baseline.validation,baseline.accepted,baseline.clean_to_screen is not None,limits)
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
        validation={**baseline.validation,'component_acceptance':audit})


def register(image,template,*,options=None,frame=None):
    start=perf_counter();baseline=strict_register(image,template,options=options,frame=frame)
    result=apply_policy(baseline)
    return replace(result,timings={**result.timings,'method':'component_tolerance','total':(perf_counter()-start)*1000})
