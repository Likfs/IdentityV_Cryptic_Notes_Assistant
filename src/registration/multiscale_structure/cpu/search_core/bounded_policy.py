"""Keep contour review poses; allow one bounded family of marginal evidence at a time."""
from copy import deepcopy
from dataclasses import dataclass,replace,asdict
from math import isfinite
from time import perf_counter
from .strict import register as strict_register


@dataclass(frozen=True)
class Policy:
    viewport_px: float = 12.
    holdout_px: float = 12.
    support: float = .70
    material: float = .97


PROFILES={'mild':Policy(10,10,.72,.975),'balanced':Policy(),'wide':Policy(16,14,.66,.96)}
FAMILIES={'viewport_scale_uncertain':'scale','single_wall_holdout_uncertain':'scale',
          'trusted_edge_support':'support','trusted_material_conflict':'material'}


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and isfinite(value)


def decide(validation,accepted,has_pose,policy=Policy()):
    audit=deepcopy(validation.get('contour_acceptance',{}));checks=audit.get('checks',{})
    g=validation.get('geometry',{})
    info={'version':'bounded_tolerance','policy':asdict(policy),'baseline_accepted':bool(accepted),
          'accepted':bool(accepted),'tier':'strict' if accepted else 'rejected',
          'relaxed_families':[],'trusted_history_eligible':False,
          'history_note':'No new trusted history authorization; preserve existing caller restrictions.'}
    if accepted:return {**info,'checks':checks,'reasons':[]}
    required={'trusted_semantics_insufficient','position_ambiguous','structure_not_distinctive',
        'trusted_material_conflict','known_background_conflict','wall_distribution_insufficient',
        'trusted_edge_support','trusted_edge_median','viewport_scale_uncertain','single_wall_holdout_uncertain',
        'scale_probe_boundary','scale_search_boundary','raster_mapped_structure_fraction',
        'raster_structure_precision','raster_semantic_precision','existing_geometry_confirmation'}
    if not has_pose or not required.issubset(checks):
        return {**info,'checks':checks,'reasons':['incomplete_fixed_pose_evidence']}
    unexpected=set(audit.get('reasons',[]))-required-{'existing_geometry_not_confirmed'}
    limits={'viewport_scale_uncertain':policy.viewport_px,'single_wall_holdout_uncertain':policy.holdout_px,
            'trusted_edge_support':policy.support,'trusted_material_conflict':policy.material}
    used=set()
    for name,limit in limits.items():
        c=checks[name];value=c.get('value');original=bool(c.get('passed'))
        passed=finite(value) and (value<=limit if name.endswith('uncertain') else value>=limit)
        if not original:used.add(FAMILIES[name])
        c.update(strict_threshold=c['threshold'],threshold=limit,passed=passed)
    # Require the selected pose's original evidence as well as contour review's interior
    # recheck. Failed recovery evidence must never confirm a different pose.
    original_limits=[('semantic_precision',policy.material,True,.98,'material'),
        ('wall_support_fraction',policy.support,True,.75,'support'),
        ('viewport_disagreement_p95_px',policy.viewport_px,False,8.,'scale'),
        ('leave_one_wall_out_p95_px',policy.holdout_px,False,10.,'scale')]
    original_ok=True
    for name,limit,minimum,strict,family in original_limits:
        value=g.get(name);passed=finite(value) and (value>=limit if minimum else value<=limit)
        if not finite(value) or (value<strict if minimum else value>strict):used.add(family)
        checks['original_'+name]={'value':value,'threshold':limit,'passed':passed,
                                  'unit':'fraction' if minimum else 'screen_px'}
        original_ok &= passed
    original_ok &= (g.get('position_status')=='confirmed' and g.get('reliable_wall_groups') is True
        and g.get('competitive_branches')==0 and g.get('distinctive_structure') is True
        and g.get('boundary') is False and finite(g.get('background_conflict'))
        and g['background_conflict']<=.12 and finite(g.get('wall_median_screen_px'))
        and finite(g.get('wall_median_budget_screen_px'))
        and g['wall_median_screen_px']<=g['wall_median_budget_screen_px'])
    # Multiple slightly weak families must not add up to a weak global decision.
    checks['bounded_original_geometry']={'value':bool(original_ok),'threshold':True,'passed':bool(original_ok),'unit':'bool'}
    checks['relaxation_family_budget']={'value':len(used),'threshold':1,'passed':len(used)<=1,'unit':'families'}
    checks.pop('existing_geometry_confirmation')
    reasons=sorted(unexpected)+[k for k,c in checks.items() if not c.get('passed')]
    return {**info,'accepted':not reasons,'tier':'tolerant' if not reasons else 'rejected',
            'relaxed_families':sorted(used),'checks':checks,'reasons':reasons}


def apply_policy(baseline,policy=Policy()):
    audit=decide(baseline.validation,baseline.accepted,baseline.clean_to_screen is not None,policy)
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
        validation={**baseline.validation,'component_acceptance':audit})


def register(image,template,*,options=None,frame=None,policy=Policy()):
    start=perf_counter();baseline=strict_register(image,template,options=options,frame=frame)
    result=apply_policy(baseline,policy)
    return replace(result,timings={**result.timings,'method':'bounded_tolerance','total':(perf_counter()-start)*1000})
