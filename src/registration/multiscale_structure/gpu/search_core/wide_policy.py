"""Wide reversible acceptance experiment; search and measured evidence stay fixed."""
from copy import deepcopy
from dataclasses import dataclass,asdict,replace
from time import perf_counter
from .strict import register as strict_register
from .bounded_policy import Policy,finite
from .combined_policy import decide as prior_decide


@dataclass(frozen=True)
class Limits:
    viewport_px: float=32.
    holdout_px: float=24.
    support: float=.55
    material: float=.90
    edge_median_px: float=3.5
    max_families: int=3
    debt_limit: float=4.


PROFILES={'wide':Limits(28,22,.60,.92,3.,3,3.),'geometry_tolerance':Limits(),
          'maximum':Limits(40,30,.50,.85,4.,4,6.)}


def decide(validation,accepted,has_pose,limits=Limits()):
    v=deepcopy(validation);checks=v.get('contour_acceptance',{}).get('checks',{});g=v.get('geometry',{})
    original_limits={}
    if not accepted and has_pose and 'trusted_edge_median' in checks:
        c=checks['trusted_edge_median'];strict=c.get('threshold');original_limits['current']=strict
        if finite(strict):
            c['threshold']=max(strict,limits.edge_median_px)
            c['passed']=finite(c.get('value')) and c['value']<=c['threshold']
        strict=g.get('wall_median_budget_screen_px');original_limits['original']=strict
        if finite(strict):g['wall_median_budget_screen_px']=max(strict,limits.edge_median_px)
    a=prior_decide(v,accepted,has_pose,Policy(limits.viewport_px,limits.holdout_px,limits.support,limits.material),
        max_families=limits.max_families,debt_limit=limits.debt_limit)
    a.update(version='geometry_tolerance',wide_policy=asdict(limits))
    if a['tier']=='strict' or 'combined_relaxation_debt' not in a['checks']:return a
    c=a['checks'];debts=[];used=False
    for name,value,strict in [('current',c['trusted_edge_median'].get('value'),original_limits.get('current')),
        ('original',g.get('wall_median_screen_px'),original_limits.get('original'))]:
        limit=max(strict,limits.edge_median_px) if finite(strict) else None
        good=finite(value) and finite(strict) and value<=limit
        c['wide_'+name+'_edge_median']={'value':value,'threshold':limit,'strict_threshold':strict,'passed':good,'unit':'screen_px'}
        if not finite(value) or not finite(strict):debts.append(None);used=True
        else:
            used|=value>strict
            debts.append(max(0.,(value-strict)/max(limit-strict,1e-9)))
    if used:a['relaxed_families']=sorted(set(a['relaxed_families'])|{'edge_residual'})
    edge_debt=None if None in debts else max(debts)
    a['relaxation_debts']['edge_residual']=edge_debt
    values=list(a['relaxation_debts'].values());total=None if None in values else sum(values)
    c['combined_relaxation_debt'].update(value=total,passed=total is not None and total<=limits.debt_limit+1e-9)
    c['relaxation_family_budget'].update(value=len(a['relaxed_families']),passed=len(a['relaxed_families'])<=limits.max_families)
    extra=[r for r in a['reasons'] if r not in c]
    reasons=extra+[k for k,val in c.items() if not val.get('passed')]
    a.update(accepted=not reasons,tier='tolerant' if not reasons else 'rejected',reasons=reasons)
    return a


def apply_policy(baseline,limits=Limits()):
    audit=decide(baseline.validation,baseline.accepted,baseline.clean_to_screen is not None,limits)
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
        validation={**baseline.validation,'component_acceptance':audit})


def register(image,template,*,options=None,frame=None):
    start=perf_counter();baseline=strict_register(image,template,options=options,frame=frame)
    result=apply_policy(baseline)
    return replace(result,timings={**result.timings,'method':'geometry_tolerance','total':(perf_counter()-start)*1000})
