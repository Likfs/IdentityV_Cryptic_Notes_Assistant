"""Allow two mild evidence deficits while bounding their combined magnitude."""
from dataclasses import replace
from time import perf_counter
from .strict import register as strict_register
from .bounded_policy import Policy,decide as previous_decide,finite

POLICY=Policy(20.,16.,.65,.95)


def decide(validation,accepted,has_pose,policy=POLICY,*,max_families=2,debt_limit=1.25):
    audit=previous_decide(validation,accepted,has_pose,policy)
    audit['version']='combined_tolerance'
    audit['combined_policy']={'max_families':max_families,'debt_limit':debt_limit,
        'reference_limits':{'viewport':[8,20],'holdout':[10,16],'support':[.75,.65],'material':[.98,.95]}}
    checks=audit['checks']
    if audit['tier']=='strict' or 'relaxation_family_budget' not in checks:return audit
    def deficit(keys,strict,span,minimum=False):
        values=[checks[k].get('value') for k in keys]
        if any(not finite(v) for v in values):return None
        return max(0.,max((strict-v if minimum else v-strict)/span for v in values))
    viewport=deficit(['viewport_scale_uncertain','original_viewport_disagreement_p95_px'],8,12)
    holdout=deficit(['single_wall_holdout_uncertain','original_leave_one_wall_out_p95_px'],10,6)
    debts={'scale':None if viewport is None or holdout is None else max(viewport,holdout),
        'support':deficit(['trusted_edge_support','original_wall_support_fraction'],.75,.10,True),
        'material':deficit(['trusted_material_conflict','original_semantic_precision'],.98,.03,True)}
    total=None if any(v is None for v in debts.values()) else sum(debts.values())
    count=len(audit['relaxed_families'])
    checks['relaxation_family_budget'].update(value=count,threshold=max_families,passed=count<=max_families)
    checks['combined_relaxation_debt']={'value':total,'threshold':debt_limit,
        'passed':total is not None and total<=debt_limit+1e-9,'unit':'normalized_deficit'}
    # Preserve unknown/hard vetoes; only recompute the explicit family gate.
    extra=[r for r in audit['reasons'] if r not in checks]
    reasons=extra+[k for k,c in checks.items() if not c.get('passed')]
    audit.update(accepted=not reasons,tier='tolerant' if not reasons else 'rejected',
                 reasons=reasons,relaxation_debts=debts)
    return audit


def apply_policy(baseline):
    audit=decide(baseline.validation,baseline.accepted,baseline.clean_to_screen is not None)
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
        validation={**baseline.validation,'component_acceptance':audit})


def register(image,template,*,options=None,frame=None):
    start=perf_counter();baseline=strict_register(image,template,options=options,frame=frame)
    result=apply_policy(baseline)
    return replace(result,timings={**result.timings,'method':'combined_tolerance','total':(perf_counter()-start)*1000})
