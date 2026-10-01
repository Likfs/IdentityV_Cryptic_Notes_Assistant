"""Bounded local zoom families and guarded material tolerance.

Preserve candidate transforms and screenshot evidence while constraining
the acceptance of locally consistent zoom alternatives.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from src.registration.multiscale_structure.gpu.search_core.engine import register as register_components
from src.registration.multiscale_structure.gpu.search_core.search import prepare_frame
from src.registration.multiscale_structure.gpu.search_core.bounded_policy import finite


@dataclass(frozen=True)
class Limits:
    anchor_px: float = 24.
    scale_log: float = .10
    observed_p95_px: float = 32.
    strong_observed_p95_px: float = 56.
    material_floor: float = .83


AMBIGUITY = {'position_ambiguous', 'bounded_original_geometry', 'position_components_at_current_limits'}


def strong_geometry(g, checks):
    orientations = g.get('orientation_support', [])
    support = [g.get('wall_support_fraction'), checks.get('trusted_edge_support', {}).get('value')]
    medians = [g.get('wall_median_screen_px'), checks.get('trusted_edge_median', {}).get('value')]
    return (len(orientations) == 2 and all(finite(n) and n >= 3 for n in orientations)
            and all(finite(n) and n >= .75 for n in support)
            and all(finite(n) and n <= 2. for n in medians))


def local_zoom_family(validation, frame, limits):
    g = validation.get('geometry', {})
    checks = validation.get('component_acceptance', {}).get('checks', {})
    branches = g.get('branches') or []
    detail = {'accepted': False, 'competitors': [], 'basis': 'bounded_observed_zoom_family'}
    if frame is None or not branches:
        return {**detail, 'reason': 'missing_current_frame_or_branches'}
    poses = np.asarray([[*b['anchor'], b['scale']] for b in branches], np.float64)
    scores = np.asarray([b['score'] for b in branches], np.float64)
    local = np.asarray(g.get('selected_local_pose'), np.float64)
    if (poses.ndim != 2 or poses.shape[1] != 3 or local.shape != (2, 3)
            or not np.isfinite(poses).all() or not np.isfinite(scores).all()
            or not np.isfinite(local).all() or np.any(poses[:, 2] <= 0)):
        return {**detail, 'reason': 'invalid_geometry'}
    best = int(scores.argmax()); winner = poses[best]
    # Audit the same selected pose and the same screenshot evidence; never borrow
    # competitor evidence to certify an unrelated transform.
    anchor = local[:, :2] @ frame.walls.center + local[:, 2]
    if (not np.allclose(local[:, :2], np.eye(2)*winner[2], atol=1e-5)
            or not np.allclose(anchor, winner[:2], atol=.01)):
        return {**detail, 'reason': 'selected_pose_mismatch'}
    indices = [i for i, p in enumerate(poses) if i != best
               and (np.linalg.norm(p[:2]-winner[:2])/winner[2] > 20.
                    or abs(np.log(p[2]/winner[2])) > .04)
               and scores[i] >= scores[best]-.06]
    if not indices or len(indices) != g.get('competitive_branches'):
        return {**detail, 'reason': 'incomplete_competitor_inventory'}
    yy, xx = np.nonzero(frame.observation.map_mask)
    if len(xx) < 300:
        return {**detail, 'reason': 'insufficient_visible_structure'}
    points = np.column_stack((xx, yy))[::max(1, len(xx)//4000)]
    mapped = points @ local[:, :2].T + local[:, 2]
    strong = (strong_geometry(g, checks) and g.get('wall_segments', 0) >= 20
              and finite(g.get('semantic_precision')) and g['semantic_precision'] >= .89)
    # A close zoom family cannot rescue a weak original material interpretation.
    # Interior-only colour agreement can hide a wrong room/corridor association.
    original_material = g.get('semantic_precision')
    material_guard = finite(original_material) and (original_material >= .90 or strong)
    detail['original_material_guard'] = {'passed': bool(material_guard),
                                       'value': original_material, 'strong_geometry': bool(strong)}
    if not material_guard:
        return {**detail, 'reason': 'local_family_original_material_too_weak'}
    budget = limits.strong_observed_p95_px if strong else limits.observed_p95_px
    for index in indices:
        p = poses[index]
        delta = np.linalg.norm(p[:2]-winner[:2])/winner[2]
        scale_delta = abs(np.log(p[2]/winner[2]))
        projected = (mapped-p[:2])/p[2]+frame.walls.center
        p95 = float(np.percentile(np.linalg.norm(projected-points, axis=1), 95))
        passed = delta <= limits.anchor_px and scale_delta <= limits.scale_log and p95 <= budget
        detail['competitors'].append({'index': index, 'anchor_distance_screen_px': float(delta),
            'relative_scale': float(p[2]/winner[2]), 'observed_disagreement_p95_px': p95,
            'observed_budget_px': budget, 'passed': bool(passed)})
    detail.update(accepted=all(c['passed'] for c in detail['competitors']), strong_geometry=bool(strong),
                  uncertainty_note='Candidate disagreement is not ground-truth error or a confidence interval.')
    return detail


def decide(validation, accepted, has_pose, frame=None, limits=Limits()):
    original = validation.get('component_acceptance', {})
    audit = {'version': 'bounded_zoom_family', 'baseline_accepted': bool(accepted),
             'accepted': bool(accepted), 'basis': 'unchanged_component_acceptance',
             'reasons': list(original.get('reasons', [])), 'trusted_history_eligible': False}
    if accepted:
        return audit
    checks = original.get('checks', {})
    g = validation.get('geometry', {})
    failures = {k for k, c in checks.items() if not c.get('passed')}
    # Only these explicit combinations are eligible. Unknown failure reasons,
    # missing evidence, boundaries, background conflicts and scale vetoes stay.
    reasons = set(original.get('reasons', []))
    if not has_pose or not failures or reasons != failures:
        return audit
    if failures == AMBIGUITY:
        family = local_zoom_family(validation, frame, limits)
        audit['local_zoom_family'] = family
        if family['accepted']:
            audit.update(accepted=True, basis='bounded_local_zoom_family', reasons=[])
    elif failures == {'trusted_material_conflict'}:
        value = checks['trusted_material_conflict'].get('value')
        raster = validation.get('raster', {})
        stable = all(finite(g.get(k)) and g[k] <= 12. for k in
                     ['viewport_disagreement_p95_px', 'leave_one_wall_out_p95_px'])
        guard = (strong_geometry(g, checks) and stable and g.get('competitive_branches') == 0
                 and finite(g.get('semantic_precision')) and g['semantic_precision'] >= .89
                 and finite(raster.get('structure_precision')) and raster['structure_precision'] >= .90
                 and finite(raster.get('semantic_precision')) and raster['semantic_precision'] >= .85)
        passed = guard and finite(value) and value >= limits.material_floor
        audit['material_guard'] = {'passed': bool(passed), 'strong_geometry': bool(guard),
                                   'value': value, 'threshold': limits.material_floor}
        if passed:
            audit.update(accepted=True, basis='strong_geometry_material_tolerance', reasons=[])
    return audit


def apply_policy(baseline, frame=None, limits=Limits()):
    audit = decide(baseline.validation, baseline.accepted, baseline.clean_to_screen is not None, frame, limits)
    return replace(baseline, accepted=audit['accepted'], rejection_reasons=audit['reasons'],
                   validation={**baseline.validation, 'zoom_acceptance': audit, 'trusted_history_eligible': False})


def register(image, template, *, options=None, frame=None):
    start = perf_counter()
    frame = frame or prepare_frame(image)
    baseline = register_components(image, template, options=options, frame=frame)
    result = apply_policy(baseline, frame)
    return replace(result, timings={**result.timings, 'method': 'bounded_zoom_family',
                                    'total': (perf_counter()-start)*1000})
