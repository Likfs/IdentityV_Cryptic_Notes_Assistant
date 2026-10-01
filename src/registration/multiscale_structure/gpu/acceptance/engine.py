"""Fixed-pose revalidation against visible evidence and every competing branch.

No image IDs, manual labels, CNN, route masks or session history enter decisions.
New acceptance uses observed-region stability plus full-viewport bounds, and
re-scores competitors using the same screenshot-owned edges and semantic pixels.
"""
from dataclasses import replace
from time import perf_counter
import numpy as np
import cv2
from src.registration.multiscale_structure.gpu.search_core.search import prepare_frame, local_to_clean
from src.registration.multiscale_structure.gpu.search_core.search_recovery import interior_evidence
from src.registration.multiscale_structure.gpu.search_core.pose_review import screen_disagreement
from src.registration.multiscale_structure.gpu.core.credible_geometry import score_joint, pose_matrix
from src.registration.multiscale_structure.gpu.search_core.bounded_policy import finite
from src.registration.multiscale_structure.gpu.search_core.engine import register as register_components
from .zoom_policy import decide as decide_zoom
from .joint_acceptance import review_single_category

POSITION = {'position_ambiguous', 'structure_not_distinctive', 'bounded_original_geometry',
            'position_components_at_current_limits'}
SCALE = {'wall_distribution_insufficient', 'single_wall_holdout_uncertain',
         'original_leave_one_wall_out_p95_px', 'extended_scale_evidence', 'scale_probe_boundary'}
MATERIAL = {'trusted_material_conflict', 'original_semantic_precision',
            'raster_structure_precision', 'raster_semantic_precision', 'combined_relaxation_debt'}
ALLOWED = POSITION | SCALE | MATERIAL


def eligible(validation, accepted, has_pose):
    if accepted or not has_pose: return False
    audit = validation.get('component_acceptance', {})
    checks = audit.get('checks', {})
    failures = {k for k, c in checks.items() if not c.get('passed')}
    return bool(failures) and failures <= ALLOWED and failures == set(audit.get('reasons', []))


def visible_review(validation, frame, template):
    g = validation.get('geometry', {}); checks = validation['component_acceptance']['checks']
    raster = validation.get('raster', {}); branches = g.get('branches', [])
    out = {'accepted': False, 'checks': {}, 'competitors': [],
           'note': 'Candidate/perturbation displacement is sensitivity, not ground-truth error.'}
    if frame is None or template is None or not branches:
        return {**out, 'reason': 'missing_live_evidence'}
    local = np.asarray(g.get('selected_local_pose'), np.float32)
    alternatives = np.asarray(g.get('uncertainty_poses', []), np.float32)
    branch_poses = np.asarray([[*b['anchor'], b['scale']] for b in branches], np.float32)
    scores = np.asarray([b['score'] for b in branches], np.float32)
    if (local.shape != (2,3) or not np.isfinite(local).all() or local[0,0] <= 0
            or not np.allclose(local[:,:2], np.eye(2)*local[0,0])
            or alternatives.ndim != 3 or alternatives.shape[1:] != (2,3) or len(alternatives) < 2
            or not np.isfinite(alternatives).all() or np.any(alternatives[:,0,0] <= 0)
            or branch_poses.shape != (len(branches),3) or not np.isfinite(branch_poses).all()
            or np.any(branch_poses[:,2] <= 0) or not np.isfinite(scores).all()):
        return {**out, 'reason': 'invalid_geometry'}
    e = frame.walls
    p = np.float32([*(local[:,:2]@e.center+local[:,2]), local[0,0]])
    best = int(scores.argmax())
    if not np.allclose(p, branch_poses[best], atol=.01):
        return {**out, 'reason': 'selected_branch_mismatch'}
    competitors = [i for i,q in enumerate(branch_poses) if i != best
        and (np.linalg.norm(q[:2]-p[:2])/p[2] > 20 or abs(np.log(q[2]/p[2])) > .04)
        and scores[i] >= scores[best]-.06]
    if len(competitors) != g.get('competitive_branches'):
        return {**out, 'reason': 'incomplete_competitor_inventory'}
    yy, xx = np.nonzero(frame.observation.map_mask)
    if len(xx) < 300 or len(e.walls) < 5:
        return {**out, 'reason': 'insufficient_visible_structure'}
    points = np.column_stack((xx, yy))[::max(1,len(xx)//2000)]
    poses = np.concatenate((p[None], branch_poses), axis=0)
    _, distances, semantics, background = score_joint(template.processed, e, poses)
    _, _, interior, _ = score_joint(template.processed, interior_evidence(e), poses)
    medians = np.median(distances, axis=2)
    support = np.mean(medians <= 3.5, axis=1)
    orientations = [int(np.sum((medians[0] <= 3.5)&(e.directions==d))) for d in (0,1)]
    visible_uncertainty = screen_disagreement(alternatives, local, points)

    def check(name, passed, **detail):
        out['checks'][name] = {'passed': bool(passed), **detail}

    # Material corruption is only tolerable when many independently extracted
    # walls support both axes. This does not suppress or relabel any input pixel.
    dense = (len(e.walls)>=32 and min(orientations)>=6 and support[0]>=.60
        and np.median(medians[0])<=2.5 and semantics[0]>=.78 and interior[0]>=.78
        and raster.get('structure_precision',0)>=.70 and raster.get('semantic_precision',0)>=.60
        and visible_uncertainty is not None and visible_uncertainty<=8)
    standard = (semantics[0]>=.92 and interior[0]>=.90
        and raster.get('structure_precision',0)>=.90 and raster.get('semantic_precision',0)>=.72
        and support[0]>=.65 and np.median(medians[0])<=3.2)
    check('selected_pose_quality', standard or dense, profile='dense_geometry_material_tolerance' if dense and not standard else 'standard',
          semantic=float(semantics[0]), interior=float(interior[0]), support=float(support[0]),
          median_screen_px=float(np.median(medians[0])), orientations=orientations,
          raster_semantic=raster.get('semantic_precision'), raster_semantic_threshold=.72,
          raster_structure=raster.get('structure_precision'))
    check('two_axis_support', min(orientations)>=2, value=orientations)
    # A sparse partial fit can be stable yet refer to the wrong repeated room.
    # When fewer than 3/4 of observed walls fit, require spatially richer support
    # in both directions instead of trusting its low median residual alone.
    check('partial_fit_redundancy', support[0]>=.75 or (len(e.walls)>=16 and min(orientations)>=5),
          support=float(support[0]), independent_segments=len(e.walls), orientations=orientations)
    check('visible_perturbation', finite(visible_uncertainty) and visible_uncertainty<=12,
          p95_screen_px=visible_uncertainty, threshold=12)
    full = [g.get('viewport_disagreement_p95_px'), g.get('leave_one_wall_out_p95_px')]
    check('full_viewport_bound', all(finite(x) for x in full) and full[0]<=40 and full[1]<=50,
          viewport_p95_px=full[0], holdout_p95_px=full[1])
    check('background_and_extent', background[0]<=.12 and g.get('reliable_wall_groups') is True
          and raster.get('mapped_structure_fraction',0)>=.95
          and checks.get('known_background_conflict',{}).get('passed') is True)
    # Failed unrelated checks cannot be discarded merely because some wall fit
    # looks good. Completeness and the search's hard scale limits remain gates.
    check('original_hard_gates', all(checks.get(k,{}).get('passed') is True for k in
        ['trusted_semantics_insufficient','scale_search_boundary','raster_mapped_structure_fraction',
         'trusted_edge_support','trusted_edge_median','viewport_scale_uncertain',
         'original_wall_support_fraction','original_viewport_disagreement_p95_px',
         'wide_current_edge_median','wide_original_edge_median','relaxation_family_budget']))
    for i in competitors:
        q = branch_poses[i]; j = i+1
        delta = float(np.linalg.norm(q[:2]-p[:2])/p[2]); ratio = float(q[2]/p[2])
        p95 = screen_disagreement([pose_matrix(q,e.center)], local, points)
        nearby = delta<=24 and abs(np.log(ratio))<=.10 and p95<=40
        semantic_dominated = (semantics[0]-semantics[j]>=.10 and interior[0]-interior[j]>=.10
                               and support[0]>=support[j]+.10)
        edge_dominated = (support[0]>=.75 and support[0]-support[j]>=.25
                          and support[j]<.75 and semantics[0]>=.92)
        passed = nearby or semantic_dominated or edge_dominated
        out['competitors'].append({'index':i,'passed':bool(passed),
            'basis':'nearby_scale_family' if nearby else 'material_and_support_dominated' if semantic_dominated else 'edge_support_dominated' if edge_dominated else 'unresolved_spatial_competitor',
            'anchor_screen_px':delta, 'relative_scale':ratio, 'visible_p95_screen_px':p95,
            'semantic':float(semantics[j]),'interior':float(interior[j]),'support':float(support[j])})
    check('all_competitors_resolved', all(x['passed'] for x in out['competitors']), count=len(competitors))
    if g.get('boundary'):
        # Extend a local probe in both directions instead of treating an edge
        # hit in the old narrow grid as sufficient evidence to reject/accept.
        grid = np.float32([[p[0]+dx*p[2],p[1]+dy*p[2],p[2]*ratio]
                          for ratio in np.linspace(.88,1.12,25) for dx in (-4,0,4) for dy in (-4,0,4)])
        values = score_joint(template.processed,e,grid)[0]; optimum = grid[int(values.argmax())]
        relative = float(optimum[2]/p[2])
        shift = screen_disagreement([pose_matrix(optimum,e.center)],local,points)
        check('extended_scale_probe', .89<relative<1.11 and shift<=12,
              best_relative_scale=relative, visible_p95_screen_px=shift, evaluated=len(grid))
    out['accepted'] = all(c['passed'] for c in out['checks'].values())
    out['reasons'] = [k for k,c in out['checks'].items() if not c['passed']]
    if out['reasons'] == ['selected_pose_quality']:
        supplement = review_single_category(out, raster)
        out['supplemental_review'] = supplement
        if supplement['accepted']:
            out['normal_reasons'] = list(out['reasons'])
            out['checks']['selected_pose_quality'].update(
                passed=True, normal_passed=False, profile=supplement['policy'])
            out.update(accepted=True, reasons=[])
    return out


def decide(validation, accepted, has_pose, frame=None, template=None):
    previous = decide_zoom(validation, accepted, has_pose, frame)
    audit = {'version':'multiscale_structure','baseline_component_accepted':bool(accepted),
             'baseline_zoom_accepted':previous['accepted'],'accepted':previous['accepted'],
             'basis':'unchanged_zoom_acceptance','reasons':previous['reasons'],'trusted_history_eligible':False}
    if previous['accepted'] or not eligible(validation, accepted, has_pose): return audit
    review = visible_review(validation,frame,template); audit['visible_review'] = review
    if review['accepted']:
        basis = ('single_category_geometry_support' if review.get('supplemental_review', {}).get('accepted')
                 else 'visible_pose_competition_review')
        audit.update(accepted=True,basis=basis,reasons=[])
    return audit


def apply_policy(baseline, frame=None, template=None):
    audit = decide(baseline.validation, baseline.accepted, baseline.clean_to_screen is not None, frame, template)
    if audit['accepted'] and not audit['baseline_zoom_accepted']:
        local = np.asarray(baseline.validation['geometry']['selected_local_pose'],np.float64)
        expected = cv2.invertAffineTransform(local_to_clean(local,template,frame.bbox))
        if not np.allclose(expected,baseline.clean_to_screen,atol=.001,rtol=1e-6):
            audit.update(accepted=False,basis='pose_coordinate_mismatch',reasons=['pose_coordinate_mismatch'])
    return replace(baseline,accepted=audit['accepted'],rejection_reasons=audit['reasons'],
                   validation={**baseline.validation,'visible_acceptance':audit,'trusted_history_eligible':False})


def register(image,template,*,options=None,frame=None):
    start=perf_counter();frame=frame or prepare_frame(image)
    baseline=register_components(image,template,options=options,frame=frame)
    result=apply_policy(baseline,frame,template)
    return replace(result,timings={**result.timings,'method':'visible_pose_review','total':(perf_counter()-start)*1000})
