"""Bounded recovery of rejected poses, with separate local/full confirmation."""
from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from src.registration_core.credible_geometry import screenshot_evidence, review_geometry


def boundary_interval(scale, bounds):
    """Only search the missing interval, with one overlapping coarse cell."""
    low, high = bounds
    tolerance = max(.009, low*.02)
    if scale <= low+tolerance and low > .251:
        return (.25, min(high,low+.05))
    if scale >= high-max(.009,high*.01) and high < 2.749:
        return (max(low,high-.05),min(2.75,high*1.4))
    return None


def boundary_coverage(matrix, segmentation, template, tolerance=1.5):
    field = getattr(template,'_recovery_structure_distance',None)
    if field is None:
        field = cv2.distanceTransform(255-template.map_mask,cv2.DIST_L2,3)
        template._recovery_structure_distance = field
    yy,xx = np.nonzero(segmentation.map_mask)
    points = np.float32(np.column_stack((xx,yy)))
    count = 0
    for offset in range(0,len(points),16000):
        xy = points[offset:offset+16000]@matrix[:,:2].T+matrix[:,2]
        distances = cv2.remap(field,xy[:,0][None],xy[:,1][None],cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,borderValue=1000)[0]/matrix[0,0]
        count += int(np.count_nonzero(distances<=tolerance))
    return count/max(1,len(points))


def review_result(image, prepared, result, bounds):
    from . import runtime_registration as runtime
    if result.screenshot_to_clean is None or result.observation_bbox is None:
        return result
    started = perf_counter()
    segmentation = result.source_segmentation
    if segmentation is None:
        segmentation = runtime.segment_map(image,prepared.config,infer_known_background=False,include_stairs=False)
    x,y,w,h = result.observation_bbox
    observation = runtime.add_known_background_evidence(image[y:y+h,x:x+w],
        runtime._crop_segmentation(segmentation,(x,y,w,h)),prepared.config)
    evidence = screenshot_evidence(image[y:y+h,x:x+w],observation,reliable=True)
    pose = result.screenshot_to_clean*prepared.registration_scale
    pose[:,2] -= prepared.processed.crop_offset
    pose[:,2] += pose[:,:2]@np.float32([x,y])
    seeds = [pose]
    if result.match is not None:
        seeds.extend(np.float32([[c.scale,0,c.top_left[0]],[0,c.scale,c.top_left[1]]])
                     for c in result.match.result.top_k_candidates)
        factor = prepared.config['cpu_pyramid_factor']
        seeds.extend(np.float32([[c.scale,0,c.top_left[0]/factor],[0,c.scale,c.top_left[1]/factor]])
                     for c in result.match.coarse_result.top_k_candidates)
    for branch in result.validation.get('credible_geometry',{}).get('branches',[]):
        anchor = np.float32(branch['anchor']); scale = branch['scale']
        translation = anchor-scale*evidence.center
        seeds.append(np.float32([[scale,0,translation[0]],[0,scale,translation[1]]]))
    proposal,details = review_geometry(prepared.processed,evidence,seeds,
        scale_min=bounds[0],scale_max=bounds[1],viewport=image.shape[:2],origin=(x,y),
        allow_corridor_structure=prepared.config.get('corridor_structure_enabled', True))
    validation = {**result.validation,'reliable_geometry':details,
                  'position_status':details['position_status'],'scale_status':details['scale_status']}
    reviewed = replace(result,validation=validation,rejection_reasons=list(result.rejection_reasons),timings=dict(result.timings))
    reviewed.timings['reliable_geometry'] = (perf_counter()-started)*1000
    # Position and scale are separate; use the actual latest evidence in messages.
    if details.get('position_status')=='confirmed' and not result.accepted:
        reviewed.rejection_reasons = [r for r in reviewed.rejection_reasons if not any(t in r for t in
            ('位置证据尚未确认','多个可能位置','不同缩放比例','缩放精度仍不足'))]
        reviewed.rejection_reasons.append('位置已确认，但缩放稳定性尚未通过')
    local = bool(details.get('local_accepted') and not details.get('accepted')
                 and prepared.config.get('local_projection_enabled',True))
    if proposal is None or not (details.get('accepted') or local):
        return reviewed
    matrix = proposal.copy(); matrix[:,2] -= matrix[:,:2]@np.float32([x,y])
    check = runtime._validation(matrix,segmentation,prepared.processed)
    coverage = boundary_coverage(matrix,segmentation,prepared.processed)
    validation['recovery_proposal_validation'] = {**check,'boundary_coverage':coverage}
    # Inner semantics and boundary coverage must agree; tolerance cannot pardon
    # a different room or a transform mapping structure outside the template.
    if (check['mapped_structure_fraction']<.95 or check['structure_precision']<.90
            or coverage<.97 or details['semantic_precision']<.98):
        return reviewed
    clean = matrix.copy(); clean[:,2] += prepared.processed.crop_offset
    clean /= prepared.registration_scale
    validation.update(check)
    validation.update(position_status='confirmed',scale_status='local' if local else 'confirmed',
        geometry_acceptance_policy='reliable_wall_groups_and_projection_scope',
        projection_scope='local' if local else 'full',local_projection_available=local,
        credible_geometry_confirmed=True,sparse_refinement_confirmed=True,
        acceptance_basis='reliable_wall_groups',reliable_geometry=details,
        raw_mask_validation=check,semantic_precision=details['semantic_precision'],
        edge_median_error_px=float(details['wall_median_screen_px']*matrix[0,0]),
        edge_median_screen_px=details['wall_median_screen_px'],
        edge_error_basis='reliable_wall_groups_in_processed_template_pixels')
    for key in ('sparse_spatial_rejected','sparse_scale_rejected','regional_pose_rejected',
                'scale_only_relaxed','soft_pose_reuse','history_background_soft_override',
                'route_corroborated','pre_route_rejection_reasons'):
        validation.pop(key,None)
    corridor,room = evidence.trusted_masks
    trusted = replace(observation,corridor_mask=corridor,room_mask=room,map_mask=corridor|room,
        known_background_mask=evidence.background_mask,valid_mask=corridor|room|evidence.background_mask,
        unknown_mask=cv2.bitwise_not(corridor|room|evidence.background_mask))
    matched = result.match
    if matched is not None:
        matched = replace(matched,result=replace(matched.result,best_scale=float(proposal[0,0]),
            best_position=tuple(float(v) for v in proposal[:,2])))
    return replace(reviewed,match=matched,accepted=not local,rejection_reasons=['位置已确认，等待路线复核'] if local else [],
        screenshot_to_clean=clean,clean_to_screen=cv2.invertAffineTransform(clean),
        alignment_preview=runtime._alignment_preview(image,matrix,segmentation,prepared.processed),
        observation_segmentation=trusted)


def recover(image,prepared,result,once,prior=None,memory=None,soft=None):
    started = perf_counter()
    initial_ms = result.timings.get('total',0)
    if (not prepared.config.get('registration_recovery_enabled',True)
            or not prepared.config.get('reliable_wall_recovery_enabled',True)):
        return result
    if result.accepted or result.screenshot_to_clean is None:
        return result
    # Never override actual temporal contradictions with a fresh local fit.
    if result.validation.get('history_conflict_rate',0) is not None and result.validation.get('history_conflict_rate',0)>.20:
        return result
    bounds = result.validation.get('search_scale_bounds')
    if bounds is None:
        return result
    interval = boundary_interval(result.screenshot_to_clean[0,0]*prepared.registration_scale,bounds)
    touched_boundary = interval is not None
    original = {'reasons':list(result.rejection_reasons),
                'screenshot_to_clean':result.screenshot_to_clean.tolist(),
                'structure_precision':result.validation.get('structure_precision')}
    trace = []
    # Filtering can reorder the leading coarse candidate onto a scale bound,
    # although another retained seed already supports a reliable pose. Review
    # those seeds before paying for an expanded scale search.
    if interval is not None and result.validation.get('fog_filter', {}).get('uncertain_pixels'):
        reviewed = review_result(image, prepared, result, bounds)
        if reviewed.accepted or reviewed.validation.get('local_projection_available'):
            result = reviewed
            interval = None
            result.validation['fog_filter']['scale_expansion_avoided'] = True
    if interval is not None and prepared.config.get('scale_boundary_recovery_enabled',True):
        candidate = once(image,prepared,None,None,None,_scale_bounds=interval,
                         _source_segmentation=result.source_segmentation)
        trace.append({'bounds':list(interval),'accepted_before_geometry':candidate.accepted,
                      'validation':dict(candidate.validation),'elapsed_ms':candidate.timings.get('total',0)})
        # Newly searched scales require independent geometry, even when the old
        # area validator accepts a smaller accidental room.
        candidate = replace(candidate,accepted=False,rejection_reasons=['扩展尺度候选待几何复核'])
        candidate = review_result(image,prepared,candidate,interval)
        if not (candidate.accepted or candidate.validation.get('local_projection_available')):
            candidate.validation['reliable_recovery_rejected'] = True
        if candidate.accepted or candidate.validation.get('local_projection_available'):
            result = candidate
        elif candidate.validation.get('structure_precision',0)>result.validation.get('structure_precision',0):
            result = candidate
    if (not result.accepted and not result.validation.get('local_projection_available')
            and prepared.config.get('reliable_wall_recovery_enabled',True)
            and 'reliable_geometry' not in result.validation):
        result = review_result(image,prepared,result,bounds)
    result.validation['scale_recovery_trace'] = trace
    result.validation['before_recovery'] = original
    result.validation['initial_search_scale_bounds'] = list(bounds)
    result.validation['search_touched_boundary'] = touched_boundary
    result.timings['recovery'] = (perf_counter()-started)*1000
    result.timings['total'] = initial_ms+result.timings['recovery']
    return result


def recover_unfiltered_search(image, prepared, result, once, prior=None, memory=None, soft=None):
    """One compatibility fallback if fog filtering loses the viable candidate.

    Successful filtered searches and trusted-history tracking never rerun.
    Reuse this frame's segmentation and template caches; never lower gates.
    """
    fog = result.validation.get('fog_filter', {})
    if (not fog.get('enabled') or not fog.get('uncertain_pixels')
            or result.accepted or result.validation.get('local_projection_available')
            or prior is not None or soft is not None or (memory is not None and memory.updates)
            or (result.validation.get('history_conflict_rate') or 0) > .20):
        return result
    started = perf_counter()
    raw_prepared = replace(prepared, config={**prepared.config, 'corridor_fog_filter_enabled': False})
    candidate = once(image, raw_prepared, _source_segmentation=result.source_segmentation)
    candidate = recover(image, raw_prepared, candidate, once)
    use_raw = bool(candidate.accepted or candidate.validation.get('local_projection_available'))
    selected = candidate if use_raw else result
    selected.validation['fog_filter'] = {**fog, 'raw_fallback_attempted': True,
        'raw_fallback_selected': use_raw, 'filtered_reasons': list(result.rejection_reasons),
        'raw_reasons': list(candidate.rejection_reasons)}
    selected.search_segmentation = result.search_segmentation
    elapsed = (perf_counter()-started)*1000
    selected.timings = {**selected.timings, 'fog_filter': result.timings.get('fog_filter', 0),
        'fog_raw_fallback': elapsed, 'total': result.timings.get('total', 0)+elapsed}
    return selected


def clip_local_layer(layer,registration):
    """Offline diagnostic of observed bounds and <=4 px candidate disagreement.

    Live projection displays the full verified asset, without this clipping.
    """
    if registration.validation.get('projection_scope')!='local':
        return layer
    details = registration.validation['reliable_geometry']
    x,y,w,h = registration.observation_bbox
    left,top,right,bottom = details['local_bounds']
    yy,xx = np.mgrid[0:h,0:w].astype(np.float32)
    points = np.column_stack((xx.ravel(),yy.ravel()))
    # Same local transform used to generate the uncertainty poses.
    # Store exact proposal translation rather than reconstructing the centroid.
    matrix = np.float32(details['selected_local_pose'])
    cached = getattr(registration,'_local_projection_mask',None)
    if cached is not None and cached.shape==layer.shape[:2]:
        output=layer.copy();output[:,:,3]=cv2.bitwise_and(output[:,:,3],cached)
        return output
    valid = (points[:,0]>=left)&(points[:,0]<=right)&(points[:,1]>=top)&(points[:,1]<=bottom)
    for pose in details['uncertainty_poses']:
        inverse = cv2.invertAffineTransform(np.float32(pose))
        linear = inverse[:,:2]@matrix[:,:2]-np.eye(2,dtype=np.float32)
        translation = inverse[:,:2]@matrix[:,2]+inverse[:,2]
        dx = linear[0,0]*points[:,0]+linear[0,1]*points[:,1]+translation[0]
        dy = linear[1,0]*points[:,0]+linear[1,1]*points[:,1]+translation[1]
        valid &= dx*dx+dy*dy<=16
    output = layer.copy()
    alpha = np.zeros(layer.shape[:2],np.uint8)
    alpha[y:y+h,x:x+w] = valid.reshape(h,w).astype(np.uint8)*255
    registration._local_projection_mask = alpha
    output[:,:,3] = cv2.bitwise_and(output[:,:,3],alpha)
    return output
