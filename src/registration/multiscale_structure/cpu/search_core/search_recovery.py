"""Preserve base_search results, recover refusals with screenshot-owned semantic cores.

The recovery changes measurement support, never acceptance thresholds. No
template or candidate participates in selecting screenshot semantic pixels.
All original search modes, walls and full-raster checks remain in use.
"""
from dataclasses import replace
from time import perf_counter
import cv2
import numpy as np

from src.registration_core.credible_geometry import review_geometry
from src.registration_core.sparse_pose_refiner import balanced_points
from src.registration.runtime_registration import RuntimeRegistrationResult, _validation
from . import search as base_search


def interior_evidence(evidence):
    # Existing trusted masks already exclude a 2px color fringe. Add 3px to
    # exclude wall strokes / antialiasing and material-transition uncertainty.
    masks = tuple(cv2.erode(m, np.ones((7,7),np.uint8), borderType=cv2.BORDER_CONSTANT,
                           borderValue=0) for m in evidence.trusted_masks)
    points, labels = [], []
    for label, mask in enumerate(masks):
        selected, _ = balanced_points(mask, 300)
        points.append(selected)
        labels.extend([label] * len(selected))
    return replace(evidence, semantic=np.concatenate(points), labels=np.asarray(labels), trusted_masks=masks,
        diagnostics={**evidence.diagnostics, 'semantic_support': 'screenshot_interior_extra_3px',
                     'trusted_semantic_pixels': sum(cv2.countNonZero(m) for m in masks)})


def raster_failures(raster, scale):
    checks = {
        'raster_outside_template': raster.get('mapped_structure_fraction',0) < .95,
        'raster_structure_conflict': raster.get('structure_precision',0) < .90,
        'raster_material_conflict': raster.get('semantic_precision',0) < .82,
        'raster_edge_conflict': raster.get('edge_median_error_px') is None or
            raster['edge_median_error_px'] / float(scale) > 5,
    }
    return [name for name, failed in checks.items() if failed]


def first_pass(image, template, options, frame):
    """base_search search/review with retained seeds, avoiding a second proposal search."""
    reasons, details, matrices = [], {}, []
    matrix = None
    if len(frame.sparse.edges) < 40 or len(frame.walls.walls) < 4:
        reasons.append('insufficient_independent_walls')
    else:
        seeds, details['proposals'] = base_search.proposals(template,frame,options)
        if not len(seeds):
            reasons.append('no_geometry_proposals')
        else:
            poses = base_search.refine(template.processed,frame.sparse,seeds,options)
            matrices = [np.float32([[p[2],0,p[0]-p[2]*frame.sparse.center[0]],
                                    [0,p[2],p[1]-p[2]*frame.sparse.center[1]]]) for p in poses]
            local, geometry = review_geometry(template.processed,frame.walls,matrices,
                scale_min=options.scale_min,scale_max=options.scale_max,
                viewport=image.shape[:2],origin=np.float32(frame.bbox[:2]),allow_corridor_structure=True)
            details['geometry'] = geometry
            if local is not None:
                matrix = base_search.local_to_clean(local,template,frame.bbox)
                raster = _validation(local,frame.observation,template.processed)
                details['raster'] = raster
                if raster_failures(raster,local[0,0]):
                    reasons.append('full_observation_conflict')
            if not geometry.get('accepted',False):
                reasons.append('geometry_not_unique_or_full_view_unstable')
            if local is not None and (local[0,0] <= options.scale_min*1.01 or local[0,0] >= options.scale_max*.99):
                reasons.append('scale_search_boundary')
    result = RuntimeRegistrationResult(accepted=not reasons and matrix is not None,rejection_reasons=reasons,
        clean_to_screen=cv2.invertAffineTransform(matrix) if matrix is not None else None,
        screenshot_to_clean=matrix,observation_bbox=frame.bbox,match=None,validation=details,timings={},
        observation_image=frame.observation_image,observation_mask=frame.observation.map_mask,
        observation_segmentation=frame.observation,alignment_preview=None)
    return result, matrices


def register(image, template, *, options=None, frame=None):
    started = perf_counter()
    options = options or base_search.Options()
    frame = frame or base_search.prepare_frame(image)
    if frame.image is not image:
        raise ValueError('Evidence must belong to this image')
    # Reuse base_search code and preserve its already-accepted pose bit-for-bit.
    baseline, matrices = first_pass(image, template, options, frame)
    if baseline.accepted or baseline.clean_to_screen is None:
        baseline.validation['interior_recovery'] = {'attempted': False, 'basis': 'search_accepted' if baseline.accepted else 'no_pose'}
        baseline.timings['total'] = (perf_counter()-started)*1000
        return baseline
    walls = interior_evidence(frame.walls)
    if len(walls.semantic) < 40:
        baseline.validation['interior_recovery'] = {'attempted': False, 'basis': 'insufficient_interior_pixels'}
        baseline.timings['total'] = (perf_counter()-started)*1000
        return baseline
    local, geometry = review_geometry(template.processed, walls, matrices,
        scale_min=options.scale_min,scale_max=options.scale_max,
        viewport=image.shape[:2],origin=np.float32(frame.bbox[:2]),allow_corridor_structure=True)
    reasons = []
    if not geometry.get('accepted'):
        reasons.append('interior_geometry_not_confirmed')
    raster = None
    if local is not None:
        raster = _validation(local,frame.observation,template.processed)
        reasons.extend(raster_failures(raster,local[0,0]))
        if local[0,0] <= options.scale_min*1.01 or local[0,0] >= options.scale_max*.99:
            reasons.append('scale_search_boundary')
    else:
        reasons.append('no_recovery_pose')
    attempt = {'attempted':True,'geometry':geometry,'raster':raster,'reasons':reasons,
               'baseline_reasons':baseline.rejection_reasons,'semantic_points':len(walls.semantic)}
    if reasons:
        baseline.validation['interior_recovery'] = attempt
        baseline.timings['total'] = (perf_counter()-started)*1000
        return baseline
    matrix = base_search.local_to_clean(local,template,frame.bbox)
    return RuntimeRegistrationResult(accepted=True,rejection_reasons=[],
        clean_to_screen=cv2.invertAffineTransform(matrix),screenshot_to_clean=matrix,
        observation_bbox=frame.bbox,match=None,
        validation={'geometry':geometry,'raster':raster,'proposals':baseline.validation['proposals'],'interior_recovery':attempt},
        timings={'method':'screenshot_interior_recovery','total':(perf_counter()-started)*1000},
        observation_image=frame.observation_image,observation_mask=frame.observation.map_mask,
        observation_segmentation=frame.observation,alignment_preview=None)
