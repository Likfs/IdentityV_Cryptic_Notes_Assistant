"""Geometry-only proposals -> shared evidence -> uniform 3-DOF review.

Only screenshot evidence supplies positives/negatives. Proposal correlation is
not an acceptance score. No CNN, GUI, session memory or production mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import cv2
import numpy as np

from src.registration.multiscale_structure.gpu.registration.runtime_registration import (
    runtime_registration_config, _crop_segmentation, _validation, RuntimeRegistrationResult,
)
from src.registration.multiscale_structure.gpu.core.color_segmenter import segment_map, add_known_background_evidence
from src.registration.multiscale_structure.gpu.core.fog_evidence import filter_corridor_fog
from src.registration.multiscale_structure.gpu.core.roi_extractor import component_content_bbox
from src.registration.multiscale_structure.gpu.core.sparse_pose_refiner import build_evidence
from src.registration.multiscale_structure.gpu.core.cuda_scoring import score_poses, guarded_argmax
from src.registration.multiscale_structure.gpu.core.credible_geometry import screenshot_evidence, review_geometry
from .template import Template


@dataclass(frozen=True)
class Options:
    factor: float = .125
    coarse_scale_step: float = .04
    peaks_per_scale: int = 4
    max_seeds: int = 18
    # Bounds are screenshot -> processed template, intentionally cover boundary
    # recovery in the same search, rather than rerun several acceptance paths.
    scale_min: float = .25
    scale_max: float = 2.75
    search_workers: int = 4

    def __post_init__(self):
        if self.search_workers not in (1, 2, 4):
            raise ValueError("search_workers must be 1, 2 or 4")
        if self.factor != .125 or not 0 < self.scale_min < self.scale_max:
            raise ValueError("invalid registration options")
        if self.coarse_scale_step <= 0 or self.max_seeds < 2 or self.peaks_per_scale < 2:
            raise ValueError("must retain competing proposals")


@dataclass
class Frame:
    image: np.ndarray
    bbox: tuple
    observation_image: np.ndarray
    observation: object
    search: object
    sparse: object
    walls: object
    preparation_ms: float


@lru_cache(maxsize=8)
def _scale_grid(options):
    scales = np.linspace(options.scale_min, options.scale_max,
                         int(np.ceil((options.scale_max-options.scale_min)/options.coarse_scale_step))+1)
    scales.flags.writeable = False
    return scales


_REFINE_OFFSETS = np.float32([(x,y,s) for x in (-1,0,1) for y in (-1,0,1) for s in (-1,0,1)])
_REFINE_OFFSETS.flags.writeable = False

# Two independent evidence branches; validated separately against serial mode.
PARALLEL_FRAME_PREPARATION = True


@lru_cache(maxsize=1)
def _frame_executor():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix='frame-evidence')


@lru_cache(maxsize=2)
def _search_executor(workers):
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix='scale-search')


def prepare_frame(image, *, parallel=None) -> Frame:
    started = perf_counter()
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("截图必须为 uint8 BGR")
    cfg = runtime_registration_config()
    segmentation = segment_map(image, cfg, infer_known_background=False, include_stairs=False)
    bbox = component_content_bbox(segmentation.map_mask, padding=10, min_area=300)
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w].copy()
    obs = add_known_background_evidence(roi, _crop_segmentation(segmentation, bbox), cfg)
    def search_evidence():
        search = filter_corridor_fog(image, segmentation, cfg)
        # Reuse only unchanged segmentation, never a different fog/ROI model.
        query = obs if search is segmentation else add_known_background_evidence(roi, _crop_segmentation(search, bbox), cfg)
        return query, build_evidence(query)
    use_parallel = PARALLEL_FRAME_PREPARATION if parallel is None else parallel
    if use_parallel:
        pending = _frame_executor().submit(search_evidence)
        try:
            walls = screenshot_evidence(roi, obs, reliable=True)
        except BaseException:
            # Drain the other branch before releasing this frame/template lock.
            # Never leave GPU work running after the caller enters CPU fallback.
            try:
                pending.result()
            except BaseException:
                pass
            raise
        query, sparse = pending.result()
    else:
        query, sparse = search_evidence()
        walls = screenshot_evidence(roi, obs, reliable=True)
    return Frame(image, bbox, roi, obs, query, sparse, walls, (perf_counter()-started)*1000)


def proposals(template: Template, frame: Frame, options: Options):
    """Positive-only recall plus background conflicts; unknown is neutral.

    Normalized correlation is an additional recall path (not proof of absence).
    Both paths cover the complete same scale grid before common refinement.
    """
    factor = options.factor
    query = np.stack((frame.search.corridor_mask, frame.search.room_mask), axis=2).astype(np.float32)/255
    query = cv2.resize(query, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    bg = frame.search.known_background_mask
    bg = cv2.resize(bg.astype(np.float32)/255, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)
    padding = 8
    search_arrays = getattr(template, '_proposal_arrays', None)
    if search_arrays is None:
        channels = template.coarse_channels
        dense = np.dstack((channels, channels.max(axis=2)))
        dense = cv2.copyMakeBorder(dense, padding,padding,padding,padding,cv2.BORDER_CONSTANT, value=(0,0,0))
        two_channels = dense[:, :, :2].copy()
        dense.flags.writeable = two_channels.flags.writeable = False
        search_arrays = dense, two_channels
        template._proposal_arrays = search_arrays
    dense, two_channels = search_arrays
    scales = _scale_grid(options)
    def one_scale(scale):
        found = []
        size = (max(1, round(query.shape[1]*scale)), max(1, round(query.shape[0]*scale)))
        if min(size) < 3 or size[0] > dense.shape[1] or size[1] > dense.shape[0]:
            return found
        patch = cv2.resize(query, size, interpolation=cv2.INTER_LINEAR)
        known = cv2.resize(bg, size, interpolation=cv2.INTER_LINEAR)
        count = float(patch.sum())
        if count < 4:
            return found
        known_sum = known.sum()
        penalty = .25*count/max(1., float(known_sum)) if known_sum >= 6 else 0.
        response = cv2.matchTemplate(dense, np.dstack((patch, -penalty*known)), cv2.TM_CCORR)/count
        normalized = cv2.matchTemplate(two_channels, patch, cv2.TM_CCORR_NORMED)
        for kind, scores in (("positive", response), ("normalized", normalized)):
            for _ in range(options.peaks_per_scale):
                _, score, _, point = cv2.minMaxLoc(scores)
                if not np.isfinite(score) or score < .35:
                    break
                found.append((float(score), (np.float32(point)-padding)/factor, float(scale), kind))
                cv2.circle(scores, point, 5, -1., -1)
        return found
    # Ordered map preserves scale order and stable tie breaking, irrespective
    # of worker completion order. Each response/mutable peak mask is private.
    batches = (map(one_scale, scales) if options.search_workers == 1
               else _search_executor(options.search_workers).map(one_scale, scales))
    found = [candidate for batch in batches for candidate in batch]
    # Reserve proposals from both recall objectives; ranking will be common.
    seeds = []
    for kind in ("positive", "normalized"):
        selected = []
        for score, xy, scale, _ in sorted((f for f in found if f[3] == kind), key=lambda f: -f[0]):
            anchor = xy + scale*frame.sparse.center
            if any(np.linalg.norm(anchor-p[:2])/scale < 18 and abs(np.log(scale/p[2])) < .07 for p in selected):
                continue
            selected.append(np.float32([*anchor, scale]))
            if len(selected) >= options.max_seeds:
                break
        seeds.extend(selected)
    if not seeds:
        return np.empty((0,3), np.float32), {"raw_count": len(found), "scale_count": len(scales)}
    poses = np.asarray(seeds)
    scores = score_poses(template.processed, frame.sparse, poses, include_hit=False)[0]
    # Keep spatial/scale modes after score normalization; never accept a coarse hit.
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        p = poses[index]
        if any(np.linalg.norm(p[:2]-q[:2])/p[2] < 16 and abs(np.log(p[2]/q[2])) < .05 for q in selected):
            continue
        selected.append(p)
        if len(selected) >= options.max_seeds:
            break
    return np.asarray(selected), {"raw_count": len(found), "scale_count": len(scales), "seed_count":len(selected)}


def refine(template, evidence, poses, options):
    """All proposal batches reuse the same stratified point evidence."""
    initial = poses.copy()
    offsets = _REFINE_OFFSETS
    lower = initial - np.float32([24,24,.08])
    upper = initial + np.float32([24,24,.08])
    lower[:,2] = np.maximum(lower[:,2], options.scale_min)
    upper[:,2] = np.minimum(upper[:,2], options.scale_max)
    for shift, zoom in ((8,.024),(4,.012),(2,.006),(1,.003)):
        for _ in range(2):
            trials = np.clip(poses[:,None] + offsets[None]*np.float32([shift,shift,zoom]),lower[:,None],upper[:,None])
            scores = score_poses(template, evidence, trials.reshape(-1,3), include_hit=False)[0].reshape(len(poses),-1)
            updated = trials[np.arange(len(poses)), guarded_argmax(template, evidence, trials, scores)]
            if np.max(np.abs(updated-poses)) < 1e-4:
                break
            poses = updated
    return poses


def local_to_clean(matrix, template, bbox):
    """Observation -> processed becomes full screenshot -> original clean."""
    out = np.asarray(matrix, np.float64).copy()
    out[:,2] -= out[:,:2] @ np.float64(bbox[:2])
    out[:,2] += template.processed.crop_offset
    return out / template.registration_scale


def register(image, template: Template, *, options=None, frame=None):
    options = options or Options()
    started = perf_counter()
    own_frame = frame is None
    if frame is not None and frame.image is not image:
        raise ValueError("共享证据必须属于当前截图对象，禁止复用上一帧证据")
    frame = frame or prepare_frame(image)
    timings = {"frame": frame.preparation_ms if own_frame else 0., "method": "lab_uniform_geometry"}
    reasons, details = [], {}
    matrix = None
    if len(frame.sparse.edges) < 40 or len(frame.walls.walls) < 4:
        reasons.append("insufficient_independent_walls")
    else:
        tick = perf_counter()
        seeds, details["proposals"] = proposals(template, frame, options)
        timings["proposals"] = (perf_counter()-tick)*1000
        if not len(seeds):
            reasons.append("no_geometry_proposals")
        else:
            tick = perf_counter()
            poses = refine(template.processed, frame.sparse, seeds, options)
            timings["point_refinement"] = (perf_counter()-tick)*1000
            matrices = [np.float32([[p[2],0,p[0]-p[2]*frame.sparse.center[0]],
                                    [0,p[2],p[1]-p[2]*frame.sparse.center[1]]]) for p in poses]
            tick = perf_counter()
            local, geometry = review_geometry(template.processed, frame.walls, matrices,
                scale_min=options.scale_min, scale_max=options.scale_max,
                viewport=image.shape[:2], origin=np.float32(frame.bbox[:2]), allow_corridor_structure=True)
            timings["geometry"] = (perf_counter()-tick)*1000
            details["geometry"] = geometry
            if local is not None:
                matrix = local_to_clean(local, template, frame.bbox)
                # Full raster corroboration uses the original observation, not
                # just points retained by wall extraction or fog filtering.
                raster = _validation(local, frame.observation, template.processed)
                details["raster"] = raster
                if (raster.get("mapped_structure_fraction",0) < .95 or
                    raster.get("structure_precision",0) < .90 or
                    raster.get("semantic_precision",0) < .82 or
                    raster.get("edge_median_error_px") is None or
                    raster["edge_median_error_px"]/float(local[0,0]) > 5):
                    reasons.append("full_observation_conflict")
            if not geometry.get("accepted", False):
                reasons.append("geometry_not_unique_or_full_view_unstable")
            if local is not None and (local[0,0] <= options.scale_min*1.01 or local[0,0] >= options.scale_max*.99):
                reasons.append("scale_search_boundary")
    timings["total"] = (perf_counter()-started)*1000
    inverse = cv2.invertAffineTransform(matrix) if matrix is not None else None
    return RuntimeRegistrationResult(accepted=not reasons and matrix is not None,
        rejection_reasons=reasons, clean_to_screen=inverse, screenshot_to_clean=matrix,
        observation_bbox=frame.bbox, match=None, validation=details, timings=timings,
        observation_image=frame.observation_image, observation_mask=frame.observation.map_mask,
        alignment_preview=None, observation_segmentation=frame.observation)


def register_candidates(image, templates, *, options=None):
    """Optional CNN-free geometry comparison; no top-K layout exclusion."""
    frame = prepare_frame(image)
    results = []
    winner = result = None
    winner_key = None
    for template in templates:
        current = register(image,template,options=options,frame=frame)
        score = current.validation.get("geometry",{}).get("score",-1)
        key = (-score,template.layout)
        if winner_key is None or key < winner_key:
            winner,result,winner_key = template,current,key
        results.append((template.layout,current))
    if result is None:
        raise ValueError("empty template collection")
    results.sort(key=lambda item: (-item[1].validation.get("geometry",{}).get("score",-1), item[0]))
    best = result.validation.get("geometry",{}).get("score",-1)
    competitors = [layout for layout,r in results[1:]
        if r.validation.get("geometry",{}).get("score",-1) >= best-.06]
    if competitors:
        result.accepted = False
        result.rejection_reasons.append("cross_layout_ambiguity")
    return winner, result, {"frame_ms":frame.preparation_ms, "competing_layouts":competitors,
        "results":[{"layout":layout,"accepted":r.accepted,"diagnostics":r.diagnostics()} for layout,r in results]}
