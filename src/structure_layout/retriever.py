"""Thread-safe CPU index and manual-confirmation layout candidates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
import json
import threading
import time

import cv2
import numpy as np

from src.layout_retrieval import LayoutCandidate, LayoutRetrievalResult
from src.recognition_control import check_cancelled, recognition_lock
from .features import target_wall_edges, PARAMETERS as FEATURE_PARAMETERS
from .local import attach_raw, evidence, review, PARAMETERS as LOCAL_PARAMETERS
from .search import (search, merge, R, SIGMA, PENALTY, MINIMUM_SCALES,
                     prepare_response_cache, RESPONSE_CACHE_POLICY)
from .scales import read_scale_overrides, matched_scale_mode
from .normalized import read_normalized_index, original_pose, COARSE_RESIZE

PARAMETERS = {
    'algorithm_version': 'minimum_structure_cpu_2',
    'scale_policy': 'per_source_verified_override_or_legacy_minimum',
    'rotation_degrees': [0], 'coarse_scales': list(MINIMUM_SCALES),
    'pose_refinement': False, 'execution_backend': 'cpu',
    'correlation_workspace': 'request_worker_two_float32_v1',
    'coarse_resize': R, 'wall_distance_screen_px': SIGMA,
    'baseline_extra_weight': PENALTY, 'features': FEATURE_PARAMETERS,
    'local_review': LOCAL_PARAMETERS, 'score_is_probability': False,
    'coordinate_space': 'normalized_capture_roi_to_original_clean',
}


def prepare_target(image, layout, floor, pixel_scale=1., coarse_resize=R):
    edges = target_wall_edges(image, pixel_scale)
    full, dist, points, low = [], [], [], []
    for channel in edges:
        full.append(cv2.distanceTransform(1-channel, cv2.DIST_L2, 5))
        y, x = np.where(channel)
        points.append(np.column_stack((x, y)).astype(np.float32))
        mask = cv2.resize(channel.astype(np.float32), None, fx=coarse_resize, fy=coarse_resize,
                          interpolation=cv2.INTER_AREA)
        low.append(mask)
        dist.append(cv2.distanceTransform((mask < .1).astype(np.uint8), cv2.DIST_L2, 5))
    return attach_raw(dict(layout=layout, floor=floor, dist=dist, full=full, coarse_resize=coarse_resize,
                           points=points, all_points=np.concatenate(points), low=sum(low)), image, pixel_scale)


def review_candidate(candidate, lookup, query, box, cancelled):
    check_cancelled(cancelled)
    details = review(lookup[candidate['layout'], candidate['floor']], candidate, query, box)
    check_cancelled(cancelled)
    return dict(candidate, **details, score=candidate['score']-details['local_penalty'])


class PureStructureRetriever:
    """Profile-local indexes; fixed resources are validated before first publication.

    No icon resources or disk feature cache are required. The lock serializes
    preparation and recognition. Maintenance callers retain boundary validation;
    fixed-resource callers reuse the complete in-memory index until close().
    """
    def __init__(self, clean_root, cache_root=None, *, workers=None, floors=('2nd', '1st', 'b1'), index_root=None,
                 fixed_resources=False):
        self.clean_root = Path(clean_root)
        self.index_root = index_root
        self.fixed_resources = bool(fixed_resources)
        self.floors = tuple(floors)
        if not self.floors or len(set(self.floors)) != len(self.floors) or any(f not in ('2nd', '1st', 'b1') for f in self.floors):
            raise ValueError('识别楼层配置无效')
        self._lock = threading.RLock()
        self._fingerprint = ''
        self._targets = []
        self.workers = min(8, max(1, os.cpu_count() or 1)) if workers is None else min(8, max(1, int(workers)))
        self._executor = None

    def close(self):
        """Release workers after any active recognition finishes."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            self._targets = []
            self._fingerprint = ''

    def prepare(self, *, cancelled=None):
        with recognition_lock(self._lock, cancelled):
            if self.fixed_resources and self._fingerprint:
                return self._fingerprint
            paths = [p for p in sorted(self.clean_root.glob('layout_*/*.png'))
                     if p.stem.rsplit('_', 1)[-1] in self.floors]
            if not paths:
                raise ValueError('缺少纯结构识别底图')
            content = {}
            resolved = {p: p.resolve() for p in paths} if self.fixed_resources else {}
            for p in paths:
                check_cancelled(cancelled)
                floor = p.stem.rsplit('_', 1)[-1]
                if p.name != f'{p.parent.name}_{floor}.png':
                    raise ValueError('识别底图的布局目录与文件名不一致')
                content[p] = p.read_bytes()
            digests = {p: hashlib.sha256(data).hexdigest() for p, data in content.items()}
            # Decode the same bytes that were validated. Fixed mode retains
            # their derived index, not another persistent copy of the PNG data.
            source_hashes = {p.resolve():digest for p,digest in digests.items()}
            snapshot = {p.relative_to(self.clean_root).as_posix():(data,digests[p]) for p,data in content.items()}
            overrides = read_scale_overrides(self.clean_root, source_hashes=source_hashes)
            normalized, index_images = read_normalized_index(self.clean_root, overrides, self.index_root,
                                                            source_snapshot=snapshot)
            for p in paths:
                record = overrides.get(p.relative_to(self.clean_root).as_posix())
                if record and record['source_sha256'] != digests[p]:
                    raise ValueError(f'识别尺度对应底图已改变，请重新核对：{p.name}')
            identity = dict(parameters=PARAMETERS, opencv=cv2.__version__,
                            scale_overrides=overrides,
                            normalized_index=normalized,
                            normalized_features='source_pixel_lengths_v1' if normalized else None,
                            normalized_coarse_resize=COARSE_RESIZE if normalized else None,
                            response_cache_policy=RESPONSE_CACHE_POLICY if normalized else None,
                            floors=self.floors,
                            root=str(self.clean_root.resolve()),
                            sources=[(p.relative_to(self.clean_root).as_posix(), digests[p]) for p in paths])
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if fingerprint == self._fingerprint:
                return fingerprint
            def build_target(p):
                check_cancelled(cancelled)
                name = p.relative_to(self.clean_root).as_posix()
                image = cv2.imdecode(np.frombuffer(index_images.get(name, content[p]), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'无法读取识别资源：{p.name}')
                extra = {'pixel_scale':normalized['targets'][name]['original_to_index'][0][0],
                         'coarse_resize':COARSE_RESIZE} if normalized else {}
                target = prepare_target(image, p.parent.name, p.stem.rsplit('_', 1)[-1], **extra)
                target.update(path=p.relative_to(self.clean_root).as_posix(), sha256=digests[p])
                record = overrides.get(target['path'])
                target['scales'] = tuple(float(s) for s in record['scales']) if record else MINIMUM_SCALES
                target['scale_modes'] = record.get('scale_modes', {}) if record else {}
                target['scale_provenance'] = 'verified_source_override' if record else 'legacy_minimum'
                if normalized:
                    target['source_scales'] = target['scales']
                    target['scales'] = tuple(normalized['scale_modes'].values())
                    target['scale_modes'] = normalized['scale_modes']
                    target['original_to_index'] = normalized['targets'][name]['original_to_index']
                check_cancelled(cancelled)
                return target

            if self.workers == 1:
                targets = [build_target(p) for p in paths]
            else:
                # Full-resolution intermediates are large: allow only two builds
                # at once, then publish one complete snapshot in source order.
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix='layout-index') as executor:
                    futures = [executor.submit(build_target, p) for p in paths]
                    try:
                        targets = [future.result() for future in futures]
                    finally:
                        for future in futures:
                            future.cancel()
            if normalized:
                prepare_response_cache(targets, cancelled=cancelled)
            if self.fixed_resources:
                # Recheck the complete batch after building, before publishing.
                # A changed source or link must never publish a mixed index.
                sources = {p: (resolved[p], digests[p]) for p in paths}
                self._verify_resources(overrides, normalized, sources=sources, cancelled=cancelled)
            check_cancelled(cancelled)
            self._targets, self._fingerprint = targets, fingerprint
            self._scale_overrides = overrides
            self._normalized_index = normalized
            return fingerprint

    def _verify_resources(self, overrides, normalized, *, sources=None, cancelled=None):
        snapshot = {}
        source_hashes = {}
        if sources is not None:
            paths = [p for p in sorted(self.clean_root.glob('layout_*/*.png'))
                     if p.stem.rsplit('_', 1)[-1] in self.floors]
            if set(paths) != set(sources):
                raise ValueError('识别索引准备期间底图清单已改变，请重新识别')
        else:
            paths = [self.clean_root/name for name in normalized.get('targets', {})]
        for path in paths:
            check_cancelled(cancelled)
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            resolved = path.resolve()
            if sources is not None and (resolved, digest) != sources[path]:
                raise ValueError('识别索引准备期间底图或目录链接已改变，请重新识别')
            snapshot[path.relative_to(self.clean_root).as_posix()] = (content, digest)
            source_hashes[resolved] = digest
        if read_scale_overrides(self.clean_root, source_hashes=source_hashes) != overrides:
            raise ValueError('识别期间尺度配置已改变，请重新识别')
        if read_normalized_index(self.clean_root, overrides, self.index_root,
                                 source_snapshot=snapshot)[0] != normalized:
            raise ValueError('识别期间小图配置已改变，请重新识别')

    def recognize(self, screenshot_bgr, reference_size=(1180, 1050), *, cancelled=None):
        check_cancelled(cancelled)
        started = time.perf_counter()
        result = LayoutRetrievalResult(backend='pure_structure',
            parameters=dict(PARAMETERS, workers=self.workers), reference_size=list(reference_size))
        try:
            if (not isinstance(screenshot_bgr, np.ndarray) or screenshot_bgr.dtype != np.uint8 or
                    screenshot_bgr.ndim != 3 or screenshot_bgr.shape[2] != 3 or not screenshot_bgr.size):
                raise ValueError('截图必须为非空 BGR 三通道 uint8 图片')
            result.input_size = [screenshot_bgr.shape[1], screenshot_bgr.shape[0]]
            width, height = map(int, reference_size)
            if width < 50 or height < 50:
                raise ValueError('参考截图尺寸不足以识别结构')
            image = screenshot_bgr if result.input_size == [width, height] else cv2.resize(
                screenshot_bgr, (width, height), interpolation=cv2.INTER_AREA
                if width < screenshot_bgr.shape[1] else cv2.INTER_LINEAR)
            with recognition_lock(self._lock, cancelled):
                preparation = time.perf_counter()
                result.resource_fingerprint = self.prepare(**({'cancelled': cancelled} if cancelled is not None else {}))
                if self._normalized_index:
                    result.parameters.update(coarse_resize=COARSE_RESIZE,
                        target_features='source_pixel_lengths_v1', recognition_index='normalized_two_scales_v1')
                    result.parameters['response_cache'] = (RESPONSE_CACHE_POLICY['version']
                        if all('response_maps' in t for t in self._targets) else 'budget_fallback')
                result.timings_ms['index_prepare'] = (time.perf_counter()-preparation)*1000
                searching = time.perf_counter()
                if self.workers > 1 and self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                                        thread_name_prefix='layout-structure')
                raw = search(image, self._targets, executor=self._executor,
                             **({'cancelled': cancelled} if cancelled is not None else {}))
                if not self.fixed_resources:
                    self._verify_resources(self._scale_overrides, self._normalized_index, cancelled=cancelled)
                candidates = []
                if raw.get('floor_candidates'):
                    query = evidence(image, raw['query_edges'])
                    lookup = {(t['layout'], t['floor']): t for t in self._targets}
                    # Reuse the same bounded pool after coarse search; snapshots are read-only.
                    futures = ([] if self._executor is None else [self._executor.submit(
                        review_candidate, c, lookup, query, raw['box'], cancelled) for c in raw['floor_candidates']])
                    results = (map(lambda c: review_candidate(c, lookup, query, raw['box'], cancelled),
                                   raw['floor_candidates']) if self._executor is None else (f.result() for f in futures))
                    try:
                        for candidate in results:
                            check_cancelled(cancelled)
                            candidates.append(candidate)
                    finally:
                        for future in futures:
                            future.cancel()
                    for rank, fit in enumerate(merge(candidates), 1):
                        if not np.isfinite(fit['score']):
                            continue
                        target = lookup[fit['layout'], fit['floor']]
                        # Diagnostic search geometry is not a confirmed projection pose.
                        details = dict(original_pose(fit, target), box=raw['box'], rotation=0,
                                       recognition_zoom=matched_scale_mode(target['scale_modes'], fit['scale']),
                                       scale_provenance=target['scale_provenance'],
                                       configured_scales=list(target.get('source_scales', target['scales'])),
                                       index_scales=list(target['scales']),
                                       candidate_floor_only=True, target_count=len(self._targets),
                                       coordinate_space=PARAMETERS['coordinate_space'])
                        result.candidates.append(LayoutCandidate(
                            rank, fit['layout'], target['path'], target['sha256'], fit['score'],
                            fit['wall_support'], 0., 0., '纯结构候选，需人工确认', details))
                result.timings_ms['search'] = (time.perf_counter()-searching)*1000
                if not result.candidates:
                    result.reasons.append('当前截图没有足够的可匹配墙线')
        except (ValueError, OSError, cv2.error) as exc:
            result.reasons.append(str(exc))
        result.timings_ms['total'] = (time.perf_counter()-started)*1000
        check_cancelled(cancelled)
        return result
