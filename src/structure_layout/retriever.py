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
from .search import search, merge, R, SIGMA, PENALTY, MINIMUM_SCALES

PARAMETERS = {
    'algorithm_version': 'minimum_structure_cpu_1',
    'rotation_degrees': [0], 'coarse_scales': list(MINIMUM_SCALES),
    'pose_refinement': False, 'execution_backend': 'cpu',
    'coarse_resize': R, 'wall_distance_screen_px': SIGMA,
    'baseline_extra_weight': PENALTY, 'features': FEATURE_PARAMETERS,
    'local_review': LOCAL_PARAMETERS, 'score_is_probability': False,
    'coordinate_space': 'normalized_capture_roi_to_original_clean',
}


def prepare_target(image, layout, floor):
    edges = target_wall_edges(image)
    full, dist, points, low = [], [], [], []
    for channel in edges:
        full.append(cv2.distanceTransform(1-channel, cv2.DIST_L2, 5))
        y, x = np.where(channel)
        points.append(np.column_stack((x, y)).astype(np.float32))
        mask = cv2.resize(channel.astype(np.float32), None, fx=R, fy=R,
                          interpolation=cv2.INTER_AREA)
        low.append(mask)
        dist.append(cv2.distanceTransform((mask < .1).astype(np.uint8), cv2.DIST_L2, 5))
    return attach_raw(dict(layout=layout, floor=floor, dist=dist, full=full,
                           points=points, all_points=np.concatenate(points), low=sum(low)), image)


class PureStructureRetriever:
    """Profile-local immutable index snapshots, rebuilt on content changes.

    No icon resources or disk feature cache are required. The lock serializes
    preparation and recognition so an index cannot change during a search.
    """
    def __init__(self, clean_root, cache_root=None, *, workers=None):
        self.clean_root = Path(clean_root)
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
            paths = [p for p in sorted(self.clean_root.glob('layout_*/*.png'))
                     if p.stem.rsplit('_', 1)[-1] in ('2nd', '1st', 'b1')]
            if not paths:
                raise ValueError('缺少纯结构识别底图')
            content = {}
            for p in paths:
                check_cancelled(cancelled)
                floor = p.stem.rsplit('_', 1)[-1]
                if p.name != f'{p.parent.name}_{floor}.png':
                    raise ValueError('识别底图的布局目录与文件名不一致')
                content[p] = p.read_bytes()
            digests = {p: hashlib.sha256(data).hexdigest() for p, data in content.items()}
            identity = dict(parameters=PARAMETERS, opencv=cv2.__version__,
                            root=str(self.clean_root.resolve()),
                            sources=[(p.relative_to(self.clean_root).as_posix(), digests[p]) for p in paths])
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if fingerprint == self._fingerprint:
                return fingerprint
            targets = []
            for p in paths:
                check_cancelled(cancelled)
                image = cv2.imdecode(np.frombuffer(content[p], np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'无法读取识别资源：{p.name}')
                target = prepare_target(image, p.parent.name, p.stem.rsplit('_', 1)[-1])
                target.update(path=p.relative_to(self.clean_root).as_posix(), sha256=digests[p])
                targets.append(target)
            check_cancelled(cancelled)
            self._targets, self._fingerprint = targets, fingerprint
            return fingerprint

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
                result.timings_ms['index_prepare'] = (time.perf_counter()-preparation)*1000
                searching = time.perf_counter()
                if self.workers > 1 and self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                                        thread_name_prefix='layout-structure')
                raw = search(image, self._targets, executor=self._executor,
                             **({'cancelled': cancelled} if cancelled is not None else {}))
                candidates = []
                if raw.get('floor_candidates'):
                    query = evidence(image, raw['query_edges'])
                    lookup = {(t['layout'], t['floor']): t for t in self._targets}
                    for candidate in raw['floor_candidates']:
                        check_cancelled(cancelled)
                        target = lookup[candidate['layout'], candidate['floor']]
                        details = review(target, candidate, query, raw['box'])
                        candidates.append(dict(candidate, **details))
                        candidates[-1]['score'] = candidate['score']-details['local_penalty']
                    for rank, fit in enumerate(merge(candidates), 1):
                        if not np.isfinite(fit['score']):
                            continue
                        target = lookup[fit['layout'], fit['floor']]
                        # Diagnostic search geometry is not a confirmed projection pose.
                        details = dict(fit, box=raw['box'], rotation=0,
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
