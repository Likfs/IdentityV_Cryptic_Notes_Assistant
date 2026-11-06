"""Thread-safe CPU index and manual-confirmation layout candidates."""
from pathlib import Path
from src.floor_definitions import FLOORS, DEFAULT_FLOORS
import hashlib
import importlib.util
import json
import marshal
import sys
import threading
import time

import cv2
import numpy as np

from src.layout_retrieval import LayoutCandidate, LayoutRetrievalResult
from src.recognition_control import check_cancelled, recognition_lock
from .features import target_wall_edges, PARAMETERS as FEATURE_PARAMETERS
from .local import attach_raw, evidence, review, PARAMETERS as LOCAL_PARAMETERS
from .search import merge, R, SIGMA, PENALTY
from .scales import read_scale_overrides
from .feature_store import FeatureStore, digest, file_digest, FORMAT_VERSION

PARAMETERS = {
    'algorithm_version': 'minimum_structure_cpu_2',
    'scale_policy': 'per_source_published_minimum',
    'rotation_degrees': [0], 'coarse_scales': [],
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


def algorithm_code_fingerprints():
    """Loader code is available in both source checkouts and frozen PYZ packages."""
    result = {}
    for name in ('features', 'local', 'search', 'retriever', 'correlation', 'feature_store'):
        fullname = f'{__package__}.{name}'
        spec = importlib.util.find_spec(fullname)
        code = spec.loader.get_code(fullname)
        if code is None:
            raise ValueError(f'无法核对识别算法版本：{fullname}')
        result[name] = hashlib.sha256(marshal.dumps(code)).hexdigest()
    return result


class PureStructureRetriever:
    """Profile-local indexes; fixed resources are validated before first publication.

    Published source-image profiles keep only metadata and verified disk handles.
    The lock serializes preparation, recognition and mapped-file closure.
    """
    def __init__(self, clean_root, cache_root=None, *, workers=None, floors=DEFAULT_FLOORS,
                 fixed_resources=True):
        self.clean_root = Path(clean_root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._feature_store = None
        self.fixed_resources = bool(fixed_resources)
        self.floors = tuple(floors)
        if not self.floors or len(set(self.floors)) != len(self.floors) or any(f not in FLOORS for f in self.floors):
            raise ValueError('识别楼层配置无效')
        self._lock = threading.RLock()
        self._fingerprint = ''
        self._targets = []

    def close(self):
        """Release verified feature handles after any active recognition finishes."""
        with self._lock:
            self._targets = []
            self._fingerprint = ''
            if self._feature_store is not None:
                self._feature_store.close()
                self._feature_store = None

    def prepare(self, *, cancelled=None, progress=None):
        with recognition_lock(self._lock, cancelled):
            if self.fixed_resources and self._fingerprint:
                return self._fingerprint
            from src.runtime_resources import recognition_profile
            published = recognition_profile(self.clean_root)
            if published is None or not published.get('source_images'):
                raise ValueError('缺少已发布的干净底图识别配置，请重新发布素材')
            return self._prepare_disk(published, cancelled, progress)

    def _disk_sources(self, published, cancelled):
        paths = [p for p in sorted(self.clean_root.glob('layout_*/*.png'))
                 if p.stem.rsplit('_', 1)[-1] in self.floors]
        declared = {name for name in published['targets']
                    if name.rsplit('_', 1)[-1].removesuffix('.png') in self.floors}
        if not paths or {p.relative_to(self.clean_root).as_posix() for p in paths} != declared:
            raise ValueError('识别底图清单与发布清单不一致，请重新发布')
        sources = {}
        root = self.clean_root.resolve()
        for path in paths:
            check_cancelled(cancelled)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            name = path.relative_to(self.clean_root).as_posix()
            with path.open('rb') as stream:
                value = file_digest(stream, cancelled)
            if value != published['targets'][name]['source_sha256']:
                raise ValueError(f'识别底图与发布清单不一致，请重新发布：{name}')
            sources[path] = (resolved, value)
        return sources

    def _prepare_disk(self, published, cancelled, progress):
        from src.paths import CACHE_ROOT
        from src.runtime_resources import recognition_profile
        sources = self._disk_sources(published, cancelled)
        overrides = read_scale_overrides(self.clean_root)
        code = algorithm_code_fingerprints()
        algorithm = dict(format=FORMAT_VERSION, parameters=PARAMETERS, code=code,
                         python=list(sys.version_info[:3]), opencv=cv2.__version__, numpy=np.__version__)
        identity = dict(algorithm=algorithm, published=published, floors=self.floors,
                        root=str(self.clean_root.resolve()),
                        sources=[(p.relative_to(self.clean_root).as_posix(), value[1])
                                 for p, value in sources.items()])
        fingerprint = digest(identity)
        if fingerprint == self._fingerprint:
            return fingerprint
        root = self.cache_root or CACHE_ROOT/'structure_features'/digest(str(self.clean_root.resolve()))[:16]
        store = FeatureStore(root)
        targets = []
        try:
            if progress is not None:
                progress(0, len(sources))
            for path, (_, source_hash) in sources.items():
                check_cancelled(cancelled)
                name = path.relative_to(self.clean_root).as_posix()
                record = overrides[name]
                metadata = dict(layout=path.parent.name, floor=path.stem.rsplit('_', 1)[-1],
                                path=name, sha256=source_hash, scales=record['scales'],
                                scale_provenance='verified_source_override', coarse_resize=R)
                feature_identity = dict(algorithm=algorithm, source=metadata)
                def build():
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != source_hash:
                        raise ValueError(f'识别底图在特征生成期间变化：{name}')
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f'无法读取识别底图：{name}')
                    target = prepare_target(image, metadata['layout'], metadata['floor'])
                    target.update(metadata)
                    return target
                key = store.ensure(feature_identity, build, cancelled=cancelled)
                targets.append(dict(metadata, cache_key=key))
                if progress is not None:
                    progress(len(targets), len(sources))
            # Publish neither changed sources nor a partial batch. Completed floor
            # artifacts remain reusable after cancellation or a later repair.
            if (recognition_profile(self.clean_root) != published
                    or self._disk_sources(published, cancelled) != sources
                    or algorithm_code_fingerprints() != code):
                raise ValueError('识别资源或算法在特征准备期间变化，请重新启动')
            check_cancelled(cancelled)
        except BaseException:
            store.close()
            raise
        previous = self._feature_store
        self._feature_store, self._targets, self._fingerprint = store, targets, fingerprint
        self._published_profile = published
        self.feature_cache_stats = dict(total=len(targets), built=store.built,
                                        reused=len(targets)-store.built)
        if previous is not None:
            previous.close()
        store.prune()
        return fingerprint

    def _search_disk(self, image, cancelled):
        from .search import prepare_query, search_target, CorrelationWorkspace
        query = prepare_query(image, self._targets, cancelled=cancelled)
        if query is None:
            return dict(floor_candidates=[])
        local_query = evidence(image, query['query_edges'])
        workspace = CorrelationWorkspace()
        arguments = {k: query[k] for k in ('templates', 'qpoints', 'negative', 'box', 'query_geometry')}
        candidates = []
        for metadata in self._targets:
            check_cancelled(cancelled)
            with self._feature_store.mapped(metadata['cache_key']) as target:
                poses = search_target(target, **arguments, workspace=workspace, cancelled=cancelled)
                lookup = {(target['layout'], target['floor']): target}
                for pose in poses or []:
                    candidates.append(review_candidate(pose, lookup, local_query, query['box'], cancelled))
                del lookup
        return dict(floor_candidates=candidates, box=query['box'], query_edges=query['query_edges'])

    def recognize(self, screenshot_bgr, reference_size=(1180, 1050), *, cancelled=None):
        check_cancelled(cancelled)
        started = time.perf_counter()
        result = LayoutRetrievalResult(backend='pure_structure',
            parameters=dict(PARAMETERS, workers=1), reference_size=list(reference_size))
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
                result.parameters['coarse_scales'] = sorted({scale for target in self._targets for scale in target['scales']})
                result.timings_ms['index_prepare'] = (time.perf_counter()-preparation)*1000
                searching = time.perf_counter()
                result.parameters.update(workers=1, feature_storage='disk_mapped_floor_v1')
                raw = self._search_disk(image, cancelled)
                if not self.fixed_resources:
                    from src.runtime_resources import recognition_profile
                    if recognition_profile(self.clean_root) != self._published_profile:
                        raise ValueError('识别期间发布配置已改变')
                    self._disk_sources(self._published_profile, cancelled)
                candidates = []
                if raw.get('floor_candidates'):
                    lookup = {(t['layout'], t['floor']): t for t in self._targets}
                    candidates = raw['floor_candidates']
                    for rank, fit in enumerate(merge(candidates), 1):
                        if not np.isfinite(fit['score']):
                            continue
                        target = lookup[fit['layout'], fit['floor']]
                        # Diagnostic search geometry is not a confirmed projection pose.
                        details = dict(fit, box=raw['box'], rotation=0,
                                       recognition_zoom=None,
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
