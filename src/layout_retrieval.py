"""Fixed-direction icon/structure retrieval over the active profile's clean maps."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src.paths import PROJECT_ROOT, CACHE_ROOT

RESOURCE_ROOT = PROJECT_ROOT / 'assets/layout_retrieval'
BACKEND_LABELS = {'pure_structure': '纯结构识别（最小缩放·CPU）', 'icon_structure': '图标与结构匹配'}

def normalize_recognition_backend(value):
    return value if isinstance(value, str) and value in BACKEND_LABELS else 'pure_structure'
PARAMETERS = {
    'algorithm_version': 'fixed_direction_icon_structure_layout_v1',
    'rotation_degrees': [0], 'scale_min': .30, 'scale_max': 1.40,
    'scale_step': .005, 'target_wall_resize': .6, 'wall_distance_sigma': 1.6,
    'wall_weight': .8, 'floor_weight': .2, 'geometry_min': .60,
    'target_icon_min': .77, 'query_icon_min': .90,
    'occluded_query_icon_min': .65, 'occlusion_min': .20,
    'local_icon_min': .65, 'anchor_icon_widths': list(range(18, 55, 2)),
    'fallback_scales': [.19, .81, .015], 'local_icon_factors': [.7, .85, 1., 1.15, 1.3],
    'wall_point_step_divisor': 2500, 'floor_point_step_divisor': 3000,
    'score_is_probability': False,
}


@dataclass(frozen=True)
class LayoutCandidate:
    rank: int
    layout: str
    path: str
    sha256: str
    score: float
    wall_score: float
    floor_score: float
    icon_score: float
    evidence_status: str
    details: dict


@dataclass
class LayoutRetrievalResult:
    candidates: list[LayoutCandidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    resource_fingerprint: str = ''
    input_size: list[int] = field(default_factory=list)
    reference_size: list[int] = field(default_factory=list)
    backend: str = 'icon_structure'
    parameters: dict = field(default_factory=lambda: dict(PARAMETERS))

    @property
    def accepted(self):
        # Available candidates can be inspected even when evidence is weak.
        return bool(self.candidates)

    @property
    def strong_evidence(self):
        return bool(self.candidates and self.candidates[0].evidence_status == '图标位置通过')

    @property
    def margin(self):
        return self.candidates[0].score - self.candidates[1].score if len(self.candidates) > 1 else None

    def diagnostics(self):
        return {**asdict(self), 'backend': self.backend, 'accepted': self.accepted,
                'strong_evidence': self.strong_evidence, 'top1_top2_gap': self.margin,
                'parameters': self.parameters}

    def decision(self):
        """Expose manual-confirmation candidates without inventing probabilities."""
        if not self.candidates:
            return None
        return {
            'backend': self.backend, 'accepted': False, 'layout': None,
            'best_label': self.candidates[0].layout, 'score': self.candidates[0].score,
            'margin': self.margin or 0., 'score_is_probability': False,
            'ranking': [{'layout': item.layout, 'score': item.score,
                         'evidence_status': item.evidence_status} for item in self.candidates],
        }


def create_layout_retriever(backend, clean_root, **kwargs):
    if backend == 'pure_structure':
        from .structure_layout import PureStructureRetriever
        return PureStructureRetriever(clean_root, **kwargs)
    if backend == 'icon_structure':
        return LayoutRetriever(clean_root, **kwargs)
    raise ValueError('未知的布局识别方法')


def read_image(path: Path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取识别资源：{path.name}')
    return image


def wall_edges(image, query=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 65)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    icons = (((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 170)) &
             (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 110)).astype(np.uint8)
    icons = cv2.dilate(icons, np.ones((9, 9), np.uint8))
    edges[icons > 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                           minLineLength=22 if query else 18, maxLineGap=5)
    channels = np.zeros((2, *gray.shape), np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx <= 2 and dy >= 22:
                cv2.line(channels[0], (x1, y1), (x2, y2), 1, 1)
            elif dy <= 2 and dx >= 22:
                cv2.line(channels[1], (x1, y1), (x2, y2), 1, 1)
    return channels


def target_maps(image):
    maps = []
    for channel in wall_edges(image):
        channel = cv2.resize(channel.astype(np.float32), None, fx=.6, fy=.6,
                             interpolation=cv2.INTER_AREA)
        distance = cv2.distanceTransform((channel < .1).astype(np.uint8), cv2.DIST_L2, 3)
        maps.append(np.exp(-distance * distance / (2 * 1.6 ** 2)))
    return maps


def floor_masks(image, query=False):
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    room = ((hue >= 5) & (hue <= 32) & (saturation >= 22) &
            (saturation < 120) & (value > 65)).astype(np.float32)
    corridor = ((((hue >= 90) & (hue <= 125) & (saturation < 100)) |
                 (saturation < 25)) & (value > 60)).astype(np.float32)
    if query:
        room = cv2.erode(room, np.ones((5, 5), np.uint8))
        corridor = cv2.erode(corridor, np.ones((5, 5), np.uint8))
    return [room, corridor]


def query_data(image):
    channels = wall_edges(image, True)
    yy, xx = np.where(channels.sum(axis=0) > 0)
    if not len(xx):
        raise ValueError('截图没有有效墙线结构，请保持地图打开后重试')
    box = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
    def sampled(masks, divisor):
        points = []
        for mask in masks:
            y, x = np.where(mask > 0)
            step = max(1, len(x) // divisor)
            points.append((x[::step], y[::step]))
        return points
    return dict(walls=sampled(channels, 2500), floors=sampled(floor_masks(image, True), 3000), box=box)


def detect_query(image, reference):
    ref = reference.astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    if gray.shape[0] < ref.shape[0] or gray.shape[1] < ref.shape[1]:
        raise ValueError('截图尺寸不足以定位入口图标')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    occluded = (((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 170)) &
                (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 100)).astype(np.uint8)
    valid = (1 - cv2.dilate(occluded, np.ones((3, 3), np.uint8))).astype(np.float32)
    ones = np.ones(ref.shape, np.float32)
    def correlate(a, b):
        return cv2.matchTemplate(a, b, cv2.TM_CCORR)
    n = correlate(valid, ones)
    si, si2 = correlate(gray * valid, ones), correlate(gray * gray * valid, ones)
    st, st2, sit = correlate(valid, ref), correlate(valid, ref * ref), correlate(gray * valid, ref)
    vi, vt = si2 - si * si / np.maximum(n, 1), st2 - st * st / np.maximum(n, 1)
    scores = (sit - si * st / np.maximum(n, 1)) / np.sqrt(np.maximum(vi * vt, 1e-8))
    scores[(n < ref.size * .25) | (vi / np.maximum(n, 1) < .001) |
           (vt / np.maximum(n, 1) < .001)] = -1
    _, score, _, loc = cv2.minMaxLoc(np.clip(scores, -1, 1))
    height, width = ref.shape
    return dict(confidence=float(score), x=loc[0], y=loc[1], w=width, h=height,
                cx=loc[0] + width / 2, cy=loc[1] + height / 2, ref=0, mirrored=False,
                occlusion_fraction=float(np.clip(1 - n[loc[1], loc[0]] / ref.size, 0, 1)), masked=True)


def detect_target(image, reference):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best = {'confidence': -1}
    for width in range(18, 55, 2):
        height = round(reference.shape[0] * width / reference.shape[1])
        if height > gray.shape[0] or width > gray.shape[1]:
            continue
        template = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
        for mirrored in [False, True]:
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(
                gray, cv2.flip(template, 1) if mirrored else template, cv2.TM_CCOEFF_NORMED))
            if score > best['confidence']:
                best = dict(confidence=float(score), x=loc[0], y=loc[1], w=width, h=height,
                            cx=loc[0] + width / 2, cy=loc[1] + height / 2, mirrored=mirrored)
    return best


def reliable_query(icon):
    return icon['confidence'] >= .9 or (icon['confidence'] >= .65 and icon.get('occlusion_fraction', 0) > .2)


def anchor_fit(query, target, qi, ti):
    # Direction is fixed in live play. Mirrored glyph templates do not rotate the map.
    scales = np.arange(.30, 1.401, .005, dtype=np.float32)
    scores, count = np.zeros(len(scales), np.float32), 0
    for (xx, yy), wall in zip(query['walls'], target['walls']):
        if not len(xx):
            continue
        mx = (ti['cx'] + (xx[None, :] - qi['cx']) * scales[:, None]) * .6
        my = (ti['cy'] + (yy[None, :] - qi['cy']) * scales[:, None]) * .6
        values = cv2.remap(wall, mx.astype(np.float32), my.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        scores += values.sum(axis=1)
        count += len(xx)
    scores /= max(count, 1)
    best = {'score': -1}
    for idx in np.argsort(scores)[-12:]:
        scale, total, num = float(scales[idx]), 0., 0
        for (xx, yy), mask in zip(query['floors'], target['floors']):
            if not len(xx):
                continue
            mx, my = ti['cx'] + (xx - qi['cx']) * scale, ti['cy'] + (yy - qi['cy']) * scale
            values = cv2.remap(mask, mx.astype(np.float32)[None, :], my.astype(np.float32)[None, :],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            total += float(values.sum())
            num += len(xx)
        floor = total / max(num, 1)
        base = .8 * float(scores[idx]) + .2 * floor
        if base > best['score']:
            x0, y0, _, _ = query['box']
            x, y = ti['cx'] - scale * (qi['cx'] - x0), ti['cy'] - scale * (qi['cy'] - y0)
            best = dict(score=base, wall_score=float(scores[idx]), floor_score=floor, scale=scale,
                        rotation=0, x=float(x), y=float(y), matrix=[[scale, 0., float(x)], [0., scale, float(y)]],
                        box=query['box'])
    base, icon = best['score'], float(ti['confidence'])
    best.update(base_score=base, icon_score=icon, query_icon=qi, target_icon=ti,
                matched_icon=dict(confidence=icon, cx=ti['cx'], cy=ti['cy']), fit='图标锚定定位')
    best.update(score=.6 * base + .4 * icon if base >= .60 else .2 * base,
                icon_status='图标位置通过' if base >= .60 else '几何证据较弱（图标已定位）')
    return best


def fallback_templates(image, box):
    x0, y0, x1, y1 = box
    edges = wall_edges(image, True)[:, y0:y1, x0:x1]
    templates = []
    for scale in np.arange(.19, .811, .015):
        channels = [cv2.resize(ch.astype(np.float32), None, fx=float(scale), fy=float(scale),
                               interpolation=cv2.INTER_AREA) for ch in edges]
        total = sum(ch.sum() for ch in channels)
        if total > 0:
            templates.append((scale, channels, total))
    return templates


def fallback_fit(templates, box, image, target, qi, references):
    best = {'score': -1}
    for scale, channels, total in templates:
        height, width = channels[0].shape
        if height >= target['walls'][0].shape[0] or width >= target['walls'][0].shape[1]:
            continue
        scores = sum(cv2.matchTemplate(mask, ch, cv2.TM_CCORR)
                     for mask, ch in zip(target['walls'], channels)) / total
        _, score, _, loc = cv2.minMaxLoc(scores)
        if score > best['score']:
            best = dict(score=float(score), wall_score=float(score), floor_score=0.,
                        scale=float(scale / .6), rotation=0, x=loc[0] / .6, y=loc[1] / .6, box=box)
    if best['score'] < 0:
        return None
    ti = target['icon']
    best.update(base_score=best['score'], query_icon=qi, target_icon=ti, fit='墙线搜索（图标未可靠定位）')
    if not reliable_query(qi):
        best.update(score=.6 * best['base_score'], icon_score=0., icon_status='入口图标未可靠定位')
        return best
    x0, y0, _, _ = box
    cx, cy = best['x'] + (qi['cx'] - x0) * best['scale'], best['y'] + (qi['cy'] - y0) * best['scale']
    local = dict(confidence=0., cx=cx, cy=cy)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for reference in references:
        for factor in [.7, .85, 1., 1.15, 1.3]:
            width = max(9, round(qi['w'] * best['scale'] * factor))
            height, radius = round(width * reference.shape[0] / reference.shape[1]), max(3, round(width * .16))
            left, top = round(cx - width / 2) - radius, round(cy - height / 2) - radius
            right, bottom = left + width + 2 * radius, top + height + 2 * radius
            if left < 0 or top < 0 or right > gray.shape[1] or bottom > gray.shape[0]:
                continue
            crop, template = gray[top:bottom, left:right], cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
            for mirrored in [False, True]:
                _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(
                    crop, cv2.flip(template, 1) if mirrored else template, cv2.TM_CCOEFF_NORMED))
                if score > local['confidence']:
                    local = dict(confidence=float(score), cx=left + loc[0] + width / 2, cy=top + loc[1] + height / 2)
    best.update(icon_score=local['confidence'], matched_icon=local)
    if best['base_score'] < .60:
        best.update(score=.2 * best['base_score'], icon_status='墙线与地面不符')
    elif local['confidence'] >= .65:
        best.update(score=.6 * best['base_score'] + .4 * local['confidence'], icon_status='图标位置通过')
    elif ti['confidence'] >= .77:
        best.update(score=.2 * best['base_score'], icon_status='图标位置不符')
    else:
        best.update(score=.6 * best['base_score'], icon_status='候选图标未可靠检出')
    return best


class LayoutRetriever:
    """Thread-safe lazy index, invalidated by candidate and icon content hashes."""
    def __init__(self, clean_root: Path, resource_root: Path = RESOURCE_ROOT, cache_root: Path | None = None):
        self.clean_root = Path(clean_root)
        self.resource_root = Path(resource_root)
        profile_hash = hashlib.sha256(str(self.clean_root.resolve()).encode('utf-8')).hexdigest()[:16]
        self.cache_root = Path(cache_root) if cache_root is not None else CACHE_ROOT / 'layout_retrieval' / profile_hash
        self._lock = threading.RLock()
        self._fingerprint = ''
        self._targets = []
        self._references = []

    def prepare(self):
        with self._lock:
            paths = sorted(self.clean_root.glob('layout_*/layout_*_2nd.png'))
            if any(p.name != f'{p.parent.name}_2nd.png' for p in paths):
                raise ValueError('识别底图的布局目录与文件名不一致')
            icons = [self.resource_root / 'icons' / f'{name}.png' for name in ('query', 'clean_09', 'clean_15')]
            if not paths or any(not p.is_file() for p in icons):
                raise ValueError('新算法识别资源不完整，缺少候选底图或图标模板')
            digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + icons}
            fingerprint = hashlib.sha256((PARAMETERS['algorithm_version'] + cv2.__version__ +
                ''.join(p.relative_to(self.clean_root).as_posix() + digests[p] for p in paths) +
                ''.join(p.name + digests[p] for p in icons)).encode('utf-8')).hexdigest()
            if fingerprint == self._fingerprint:
                return fingerprint
            references = [cv2.cvtColor(read_image(p), cv2.COLOR_BGR2GRAY) for p in icons]
            cache_file = self.cache_root / 'target_icons.json'
            try:
                cache = json.loads(cache_file.read_text(encoding='utf-8'))
                if not isinstance(cache, dict):
                    cache = {}
            except (OSError, ValueError):
                cache = {}
            targets = []
            for path in paths:
                image = read_image(path)
                key = f'{PARAMETERS["algorithm_version"]}:{cv2.__version__}:{digests[icons[0]]}:{digests[path]}'
                icon = cache.get(key)
                if not (isinstance(icon, dict) and all(isinstance(icon.get(k), (int, float))
                        and np.isfinite(icon[k]) for k in ('confidence', 'cx', 'cy', 'w', 'h', 'x', 'y'))):
                    icon = detect_target(image, references[0])
                cache[key] = icon
                targets.append(dict(image=image, walls=target_maps(image), floors=floor_masks(image),
                                    icon=icon, layout=path.parent.name,
                                    path=path.relative_to(self.clean_root).as_posix(),
                                    sha256=digests[path]))
            # A cache write failure must not prevent recognition of valid resources.
            try:
                self.cache_root.mkdir(parents=True, exist_ok=True)
                pending = cache_file.with_suffix('.tmp')
                pending.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
                pending.replace(cache_file)
            except OSError:
                pass
            self._references, self._targets, self._fingerprint = references, targets, fingerprint
            return fingerprint

    def recognize(self, screenshot_bgr, reference_size=(1180, 1050)):
        started = time.perf_counter()
        result = LayoutRetrievalResult(reference_size=list(reference_size))
        try:
            if (not isinstance(screenshot_bgr, np.ndarray) or screenshot_bgr.dtype != np.uint8 or
                    screenshot_bgr.ndim != 3 or screenshot_bgr.shape[2] != 3 or not screenshot_bgr.size):
                raise ValueError('截图必须为非空 BGR 三通道 uint8 图片')
            result.input_size = [screenshot_bgr.shape[1], screenshot_bgr.shape[0]]
            width, height = map(int, reference_size)
            if width < 50 or height < 47:
                raise ValueError('参考截图尺寸不足以定位入口图标')
            image = screenshot_bgr if result.input_size == [width, height] else cv2.resize(
                screenshot_bgr, (width, height), interpolation=cv2.INTER_AREA
                if width < screenshot_bgr.shape[1] else cv2.INTER_LINEAR)
            query = query_data(image)
            with self._lock:
                preparation = time.perf_counter()
                result.resource_fingerprint = self.prepare()
                result.timings_ms['index_prepare'] = (time.perf_counter() - preparation) * 1000
                search = time.perf_counter()
                qi = detect_query(image, self._references[0])
                templates = None
                candidates = []
                for target in self._targets:
                    if reliable_query(qi) and target['icon']['confidence'] >= .77:
                        fit = anchor_fit(query, target, qi, target['icon'])
                    else:
                        if templates is None:
                            templates = fallback_templates(image, query['box'])
                        fit = fallback_fit(templates, query['box'], target['image'], target, qi, self._references)
                    if fit is not None and np.isfinite(fit['score']):
                        candidates.append((target, fit))
                candidates.sort(key=lambda item: (-item[1]['score'], item[0]['path']))
                result.candidates = [LayoutCandidate(
                    rank=i + 1, layout=target['layout'], path=target['path'],
                    sha256=target['sha256'], score=fit['score'], wall_score=fit['wall_score'],
                    floor_score=fit['floor_score'], icon_score=fit['icon_score'],
                    evidence_status=fit['icon_status'], details=fit)
                    for i, (target, fit) in enumerate(candidates)]
                result.timings_ms['search'] = (time.perf_counter() - search) * 1000
                if not candidates:
                    result.reasons.append('当前截图没有有效匹配候选')
        except (ValueError, OSError, cv2.error) as exc:
            result.reasons.append(str(exc))
        result.timings_ms['total'] = (time.perf_counter() - started) * 1000
        return result
