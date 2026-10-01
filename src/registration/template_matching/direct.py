"""Floor-independent 定尺度模板匹配 matching and projection in original-image coordinates."""
from dataclasses import dataclass
from time import perf_counter
import json
import cv2
import numpy as np
from src.app_catalog import PROJECT_ROOT
from src.projection.route_overlay import composite_route_layer
from src.registration.runtime_registration import RuntimeRegistrationResult
from . import runtime, matcher

_images = {}


@dataclass
class OriginalRegistrationResult(RuntimeRegistrationResult):
    original_to_screen: np.ndarray | None = None
    screenshot_to_original: np.ndarray | None = None

    def diagnostics(self):
        return {**super().diagnostics(), 'coordinate_space': 'original_source_image',
                'original_to_screen': self.original_to_screen.tolist() if self.original_to_screen is not None else None,
                'screenshot_to_original': self.screenshot_to_original.tolist() if self.screenshot_to_original is not None else None}


def _source(profile_key, layout):
    config = json.loads((PROJECT_ROOT/'configs/template_matching.json').read_text(encoding='utf-8'))
    prefix = f'assets/maps/{profile_key}/clean/{layout}/'
    entries = [entry for key, entry in config['entries'].items()
               if key.startswith(prefix) and entry['layout'] == layout]
    if not entries:
        raise ValueError('当前布局没有已导入的完整原图和尺度')
    identities = {(e['source_image'], e['source_sha256'], float(e['saved_scale'])) for e in entries}
    if len(identities) != 1:
        raise ValueError('同一布局的完整原图或尺度不一致，请重新导入')
    entry = entries[0]
    path = PROJECT_ROOT/entry['source_image']
    runtime._fingerprint(path, entry['source_sha256'])
    return entry, path, config


def load_original(profile_key, layout):
    """Resolve by layout, without loading or fingerprinting any clean floor asset."""
    with runtime._lock:
        entry, path, config = _source(profile_key, layout)
        key = (str(path), entry['source_sha256'])
        if key not in _images:
            image = matcher.read_image(path)
            # Old source images have a black background; preserve every nonblack mark.
            alpha = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > matcher.TEMPLATE_MAP_THRESH)*255
            _images.clear()  # Keep one full-resolution original, not 21 large BGRA maps.
            _images[key] = (image, np.dstack((image, alpha)))
            # Matcher caches use paths; a newly imported image must rebuild them.
            for cache_key in list(matcher._map_cache):
                if str(path) in cache_key:
                    del matcher._map_cache[cache_key]
        image, rgba = _images[key]
        return entry, path, config, image, rgba


def project_original(screenshot, profile_key, layout, opacity, *, defer_preview=False):
    started = perf_counter()
    validation = dict(registration_algorithm='fixed_scale_template', projection_scope='full',
                      coordinate_space='original_source_image', floor_detection_enabled=False,
                      trusted_history_eligible=False, route_validation_enabled=False,
                      acceptance_policy='template_candidate_without_geometry_validation',
                      registration_backend=runtime.initialize())
    result = OriginalRegistrationResult(False, [], None, None, None, None, validation,
        {'method':'fixed-scale-template', 'calibration':0., 'matching':0.}, None, None, None)
    path = None
    layer = composite = None
    timings = {}
    try:
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('投影不透明度必须在 0～1 之间')
        if not np.any(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) > matcher.TEMPLATE_GAME_THRESH):
            raise ValueError('截图中没有可匹配的可见像素，请保持地图打开后重试')
        with runtime._lock:
            entry, path, config, image, rgba = load_original(profile_key, layout)
            screen = runtime.screen_info(config.get('capture', {}).get('screen_name', ''))
            scale = float(entry['saved_scale'])*screen['factor']
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('原图尚未保存有效尺度')
            mode = int(config['perf_mode'])
            if not 0 <= mode < len(matcher.PERF_MODES):
                raise ValueError('性能档位无效')
            matcher.TEMPLATE_DS = matcher.PERF_MODES[mode]['ds']
            matching_started = perf_counter()
            match = matcher._match_template(screenshot, str(path), scale)
            result.timings['matching'] = (perf_counter()-matching_started)*1000
        if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
            raise ValueError('定尺度模板匹配未返回有限分数或坐标')
        x, y = match['location']
        result.screenshot_to_original = np.float64([[scale,0,x],[0,scale,y],[0,0,1]])
        result.original_to_screen = np.linalg.inv(result.screenshot_to_original)
        result.validation['template'] = dict(scale=scale, saved_scale=float(entry['saved_scale']),
            screen=screen, score=float(match['score']), location=[int(x),int(y)],
            ds=matcher.PERF_MODES[mode]['ds'], perf_mode=mode,
            source_sha256=entry['source_sha256'], source_image=entry['source_image'],
            target='original_source_image', scale_policy='saved_original_scale_with_screen_factor')
        result.timings['total'] = (perf_counter()-started)*1000
        projection_started = perf_counter()
        layer = cv2.warpPerspective(rgba, result.original_to_screen,
                                   (screenshot.shape[1], screenshot.shape[0]), borderValue=(0,0,0,0))
        layer[:,:,3] = np.rint(layer[:,:,3].astype(np.float32)*opacity).astype(np.uint8)
        timings['projection_warp'] = (perf_counter()-projection_started)*1000
        timings['projection_layer'] = timings['projection_warp']
        if not defer_preview:
            composite_started = perf_counter()
            composite, _ = composite_route_layer(screenshot, layer)
            timings['projection_composite'] = (perf_counter()-composite_started)*1000
        timings['projection'] = (perf_counter()-projection_started)*1000
        result.accepted = True
    except (OSError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result.rejection_reasons = [f'定尺度模板匹配 原图投影失败：{exc}']
        result.timings.setdefault('total', (perf_counter()-started)*1000)
    timings['registration'] = result.timings['total']
    return result, path, layer, composite, timings
