"""Floor-independent 定尺度模板匹配 matching and projection in original-image coordinates."""
from time import perf_counter
import json
import hashlib
import cv2
import numpy as np
from src.app_catalog import PROJECT_ROOT, TEMPLATE_VERSION_NAMES
from src.projection.route_overlay import composite_route_layer
from src.registration.result import OriginalRegistrationResult
from . import resources, matcher
from .screen import screen_info

_images = {}


def entry_fingerprint(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def saved_scale_value(candidate):
    try:
        scale = float(candidate['saved_scale'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('原图尚未保存有效尺度') from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('原图尚未保存有效尺度')
    return scale


def available_variants(profile_key, layout):
    config = json.loads((PROJECT_ROOT/'configs/template_matching.json').read_text(encoding='utf-8'))
    variants = config.get('original_variants', {}).get(profile_key, {}).get(layout, {})
    legacy = any(key.startswith(f'assets/maps/{profile_key}/clean/{layout}/')
                 for key in config.get('entries', {}))
    return [v for v in TEMPLATE_VERSION_NAMES if v in variants or (v == 'route_zhanshi_fast' and legacy)]


def _source(profile_key, layout, variant_id='route_zhanshi_fast'):
    if variant_id not in TEMPLATE_VERSION_NAMES:
        raise ValueError('模板模式不支持所选原图版本')
    config = json.loads((PROJECT_ROOT/'configs/template_matching.json').read_text(encoding='utf-8'))
    entry = config.get('original_variants', {}).get(profile_key, {}).get(layout, {}).get(variant_id)
    if entry is not None:
        reference_variant = entry.get('reference_variant_id')
        if reference_variant is not None:
            parent = config['original_variants'][profile_key][layout][reference_variant]
            if reference_variant == variant_id or entry_fingerprint(parent) != entry['reference_variant_sha256']:
                raise ValueError('底图版对应的原图尺度记录已改变，请重新导入底图版')
        candidates = entry.get('candidates', [])
        if not candidates:
            raise ValueError('当前版本没有已保存的固定尺度')
        for candidate in candidates:
            saved_scale_value(candidate)
            for dependency in candidate.get('dependencies', []):
                resources.verify_fingerprint(PROJECT_ROOT/dependency['path'], dependency['sha256'])
            reference = candidate.get('reference_entry_key')
            if reference is not None and entry_fingerprint(config['entries'][reference]) != candidate['reference_entry_sha256']:
                raise ValueError('推算尺度对应的展十记录已改变，请重新推算')
            for reference in candidate.get('reference_entries', []):
                if entry_fingerprint(config['entries'][reference['key']]) != reference['sha256']:
                    raise ValueError('展十原图尺度记录已改变，请重新导入整图版本')
        path = PROJECT_ROOT/entry['source_image']
        if 'source_parts' in entry:
            if variant_id != 'route_clean' or entry_fingerprint(entry['source_parts']) != entry['source_sha256']:
                raise ValueError('三层底图拼接记录已改变，请重新推算')
            parts = entry['source_parts']
            if [part['floor'] for part in parts] != ['2nd', '1st', 'b1']:
                raise ValueError('三层底图拼接记录不完整')
            for part in parts:
                resources.verify_fingerprint(PROJECT_ROOT/part['path'], part['sha256'])
        else:
            resources.verify_fingerprint(path, entry['source_sha256'])
        return entry, path, config
    if variant_id != 'route_zhanshi_fast':
        raise ValueError('当前布局没有所选版本的完整原图及固定尺度')
    prefix = f'assets/maps/{profile_key}/clean/{layout}/'
    entries = [entry for key, entry in config['entries'].items()
               if key.startswith(prefix) and entry['layout'] == layout]
    if not entries:
        raise ValueError('当前布局没有已导入的完整原图和尺度')
    identities = {(e['source_image'], e['source_sha256'], saved_scale_value(e)) for e in entries}
    if len(identities) != 1:
        raise ValueError('同一布局的完整原图或尺度不一致，请重新导入')
    entry = entries[0]
    path = PROJECT_ROOT/entry['source_image']
    resources.verify_fingerprint(path, entry['source_sha256'])
    return entry, path, config


def check_original_resources(profile_key, layouts):
    """Inventory and fingerprint verification; this is not image-alignment validation."""
    summary = {v: dict(name=name, layouts=0, fixed_candidates=0, errors=[])
               for v, name in TEMPLATE_VERSION_NAMES.items()}
    with resources.resource_lock:
        for layout in layouts:
            for variant_id, item in summary.items():
                try:
                    entry, _, _ = _source(profile_key, layout, variant_id)
                    item['layouts'] += 1
                    item['fixed_candidates'] += len(entry.get('candidates', [entry]))
                except (OSError, ValueError, KeyError) as exc:
                    item['errors'].append(f'{layout}: {exc}')
    return summary


def load_original(profile_key, layout, variant_id='route_zhanshi_fast'):
    """Load a complete original; derived scales additionally verify their calibration dependencies."""
    with resources.resource_lock:
        entry, path, config = _source(profile_key, layout, variant_id)
        threshold = int(entry.get('map_threshold', matcher.TEMPLATE_MAP_THRESH))
        preserve_colors = entry.get('preserve_colored_routes', variant_id == 'route_lhp_solo')
        key = (str(path), entry['source_sha256'], variant_id, threshold, preserve_colors)
        if key not in _images:
            if 'source_parts' in entry:
                parts = []
                for part in entry['source_parts']:
                    tile = matcher.read_image(PROJECT_ROOT/part['path'])
                    x, y, width, height = part['region']
                    if min(x, y) < 0 or tile.shape[:2] != (height, width):
                        raise ValueError('三层底图拼接尺寸已改变，请重新推算')
                    parts.append((tile, x, y, width, height))
                image = np.zeros((max(y+h for _, x, y, w, h in parts),
                                  max(x+w for _, x, y, w, h in parts), 3), np.uint8)
                for tile, x, y, width, height in parts:
                    image[y:y+height, x:x+width] = tile
            else:
                image = matcher.read_image(path)
            # Zhanshi source images have a black background; preserve every nonblack mark.
            foreground = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > threshold
            if preserve_colors:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                foreground |= (hsv[:,:,1] >= 75) & (hsv[:,:,2] >= 60)
            alpha = np.uint8(foreground)*255
            _images.clear()  # Bound memory to one full-resolution original.
            _images[key] = (image, np.dstack((image, alpha)))
            # Matcher caches use paths; a newly imported image must rebuild them.
            for cache_key in list(matcher._map_cache):
                if str(path) in cache_key:
                    del matcher._map_cache[cache_key]
        image, rgba = _images[key]
        return entry, path, config, image, rgba


def project_original(screenshot, profile_key, layout, opacity, *, variant_id='route_zhanshi_fast', defer_preview=False):
    started = perf_counter()
    validation = dict(registration_algorithm='fixed_scale_template', projection_scope='full',
                      coordinate_space='original_source_image', floor_detection_enabled=False,
                      trusted_history_eligible=False, route_validation_enabled=False,
                      acceptance_policy='template_candidate_without_geometry_validation',
                      registration_backend=resources.backend_info())
    result = OriginalRegistrationResult(False, [], validation,
        {'method':'fixed-scale-template', 'calibration':0., 'matching':0.})
    path = None
    layer = composite = None
    timings = {}
    try:
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('投影不透明度必须在 0～1 之间')
        if not np.any(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) > matcher.TEMPLATE_GAME_THRESH):
            raise ValueError('截图中没有可匹配的可见像素，请保持地图打开后重试')
        with resources.resource_lock:
            entry, path, config, image, rgba = load_original(profile_key, layout, variant_id)
            screen = screen_info(config.get('capture', {}).get('screen_name', ''))
            mode = int(config['perf_mode'])
            if not 0 <= mode < len(matcher.PERF_MODES):
                raise ValueError('性能档位无效')
            matcher.TEMPLATE_DS = matcher.PERF_MODES[mode]['ds']
            matching_started = perf_counter()
            attempts = []
            best = None
            for candidate in entry.get('candidates', [{'saved_scale': entry.get('saved_scale')}]):
                try:
                    saved_scale = saved_scale_value(candidate)
                    scale = saved_scale*screen['factor']
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError('原图尚未保存有效尺度')
                    options = {}
                    if 'map_threshold' in entry or candidate.get('search_region') is not None:
                        options = dict(map_threshold=entry.get('map_threshold', matcher.TEMPLATE_MAP_THRESH),
                                       search_region=candidate.get('search_region'))
                    if 'source_parts' in entry:
                        options.update(map_image=image, image_fingerprint=entry['source_sha256'])
                    match = matcher._match_template(screenshot, str(path), scale, **options)
                    if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
                        raise ValueError('定尺度模板匹配未返回有限分数或坐标')
                    attempts.append(dict(candidate_floor=candidate.get('floor'), saved_scale=saved_scale,
                                         score=float(match['score']), location=list(match['location'])))
                    if best is None or match['score'] > best[0]['score']:
                        best = match, scale, saved_scale, candidate
                except (ValueError, cv2.error) as exc:
                    attempts.append(dict(candidate_floor=candidate.get('floor'), reason=str(exc)))
            result.validation['fixed_candidates'] = attempts
            if best is None:
                raise ValueError('所有固定尺度候选均不可匹配：' + '；'.join(a['reason'] for a in attempts))
            match, scale, saved_scale, candidate = best
            result.timings['matching'] = (perf_counter()-matching_started)*1000
        if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
            raise ValueError('定尺度模板匹配未返回有限分数或坐标')
        x, y = match['location']
        result.screenshot_to_original = np.float64([[scale,0,x],[0,scale,y],[0,0,1]])
        result.original_to_screen = np.linalg.inv(result.screenshot_to_original)
        result.validation['template'] = dict(scale=scale, saved_scale=saved_scale,
            screen=screen, score=float(match['score']), location=[int(x),int(y)],
            ds=matcher.PERF_MODES[mode]['ds'], perf_mode=mode,
            source_sha256=entry['source_sha256'], source_image=entry['source_image'],
            target='original_source_image', scale_policy='saved_original_scale_with_screen_factor',
            variant_id=variant_id, candidate_floor=candidate.get('floor'),
            scale_provenance=entry.get('scale_provenance', 'imported_saved_scale'))
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
