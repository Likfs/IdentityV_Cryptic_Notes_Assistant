"""Fixed-scale matching of published clean floors and same-canvas route projection."""
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from src.app_catalog import CLEAN_VERSION_ID, PROJECT_ROOT, MapCatalog
from src.projection.route_overlay import composite_route_layer
from src.registration.result import OriginalRegistrationResult
from . import resources, matcher, floors
from .screen import screen_info

FLOOR_REVIEW_MARGIN = 1.0
FLOOR_REVIEW_DS = 0.5


def saved_scale_value(candidate):
    try:
        scale = float(candidate['saved_scale'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('原图尚未保存有效尺度') from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('原图尚未保存有效尺度')
    return scale


def _match_floor_candidates(snapshot, screenshot, screen_factor):
    """Prepare caches on the coordinator, then match at most two floors at once.

    Caller holds resource_lock across preparation, workers and final validation.
    Ordered collection preserves candidate order and the original tie policy.
    """
    prepared = matcher.ScreenshotPreparation(screenshot)
    initial_ds = matcher.TEMPLATE_DS
    jobs = []
    last_path = None
    for floor in snapshot.entries:
        try:
            entry, path, _, image, rgba = snapshot.load(floor)
            last_path = path
            saved_scale = saved_scale_value(entry)
            scale = saved_scale * screen_factor
            options = dict(map_threshold=entry.get('map_threshold', matcher.TEMPLATE_MAP_THRESH),
                           map_image=image, image_fingerprint=floors.fingerprint(entry))
            jobs.append((floor, entry, path, image, rgba, saved_scale, scale, options))
        except (ValueError, cv2.error) as exc:
            jobs.append(dict(candidate_floor=floor, reason=str(exc)))

    def match_floor(job, *, ds, expanded=False):
        if isinstance(job, dict):
            return dict(job), None
        floor, entry, path, image, rgba, saved_scale, scale, options = job
        try:
            match = matcher._match_template(screenshot, str(path), scale, prepared=prepared,
                                            expand_search=expanded, ds=ds, **options)
            if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
                raise ValueError('定尺度模板匹配未返回有限分数或坐标')
            match['ds'] = ds
            return (dict(candidate_floor=floor, saved_scale=saved_scale, ds=ds,
                         score=float(match['score']), location=list(match['location']),
                         search=match.get('search', {})),
                    (match, scale, saved_scale, floor, entry, path, rgba, image))
        except (ValueError, cv2.error) as exc:
            return dict(candidate_floor=floor, reason=str(exc),
                        failure='template_too_large' if isinstance(exc, matcher.TemplateSizeError) else 'match_error'), None

    def select_best(results):
        best = None
        for _, candidate in results:
            if candidate is not None and (best is None or candidate[0]['score'] > best[0]['score']):
                best = candidate
        return best

    def score_gap(results):
        scores = sorted((candidate[0]['score'] for _, candidate in results if candidate is not None),
                        reverse=True)
        return scores[0] - scores[1] if len(scores) >= 2 else None

    def run_resolution(ds, pool):
        ready = []
        for job in jobs:
            if isinstance(job, dict):
                ready.append(job)
                continue
            _, _, path, _, _, _, scale, options = job
            try:
                # Shared caches and query groups are populated only by the coordinator.
                matcher.prepare_map(str(path), ds, **options)
                prepared.get(scale, ds)
                ready.append(job)
            except (ValueError, cv2.error) as exc:
                ready.append(dict(candidate_floor=job[0], reason=str(exc)))
        results = list(pool.map(lambda job: match_floor(job, ds=ds), ready))
        best = select_best(results)
        trigger = None
        if best is not None and best[0].get('search', {}).get('boundary_edges'):
            trigger = 'best_at_boundary'
        elif results and best is None and all(attempt.get('failure') == 'template_too_large' for attempt, _ in results):
            trigger = 'all_candidates_too_large'
        if trigger:
            initial = results
            # Retry every configured, scaled floor under the same search policy.
            # The coordinator already prepared all shared inputs; workers only
            # allocate their own temporary padded masks and response arrays.
            results = list(pool.map(lambda job: match_floor(job, ds=ds, expanded=True), ready))
            for (attempt, _), (previous, _) in zip(results, initial):
                attempt.update(initial_attempt=previous, retry_trigger=trigger)
            best = select_best(results)
        return results, best

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='template-match') as pool:
        results, best = run_resolution(initial_ds, pool)
        gap = score_gap(results)
        if best is not None:
            review = dict(triggered=False, threshold=FLOOR_REVIEW_MARGIN, initial_ds=initial_ds,
                          review_ds=FLOOR_REVIEW_DS, initial_gap=gap)
            if initial_ds < FLOOR_REVIEW_DS and gap is not None and gap < FLOOR_REVIEW_MARGIN:
                review.update(triggered=True, initial_floor=best[3],
                              initial_candidates=[attempt for attempt, _ in results])
                reviewed, reviewed_best = run_resolution(FLOOR_REVIEW_DS, pool)
                review.update(review_candidates=[attempt for attempt, _ in reviewed],
                              review_gap=score_gap(reviewed),
                              outcome='reviewed' if reviewed_best is not None else 'failed')
                if reviewed_best is not None:
                    # Select only among review scores; scores at different resolutions
                    # are not comparable. Keep saved scales and source coordinates.
                    results, best = reviewed, reviewed_best
                review['final_floor'] = best[3]
            best[0]['resolution_review'] = review
    return [attempt for attempt, _ in results], best, last_path


def available_variants(profile_key, layout):
    session = floors.fixed_session()
    if profile_key not in session.profiles:
        raise ValueError(f'未配置地图难度：{profile_key}')
    profile = session.profiles[profile_key]
    variants = session.config.get('floor_variants', {}).get(profile_key, {}).get(layout, {})
    return [v for v in profile.template_versions if v in profile.version_names and v in variants]


def check_original_resources(profile_key, layouts):
    """Inventory and fingerprint verification; this is not image-alignment validation."""
    path = PROJECT_ROOT / 'configs/map_catalog.json'
    profile = MapCatalog(path).profile(*profile_key.split('/'))
    versions = profile.template_versions
    names = profile.version_names
    summary = {v: dict(name=names[v], layouts=0, fixed_candidates=0,
                      resource_layouts=0, partial_layouts=0, pending_scales={}, errors=[])
               for v in versions}
    with resources.resource_lock:
        for layout in layouts:
            for variant_id, item in summary.items():
                try:
                    count = floors.check(profile_key, layout, variant_id)
                    item['resource_layouts'] += 1
                    item['fixed_candidates'] += count
                    item['layouts'] += 1
                except (OSError, ValueError, KeyError) as exc:
                    item['errors'].append(f'{layout}: {exc}')
    return summary


def project_original(screenshot, profile_key, layout, opacity, *, variant_id=None, defer_preview=False):
    """Use the frozen profile's first enabled version when no version is supplied."""
    started = perf_counter()
    validation = dict(registration_algorithm='fixed_scale_template', projection_scope='single_floor',
                      coordinate_space='single_floor_source_image', floor_detection_enabled=False,
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
            if variant_id is None:
                session = floors.fixed_session()
                profile = session.profiles.get(profile_key)
                if profile is None:
                    raise ValueError(f'未配置地图难度：{profile_key}')
                if profile.asset_status == 'pending':
                    raise ValueError('资产待配置，请补充底图、路线图和最小缩放值')
                variant_id = profile.template_versions[0]
            display = floors.FrozenSourceSnapshot(profile_key, layout, variant_id)
            snapshot = floors.FrozenSourceSnapshot(profile_key, layout, CLEAN_VERSION_ID)
            result.validation['pending_scale_floors'] = []
            config = snapshot.config
            screen = screen_info(config.get('capture', {}).get('screen_name', ''))
            mode = int(config['perf_mode'])
            if not 0 <= mode < len(matcher.PERF_MODES):
                raise ValueError('性能档位无效')
            matcher.TEMPLATE_DS = matcher.PERF_MODES[mode]['ds']
            matching_started = perf_counter()
            floors.retain_working_set(snapshot.entries.values(), frozen=snapshot._session)
            attempts, best, path = _match_floor_candidates(snapshot, screenshot, screen['factor'])
            result.validation['fixed_candidates'] = attempts
            if best is None:
                raise ValueError('所有固定尺度候选均不可匹配：' + '；'.join(a['reason'] for a in attempts))
            snapshot.validate()
            match, scale, saved_scale, floor, entry, path, rgba, image = best
            validation['matching_source_image'] = entry['source_image']
            entry, path, _, image, rgba = display.load(floor)
            if image.shape[:2] != best[7].shape[:2] or entry['saved_scale'] != saved_scale:
                raise ValueError('路线图与底图画布或尺度不一致，请重新发布')
            display.validate()
            floors.retain_working_set([*snapshot.entries.values(), entry], frozen=display._session)
            result.timings['matching'] = (perf_counter()-matching_started)*1000
        if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
            raise ValueError('定尺度模板匹配未返回有限分数或坐标')
        x, y = match['location']
        result.source_image = image
        result.screenshot_to_original = np.float64([[scale,0,x],[0,scale,y],[0,0,1]])
        result.original_to_screen = np.linalg.inv(result.screenshot_to_original)
        result.validation['template'] = dict(scale=scale, saved_scale=saved_scale,
            screen=screen, score=float(match['score']), location=[int(x),int(y)],
            ds=match.get('ds', matcher.PERF_MODES[mode]['ds']), perf_mode=mode,
            source_sha256=entry['source_sha256'], source_image=entry['source_image'],
            target='single_floor_source_image', scale_policy='saved_original_scale_with_screen_factor',
            variant_id=variant_id, candidate_floor=floor, source_record_sha256=floors.fingerprint(entry),
            scale_provenance=entry.get('scale_provenance', 'imported_saved_scale'),
            search=match.get('search', {}))
        warnings = []
        if match.get('search', {}).get('boundary_edges'):
            warnings.append('匹配位置仍在搜索边界，候选可能错位，请核对位置')
        if match['score'] < 0:
            warnings.append('匹配分数为负，候选可能错位，请核对位置')
        review = match.get('resolution_review', {})
        result.validation['resolution_review'] = review
        if review.get('triggered'):
            if review['outcome'] == 'failed':
                warnings.append('50%分辨率复核未返回有效候选，保留初次候选，请核对楼层')
            elif review['review_gap'] is None or review['review_gap'] < FLOOR_REVIEW_MARGIN:
                warnings.append('50%分辨率复核后楼层仍难区分，请核对楼层')
        result.validation['warnings'] = warnings
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
        result.rejection_reasons = [f'定尺度模板匹配 单层投影失败：{exc}']
        result.timings.setdefault('total', (perf_counter()-started)*1000)
    timings['registration'] = result.timings['total']
    return result, path, layer, composite, timings
