"""Grayscale preparation and candidate image ranking utilities."""
import cv2
import numpy as np
from .fast_gray_detail import FastGrayDetail, features

MAP_THRESH, GAME_THRESH = 3, 60
MAP_MASK_RADIUS, GAME_MASK_RADIUS = 10, 3
SCREEN_FACTOR = 1.0
ADAPTIVE_TOLERANCE_COEFFICIENT = 1000.0
ACCURACY_MODES = (
    {'name': '极速', 'ds': 0.1, 'workers': 4},
    {'name': '均衡', 'ds': 0.4, 'workers': 3},
    {'name': '精度', 'ds': 0.7, 'workers': 2},
)


def adaptive_outside_allowance(scale, ds):
    effective = float(scale)*float(SCREEN_FACTOR)*float(ds)
    return max(1, int(np.ceil(ADAPTIVE_TOLERANCE_COEFFICIENT*effective*effective - 1e-9)))


def _prepare_game(game_bgr, scale, ds, trace=None):
    if game_bgr is None or game_bgr.ndim != 3:
        if trace is not None:
            trace.update(status='rejected', reason='game_image_invalid')
        return None
    effective = float(scale)*float(SCREEN_FACTOR)*float(ds)
    height, width = game_bgr.shape[:2]
    size = (max(1, int(width*effective)), max(1, int(height*effective)))
    if trace is not None:
        trace.update(scale=float(scale), ds=float(ds), map_eff=float(effective),
                     source_shape=list(game_bgr.shape), scaled_shape=[size[1], size[0]])
    gray = cv2.cvtColor(game_bgr, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if effective < 1 else cv2.INTER_LINEAR
    gray = cv2.resize(gray, size, interpolation=interpolation)
    _, binary = cv2.threshold(gray, GAME_THRESH, 255, cv2.THRESH_BINARY)
    if not cv2.countNonZero(binary):
        if trace is not None:
            trace.update(status='rejected', reason='game_mask_empty')
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAME_MASK_RADIUS*2+1, GAME_MASK_RADIUS*2+1))
    expanded = cv2.dilate(binary, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, width, height = cv2.boundingRect(expanded)
    mask = np.ascontiguousarray(expanded[y:y+height, x:x+width])
    color = cv2.resize(game_bgr, size, interpolation=interpolation)
    detail = features(color)
    template = detail[y:y+height, x:x+width]
    template = np.ascontiguousarray(np.where(mask > 0, template, 0), dtype=np.float32)
    if trace is not None:
        trace.update(status='prepared', reason=None,
                     threshold_mask_pixels=int(cv2.countNonZero(binary)),
                     expanded_mask_pixels=int(cv2.countNonZero(expanded)),
                     crop_rect=[int(x), int(y), int(width), int(height)],
                     template_shape=list(template.shape))
    return template, mask, (x, y)


def match_one_array(game_bgr, map_bgr, scale, ds=0.4, *, strict=True, trace=None):
    """Offline adapter using the recovered numerical path; no persistent config."""
    model = FastGrayDetail(map_bgr, ds, map_threshold=MAP_THRESH, map_mask_radius=MAP_MASK_RADIUS)
    prepared = _prepare_game(game_bgr, scale, ds, trace)
    if prepared is None:
        return None
    return model.match_template(*prepared, reject_outside=strict,
                                outside_allowance=adaptive_outside_allowance(scale, ds) if strict else 0,
                                trace=trace)


def rank_maps(game_bgr, candidates, ds=0.4, *, strict=True):
    """Offline adapter; candidates are (name, BGR image, saved scale).

    The original find_best_map sorts valid results by descending score; it has
    no minimum score or runner-up-margin gate within that function.
    """
    results = []
    for name, image, scale in candidates:
        trace = {}
        result = match_one_array(game_bgr, image, scale, ds, strict=strict, trace=trace)
        if result is not None:
            score, location = result
            results.append({'name': name, 'scale': scale*SCREEN_FACTOR, 'score': score,
                            'location': location, 'trace': trace})
    return sorted(results, key=lambda item: item['score'], reverse=True)
