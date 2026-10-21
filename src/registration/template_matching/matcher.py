"""Fixed-scale weighted binary correlation and grayscale scale calibration.

_match_template accepts screenshot-to-map scale; the maximum score supplies
a candidate position. Performance presets control downsampling."""
import cv2
import numpy as np
from PIL import Image
from . import fog_crop

TEMPLATE_WW, TEMPLATE_BB, TEMPLATE_WB, TEMPLATE_BW = 15.0, 0.1, -80.0, 0.0
TEMPLATE_MAP_THRESH, TEMPLATE_GAME_THRESH = 3, 56
TEMPLATE_DS = 0.1
TEMPLATE_DS_OPTIONS = (1.0, 0.5, 0.25, 0.1)
TEMPLATE_MAP_CROP_MARGIN = 35
PERF_MODES = (
    {'name': 'OFF', 'algo': 'template', 'ds': 1.0},
    {'name': 'FAST', 'algo': 'template', 'ds': 0.5},
    {'name': 'MAX', 'algo': 'template', 'ds': 0.25},
    {'name': 'TURBO', 'algo': 'template', 'ds': 0.1},
)
_map_cache = {}
# Extra retained float masks only; large templates fall back to request-local masks.
FLOAT_MASK_CACHE_BYTES = 64 * 1024 * 1024


class TemplateSizeError(ValueError):
    """The visible query cannot fit the current bounded search canvas."""


def _float_masks(map_bw):
    white = (map_bw > 0).astype(np.float32)
    black = (map_bw == 0).astype(np.float32)
    white.flags.writeable = black.flags.writeable = False
    return white, black


def read_image(path):
    return cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2BGR)


def _algo_key(map_path):
    return ('gray', map_path)


def preload_map(map_path):
    key = _algo_key(map_path)
    if key not in _map_cache:
        _map_cache[key] = cv2.cvtColor(read_image(map_path), cv2.COLOR_BGR2GRAY)
    return _map_cache[key]


def _crop_map(map_bw, ds):
    height, width = map_bw.shape[:2]
    cropped = fog_crop.crop_binary(map_bw)
    margin = max(1, int(round(TEMPLATE_MAP_CROP_MARGIN * ds)))
    left = max(0, cropped['left'] - margin)
    top = max(0, cropped['top'] - margin)
    right = min(width, cropped['left'] + cropped['image'].shape[1] + margin)
    bottom = min(height, cropped['top'] + cropped['image'].shape[0] + margin)
    image = map_bw[top:bottom, left:right]
    if image.size < 0.5 * map_bw.size or (image > 0).sum() == 0:
        return map_bw, 0, 0
    return image, left, top


def prepare_map(map_path, ds, *, map_threshold=TEMPLATE_MAP_THRESH, search_region=None,
                map_image=None, image_fingerprint=None, include_masks=False):
    """Prepare the same cached binary map for prewarming and live matching."""
    region = tuple(search_region) if search_region is not None else None
    if map_image is not None and not image_fingerprint:
        raise ValueError('内存底图必须提供内容指纹')
    key = ('template', map_path, ds, map_threshold, region, image_fingerprint)
    cached = _map_cache.get(key)
    if cached is None:
        gray = cv2.cvtColor(read_image(map_path) if map_image is None else map_image, cv2.COLOR_BGR2GRAY)
        region_left = region_top = 0
        if region is not None:
            region_left, region_top, region_width, region_height = map(int, region)
            if (min(region_left, region_top) < 0 or min(region_width, region_height) <= 0
                    or region_left + region_width > gray.shape[1]
                    or region_top + region_height > gray.shape[0]):
                raise ValueError('原图搜索区域无效')
            gray = gray[region_top:region_top+region_height, region_left:region_left+region_width]
        if ds < 1:
            gray = cv2.resize(gray, (int(gray.shape[1]*ds), int(gray.shape[0]*ds)),
                              interpolation=cv2.INTER_AREA)
        _, map_bw = cv2.threshold(gray, map_threshold, 255, cv2.THRESH_BINARY)
        map_bw, map_left, map_top = _crop_map(map_bw, ds)
        map_left += region_left * ds
        map_top += region_top * ds
        cached = (map_bw, map_left, map_top)
        retained = sum(value[3].nbytes + value[4].nbytes
                       for cache_key, value in _map_cache.items()
                       if cache_key[0] == 'template' and len(value) == 5)
        if retained + map_bw.size * 8 <= FLOAT_MASK_CACHE_BYTES:
            cached += _float_masks(map_bw)
        # Float masks share the existing key, source invalidation and eviction.
        # The coordinator/prewarmer populates this cache before floor workers run.
        _map_cache[key] = cached
    if include_masks:
        return cached if len(cached) == 5 else (*cached, *_float_masks(cached[0]))
    return cached[:3]


class ScreenshotPreparation:
    """One request's pixels, grouped by exact two-stage resize/threshold semantics."""

    def __init__(self, game):
        self.game = game
        self.gray = None
        self._groups = {}

    def get(self, scale, ds):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('固定尺度必须为有限正数')
        height, width = self.game.shape[:2]
        scaled_width, scaled_height = int(width*scale), int(height*scale)
        target = (int(scaled_width*ds), int(scaled_height*ds))
        if min(scaled_width, scaled_height, *target) <= 0:
            raise ValueError('固定尺度缩放后的截图尺寸过小')
        interpolation = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
        key = (scaled_width, scaled_height, target, ds, interpolation, TEMPLATE_GAME_THRESH)
        if key not in self._groups:
            if self.gray is None:
                self.gray = cv2.cvtColor(self.game, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(self.gray, (scaled_width, scaled_height), interpolation=interpolation)
            if ds < 1:
                scaled = cv2.resize(scaled, target, interpolation=cv2.INTER_AREA)
            _, game_bw = cv2.threshold(scaled, TEMPLATE_GAME_THRESH, 255, cv2.THRESH_BINARY)
            cropped = fog_crop.crop_binary(game_bw)
            if not np.any(game_bw):
                raise ValueError('缩放后的截图没有可匹配像素')
            game_bw = cropped['image']
            white = (game_bw > 0).astype(np.float32)
            black = (game_bw == 0).astype(np.float32)
            self._groups[key] = (white, black, float(white.sum()), float(black.sum()),
                                 cropped['left'], cropped['top'], scaled_width, scaled_height)
        return self._groups[key]


def _match_template(game, map_path, scale, *, map_threshold=TEMPLATE_MAP_THRESH, search_region=None,
                    map_image=None, image_fingerprint=None, prepared=None, expand_search=False, ds=None):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('固定尺度必须为有限正数')
    ds = TEMPLATE_DS if ds is None else ds
    if not np.isfinite(ds) or not 0 < ds <= 1:
        raise ValueError('处理分辨率必须大于0且不超过1')
    inv_ds = 1.0 / ds
    _, map_left, map_top, map_white, map_black = prepare_map(
        map_path, ds, map_threshold=map_threshold, search_region=search_region,
        map_image=map_image, image_fingerprint=image_fingerprint, include_masks=True)
    prepared = ScreenshotPreparation(game) if prepared is None else prepared
    if prepared.game is not game:
        raise ValueError('截图预处理不属于本次截图')
    (game_white, game_black, white_count, black_count, game_left, game_top,
     scaled_width, scaled_height) = prepared.get(scale, ds)
    # Retry buffers are request-local; verified pixels and cached masks stay immutable.
    # Each edge grows by at most half the query or source extent. This bounds the
    # response size and keeps at least half the query overlapping the source on
    # each axis, instead of allowing a tiny fragment to match far outside it.
    pad_y = min(game_white.shape[0], map_white.shape[0]) // 2 if expand_search else 0
    pad_x = min(game_white.shape[1], map_white.shape[1]) // 2 if expand_search else 0
    if pad_x or pad_y:
        padding = ((pad_y, pad_y), (pad_x, pad_x))
        map_white = np.pad(map_white, padding, constant_values=0)
        map_black = np.pad(map_black, padding, constant_values=1)
        map_left -= pad_x
        map_top -= pad_y
    if game_white.shape[0] > map_white.shape[0] or game_white.shape[1] > map_white.shape[1]:
        raise TemplateSizeError('固定尺度下截图可见区域大于原图搜索区域')
    # FFT roundoff can produce tiny negative counts, especially on uniform backgrounds.
    ww = cv2.matchTemplate(map_white, game_white, cv2.TM_CCORR)
    np.clip(ww, 0., white_count, out=ww)
    bb = cv2.matchTemplate(map_black, game_black, cv2.TM_CCORR)
    np.clip(bb, 0., black_count, out=bb)
    wb = np.empty_like(ww)
    np.subtract(white_count, ww, out=wb)
    np.maximum(wb, 0.0, out=wb)
    local_map_white = np.empty_like(ww)
    np.subtract(black_count, bb, out=local_map_white)
    np.maximum(local_map_white, 0.0, out=local_map_white)
    np.add(ww, local_map_white, out=local_map_white)
    np.maximum(local_map_white, 0.0, out=local_map_white)
    local_map_black = np.empty_like(ww)
    np.add(wb, bb, out=local_map_black)
    white_norm = np.empty_like(ww)
    np.multiply(white_count, local_map_white, out=white_norm)
    np.sqrt(white_norm, out=white_norm)
    # Reuse dead arrays within this floor only; cached inputs remain read-only.
    black_norm = local_map_white
    np.multiply(black_count, local_map_black, out=black_norm)
    np.sqrt(black_norm, out=black_norm)
    white_norm[white_norm <= 0] = 1.0
    black_norm[black_norm <= 0] = 1.0
    # Preserve the original float32 ufunc order, including each rounding step.
    np.divide(ww, white_norm, out=ww)
    np.multiply(TEMPLATE_WW, ww, out=ww)
    np.divide(bb, black_norm, out=bb)
    np.multiply(TEMPLATE_BB, bb, out=bb)
    np.add(ww, bb, out=ww)
    np.multiply(white_count, local_map_black, out=local_map_black)
    np.add(local_map_black, 1e-12, out=local_map_black)
    np.sqrt(local_map_black, out=local_map_black)
    np.divide(wb, local_map_black, out=wb)
    np.multiply(TEMPLATE_WB, wb, out=wb)
    np.add(ww, wb, out=ww)
    _, score, _, location = cv2.minMaxLoc(ww)
    maximum_x, maximum_y = ww.shape[1] - 1, ww.shape[0] - 1
    edges = [name for name, hit in (('left', location[0] == 0), ('right', location[0] == maximum_x),
                                    ('top', location[1] == 0), ('bottom', location[1] == maximum_y)) if hit]
    return {'location': (int((location[0]+map_left-game_left)*inv_ds),
                         int((location[1]+map_top-game_top)*inv_ds)),
            'scale': scale, 'score': score, 'size': (scaled_width, scaled_height),
            'search': dict(expanded=bool(expand_search), boundary_edges=edges,
                padding_xy=[pad_x*inv_ds, pad_y*inv_ds],
                origin_bounds_xyxy=[(map_left-game_left)*inv_ds, (map_top-game_top)*inv_ds,
                                    (maximum_x+map_left-game_left)*inv_ds,
                                    (maximum_y+map_top-game_top)*inv_ds])}


def find_best_scale(game, map_path):
    map_gray = preload_map(map_path)
    game_gray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
    game_height, game_width = game_gray.shape
    map_height, map_width = map_gray.shape

    def find_scale(start, end, step):
        best_score, best_scale = -1, 1
        for value in range(start, end, step):
            scale = value/100
            width, height = int(game_width*scale), int(game_height*scale)
            if width > map_width or height > map_height:
                continue
            scaled = cv2.resize(game_gray, (width, height), interpolation=cv2.INTER_LINEAR)
            response = cv2.matchTemplate(map_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            if score > best_score:
                best_score, best_scale = score, scale
        return best_score, best_scale

    score, scale = find_scale(150, 350, 10)
    start, end = int((scale-0.1)*100), int((scale+0.1)*100)
    score, scale = find_scale(start, end+1, 1)
    print(f'\n最佳匹配度：{score:.4f}')
    print(f'最佳缩放：x{scale}')
    return scale
