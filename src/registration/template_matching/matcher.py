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
    cropped = fog_crop.crop(map_bw)
    margin = max(1, int(round(TEMPLATE_MAP_CROP_MARGIN * ds)))
    left = max(0, cropped['left'] - margin)
    top = max(0, cropped['top'] - margin)
    right = min(width, cropped['left'] + cropped['image'].shape[1] + margin)
    bottom = min(height, cropped['top'] + cropped['image'].shape[0] + margin)
    image = map_bw[top:bottom, left:right]
    if image.size < 0.5 * map_bw.size or (image > 0).sum() == 0:
        return map_bw, 0, 0
    return image, left, top


def _match_template(game, map_path, scale, *, map_threshold=TEMPLATE_MAP_THRESH, search_region=None,
                    map_image=None, image_fingerprint=None):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('固定尺度必须为有限正数')
    ds = TEMPLATE_DS
    inv_ds = 1.0 / ds
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
        _map_cache[key] = (map_bw, map_left, map_top)
        cached = _map_cache[key]
    map_bw, map_left, map_top = cached
    map_white = (map_bw > 0).astype(np.float32)
    map_black = (map_bw == 0).astype(np.float32)
    game_gray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
    height, width = game_gray.shape
    scaled_width, scaled_height = int(width*scale), int(height*scale)
    if min(scaled_width, scaled_height, int(scaled_width*ds), int(scaled_height*ds)) <= 0:
        raise ValueError('固定尺度缩放后的截图尺寸过小')
    interpolation = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
    scaled = cv2.resize(game_gray, (scaled_width, scaled_height), interpolation=interpolation)
    if ds < 1:
        scaled = cv2.resize(scaled, (int(scaled_width*ds), int(scaled_height*ds)),
                            interpolation=cv2.INTER_AREA)
    _, game_bw = cv2.threshold(scaled, TEMPLATE_GAME_THRESH, 255, cv2.THRESH_BINARY)
    cropped = fog_crop.crop(game_bw)
    if not np.any(game_bw):
        raise ValueError('缩放后的截图没有可匹配像素')
    game_bw = cropped['image']
    game_left, game_top = cropped['left'], cropped['top']
    if game_bw.shape[0] > map_white.shape[0] or game_bw.shape[1] > map_white.shape[1]:
        raise ValueError('固定尺度下截图可见区域大于原图搜索区域')
    game_white = (game_bw > 0).astype(np.float32)
    game_black = (game_bw == 0).astype(np.float32)
    white_count, black_count = float(game_white.sum()), float(game_black.sum())
    # FFT roundoff can produce tiny negative counts, especially on uniform backgrounds.
    ww = np.clip(cv2.matchTemplate(map_white, game_white, cv2.TM_CCORR), 0., white_count)
    bb = np.clip(cv2.matchTemplate(map_black, game_black, cv2.TM_CCORR), 0., black_count)
    wb = np.maximum(white_count - ww, 0.0)
    local_map_white = np.maximum(ww + np.maximum(black_count - bb, 0.0), 0.0)
    local_map_black = wb + bb
    white_norm = np.sqrt(white_count * local_map_white)
    black_norm = np.sqrt(black_count * local_map_black)
    white_norm[white_norm <= 0] = 1.0
    black_norm[black_norm <= 0] = 1.0
    response = (TEMPLATE_WW*(ww/white_norm) + TEMPLATE_BB*(bb/black_norm)
                + TEMPLATE_WB*(wb/np.sqrt(white_count*local_map_black + 1e-12)))
    _, score, _, location = cv2.minMaxLoc(response)
    return {'location': (int((location[0]+map_left-game_left)*inv_ds),
                         int((location[1]+map_top-game_top)*inv_ds)),
            'scale': scale, 'score': score, 'size': (scaled_width, scaled_height)}


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
