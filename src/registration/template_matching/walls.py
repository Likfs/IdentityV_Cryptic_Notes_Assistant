"""Fixed-scale directional walls; crop zero query borders without losing poses."""
import cv2
import numpy as np
from . import matcher

FIELD_CACHE_BYTES = 64 * 1024 * 1024


def extract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 65)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ignore = cv2.dilate(((hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 110)).astype(np.uint8),
                        np.ones((3, 3), np.uint8)) > 0
    edges[ignore] = 0
    # A 3x3 Sobel fits int16; diagonal orientation only needs the signs.
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    ax, ay = np.abs(gx), np.abs(gy)
    vertical, horizontal = ax >= 2 * ay, ay >= 2 * ax
    diagonal = ~(vertical | horizontal)
    different = (gx < 0) != (gy < 0)
    masks = (vertical, horizontal, diagonal & different, diagonal & ~different)
    kernels = (np.ones((7, 1), np.uint8), np.ones((1, 7), np.uint8),
               np.eye(5, dtype=np.uint8), np.fliplr(np.eye(5, dtype=np.uint8)).copy())
    channels = []
    for mask, kernel in zip(masks, kernels):
        line = cv2.morphologyEx(((edges > 0) & mask).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        line = cv2.morphologyEx(line, cv2.MORPH_OPEN, kernel)
        line[ignore] = 0
        channels.append(line.astype(np.float32))
    return channels


def prepare_map(path, ds, *, map_image=None, image_fingerprint=None, map_threshold=None):
    """Coordinator-only cache writes; source eviction clears the same keyed cache.

    Over-budget fields stay request-local and are passed to the worker once.
    """
    if not np.isfinite(ds) or not 0 < ds <= 1:
        raise ValueError('处理分辨率必须大于0且不超过1')
    if map_image is not None and not image_fingerprint:
        raise ValueError('内存底图必须提供内容指纹')
    key = ('walls', str(path), ds, image_fingerprint)
    cached = matcher._map_cache.get(key)
    if cached is not None:
        return cached
    image = matcher.read_image(path) if map_image is None else map_image
    if ds < 1:
        width, height = int(image.shape[1] * ds), int(image.shape[0] * ds)
        if min(width, height) <= 0:
            raise ValueError('处理后的底图尺寸过小')
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    fields = []
    for channel in extract(image):
        if np.any(channel):
            distance = cv2.distanceTransform((channel == 0).astype(np.uint8), cv2.DIST_L2, 3)
            field = np.exp(-.5 * distance ** 2).astype(np.float32)
        else:
            field = np.zeros_like(channel)
        field.flags.writeable = False
        fields.append(field)
    cached = tuple(fields)
    retained = sum(sum(field.nbytes for field in value)
                   for cache_key, value in matcher._map_cache.items() if cache_key[0] == 'walls')
    if retained + sum(field.nbytes for field in cached) <= FIELD_CACHE_BYTES:
        matcher._map_cache[key] = cached
    return cached


def _field_crop(field, left, top, width, height):
    result = np.zeros((height, width), np.float32)
    sx, sy = max(0, left), max(0, top)
    ex, ey = min(field.shape[1], left + width), min(field.shape[0], top + height)
    if ex > sx and ey > sy:
        result[sy-top:ey-top, sx-left:ex-left] = field[sy:ey, sx:ex]
    return result


def correlate(query, fields, bounds):
    """All original origins remain searchable, including negative coordinates."""
    x0, y0, x1, y1 = map(int, bounds)
    shape = (y1-y0+1, x1-x0+1)
    response = np.zeros(shape, np.float32)
    count = sum(float(channel.sum()) for channel in query)
    if not count:
        raise ValueError('截图中没有可匹配的连续墙线，请增加可见地图结构后重试')
    if not any(np.any(field) for field in fields):
        raise ValueError('底图中没有可匹配的连续墙线')
    for channel, field in zip(query, fields):
        yy, xx = np.where(channel > 0)
        if not len(xx):
            continue
        left, right, top, bottom = xx.min(), xx.max()+1, yy.min(), yy.max()+1
        template = channel[top:bottom, left:right]
        target = _field_crop(field, x0+left, y0+top,
                             shape[1]+right-left-1, shape[0]+bottom-top-1)
        response += cv2.matchTemplate(target, template, cv2.TM_CCORR)
    response /= count
    return response, int(count)


def match(game, path, scale, *, ds, map_threshold=matcher.TEMPLATE_MAP_THRESH,
          map_image=None, image_fingerprint=None, prepared=None, wall_fields=None,
          expand_search=False):
    # Binary preparation defines precisely the previously tested bounded search
    # geometry. Its rewards/penalties do not participate in wall scoring.
    if not np.isfinite(ds) or not 0 < ds <= 1:
        raise ValueError('处理分辨率必须大于0且不超过1')
    prepared = matcher.ScreenshotPreparation(game) if prepared is None else prepared
    if prepared.game is not game:
        raise ValueError('截图预处理不属于本次截图')
    white, _, _, _, gx, gy, sw, sh = prepared.get(scale, ds)
    bw, mx, my = matcher.prepare_map(path, ds, map_threshold=map_threshold,
        map_image=map_image, image_fingerprint=image_fingerprint)
    px = min(white.shape[1], bw.shape[1]) // 2 if expand_search else 0
    py = min(white.shape[0], bw.shape[0]) // 2 if expand_search else 0
    maximum_x = bw.shape[1]+2*px-white.shape[1]
    maximum_y = bw.shape[0]+2*py-white.shape[0]
    if min(maximum_x, maximum_y) < 0:
        raise matcher.TemplateSizeError('固定尺度下截图可见区域大于原图搜索区域')
    x0, y0 = int(mx-px-gx), int(my-py-gy)
    fields = wall_fields if wall_fields is not None else prepare_map(path, ds,
        map_image=map_image, image_fingerprint=image_fingerprint)
    large = cv2.resize(game, (sw, sh), interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA)
    small = (cv2.resize(large, (int(sw*ds), int(sh*ds)), interpolation=cv2.INTER_AREA)
             if ds < 1 else large)
    response, count = correlate(extract(small), fields, (x0, y0, x0+maximum_x, y0+maximum_y))
    _, score, _, (x, y) = cv2.minMaxLoc(response)
    other = response.copy()
    other[max(0, y-8):y+9, max(0, x-8):x+9] = -1
    edges = [name for name, hit in (('left', x == 0), ('right', x == maximum_x),
                                    ('top', y == 0), ('bottom', y == maximum_y)) if hit]
    return dict(location=(int((x+x0)/ds), int((y+y0)/ds)), scale=scale, score=score, size=(sw, sh),
        wall_match=dict(wall_pixels=count, runner_up_gap=score-cv2.minMaxLoc(other)[1]),
        search=dict(expanded=bool(expand_search), boundary_edges=edges, padding_xy=[px/ds, py/ds],
            origin_bounds_xyxy=[x0/ds, y0/ds, (x0+maximum_x)/ds, (y0+maximum_y)/ds]))
