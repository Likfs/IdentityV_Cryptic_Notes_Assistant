"""Verified single-floor sources; matrices refer to pixels of the selected image."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from src.app_catalog import PROJECT_ROOT, TEMPLATE_VERSION_NAMES
from . import resources, matcher

FLOORS = ('2nd', '1st', 'b1')
_images = OrderedDict()
_config_cache = {}


def _read_config():
    """Private parsed document; callers must never expose or mutate this object."""
    path = PROJECT_ROOT / 'configs/template_matching.json'
    # Read at both request boundaries, including same-size/same-mtime edits.
    text = path.read_text(encoding='utf-8')
    if _config_cache.get('path') != path or _config_cache.get('text') != text:
        config = json.loads(text)
        _config_cache.clear()
        _config_cache.update(path=path, text=text, config=config)
    return _config_cache['config']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def source_path(value, *, root=None):
    root = PROJECT_ROOT.resolve() if root is None else root
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def sources(profile_key, layout, variant_id):
    if variant_id not in TEMPLATE_VERSION_NAMES:
        raise ValueError('模板模式不支持所选版本')
    with resources.resource_lock:
        config = _read_config()
        entry = config.get('floor_variants', {}).get(profile_key, {}).get(layout, {}).get(variant_id)
        if entry is None or set(entry) != set(FLOORS):
            raise ValueError('当前版本缺少完整的单层图片与尺度配置，请重新导入')
        root = PROJECT_ROOT.resolve()
        verified = set()
        def verify(value, digest):
            # Shared dependencies are checked once per boundary, never across requests.
            key = (value, digest)
            if key not in verified:
                resources.verify_fingerprint(source_path(value, root=root), digest)
                verified.add(key)
        for floor, item in entry.items():
            try:
                scale = float(item['saved_scale'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{floor}: 单层图片尚未保存有效尺度') from exc
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{floor}: 单层图片尚未保存有效尺度')
            verify(item['source_image'], item['source_sha256'])
            for dependency in item.get('dependencies', []):
                verify(dependency['path'], dependency['sha256'])
            reference = item.get('reference_entry_key')
            if reference and fingerprint(config['entries'][reference]) != item['reference_entry_sha256']:
                raise ValueError('单层尺度对应的参考记录已改变，请重新导入')
        # Only small request-owned records leave the cache. Unrelated layouts are not copied.
        return deepcopy(entry), deepcopy({key: config[key] for key in ('perf_mode', 'capture') if key in config})


class SourceSnapshot:
    """Request-local verified records; validate again before publishing a match.

    Callers hold resource_lock over candidate matching and final validation. Standalone
    reference/pan loads use load() instead so a previous snapshot cannot reuse a pose.
    """

    def __init__(self, profile_key, layout, variant_id):
        self.key = (profile_key, layout, variant_id)
        self.entries, self.config = sources(*self.key)
        self.identity = fingerprint(self.entries)

    def load(self, floor):
        if floor not in FLOORS:
            raise ValueError('尚未取得单层匹配候选')
        with resources.resource_lock:
            return _load_verified(self.entries[floor], self.config)

    def validate(self):
        current, config = sources(*self.key)
        if (fingerprint(current) != self.identity
                or config.get('perf_mode') != self.config.get('perf_mode')
                or config.get('capture') != self.config.get('capture')):
            raise ValueError('匹配期间单层配置已改变，请重新截图匹配')


def load(profile_key, layout, variant_id, floor, *, expected_fingerprint=None):
    """Load only one layer; reject a resource/config change since its pose was computed."""
    if floor not in FLOORS:
        raise ValueError('尚未取得单层匹配候选')
    with resources.resource_lock:
        snapshot = SourceSnapshot(profile_key, layout, variant_id)
        identity = fingerprint(snapshot.entries[floor])
        if expected_fingerprint is not None and identity != expected_fingerprint:
            raise ValueError('单层素材或尺度已改变，请重新截图匹配')
        return snapshot.load(floor)


def _load_verified(item, config, *, cancelled=lambda: False):
    """Decode/cache pixels from records verified by the owning snapshot."""
    identity = fingerprint(item)
    path = source_path(item['source_image'])
    key = (str(path), identity)
    if key not in _images:
        with Image.open(path) as source:
            rgba = cv2.cvtColor(np.array(source.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
        if list(rgba.shape[1::-1]) != item['size']:
            raise ValueError('单层图片尺寸已改变，请重新导入')
        image = rgba[:, :, :3].copy()
        image[rgba[:, :, 3] == 0] = 0
        foreground = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > item.get('map_threshold', 3)
        if item.get('preserve_colored_routes', True):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            foreground |= (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 60)
        rgba[:, :, 3] = np.where(foreground, rgba[:, :, 3], 0)
        if cancelled():
            return None
        _images[key] = (image, rgba)
        # Keep a bounded working set; changed files must not reuse matcher pixels.
        for cache_key in list(matcher._map_cache):
            if str(path) in cache_key:
                del matcher._map_cache[cache_key]
        while len(_images) > 3:
            old_key, _ = _images.popitem(last=False)
            for cache_key in list(matcher._map_cache):
                if old_key[0] in cache_key:
                    del matcher._map_cache[cache_key]
    _images.move_to_end(key)
    image, rgba = _images[key]
    return item, path, config, image, rgba


def prewarm(profile_key, layout, variant_id, *, cancelled=lambda: False):
    """Prepare bounded source/template caches only, never a pose or a UI result.

    Release the resource lock between floors. An active live match takes precedence;
    it populates the same caches itself, so prewarming need not wait for it.
    """
    if cancelled() or not resources.resource_lock.acquire(blocking=False):
        return False
    try:
        if cancelled():
            return False
        snapshot = SourceSnapshot(profile_key, layout, variant_id)
        mode = int(snapshot.config.get('perf_mode', 2))
        if not 0 <= mode < len(matcher.PERF_MODES):
            raise ValueError('性能档位无效')
        ds = matcher.PERF_MODES[mode]['ds']
    finally:
        resources.resource_lock.release()
    for floor in FLOORS:
        if cancelled() or not resources.resource_lock.acquire(blocking=False):
            return False
        try:
            if cancelled():
                return False
            loaded = _load_verified(snapshot.entries[floor], snapshot.config, cancelled=cancelled)
            if loaded is None or cancelled():
                return False
            item, path, _, image, _ = loaded
            matcher.prepare_map(str(path), ds, map_threshold=item.get('map_threshold', matcher.TEMPLATE_MAP_THRESH),
                                map_image=image, image_fingerprint=fingerprint(item))
        finally:
            resources.resource_lock.release()
    if cancelled() or not resources.resource_lock.acquire(blocking=False):
        return False
    try:
        snapshot.validate()
        return not cancelled()
    finally:
        resources.resource_lock.release()


def check(profile_key, layout, variant_id):
    """Validate fingerprints and dimensions without constructing all RGBA canvases."""
    with resources.resource_lock:
        entries, _ = sources(profile_key, layout, variant_id)
        for item in entries.values():
            with Image.open(source_path(item['source_image'])) as image:
                if list(image.size) != item['size']:
                    raise ValueError('单层图片尺寸已改变，请重新导入')
        return len(entries)
