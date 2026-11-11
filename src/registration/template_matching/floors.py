"""Verified single-floor sources; matrices refer to pixels of the selected image."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import atexit
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from src.app_catalog import PROJECT_ROOT, MapCatalog, CLEAN_VERSION_ID
from src.floor_definitions import FLOORS, DEFAULT_FLOORS
from . import resources, matcher, walls

DECODED_IMAGE_LIMIT = 8
DECODED_IMAGE_BYTES = 256 * 1024 * 1024
_images = OrderedDict()
_retained_keys = frozenset()
_fixed_session = None


def _close_fixed_session():
    global _fixed_session, _retained_keys
    with resources.resource_lock:
        if _fixed_session is not None:
            _fixed_session.close()
            _fixed_session = None
        _retained_keys = frozenset()


atexit.register(_close_fixed_session)


def fixed_session():
    """Freeze configuration once; dependent bytes load and verify lazily per version."""
    global _fixed_session
    from .frozen import FrozenResources
    with resources.resource_lock:
        if _fixed_session is not None and _fixed_session.origin != PROJECT_ROOT:
            _close_fixed_session()
        if _fixed_session is None:
            _fixed_session = FrozenResources(PROJECT_ROOT, _read_config())
        return _fixed_session


def _read_config():
    from src.runtime_resources import template_config
    config = template_config(PROJECT_ROOT)
    if config is None:
        raise ValueError('缺少运行资源配置，请重新发布素材')
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def source_path(value, *, root=None):
    root = PROJECT_ROOT.resolve() if root is None else root
    # Strict resolution rejects missing/looping links and avoids pathlib's extra
    # non-strict stat. Fingerprint verification still checks the resolved file.
    path = (root / value).resolve(strict=True)
    path.relative_to(root)
    return path


def profile_floors(profile_key, layout=None):
    path = PROJECT_ROOT / 'configs/map_catalog.json'
    return MapCatalog(path).profile(*profile_key.split('/')).floors_for_layout(layout)


def _source_records(config, expected, profile_key, layout, variant_id, *, verify_dependencies):
    entry = config.get('floor_variants', {}).get(profile_key, {}).get(layout, {}).get(variant_id)
    if entry is None or set(entry) != set(expected):
        raise ValueError('当前版本缺少完整的单层图片与尺度配置，请重新导入')
    verified = set()
    dependencies = []
    def verify(value, digest):
        # Deduplicate this batch; its verifier controls reuse across requests.
        key = (value, digest)
        if key not in verified:
            dependencies.append((value, digest))
            verified.add(key)
    for floor, item in entry.items():
        try:
            scale = float(item['saved_scale'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{floor}: 单层图片尚未保存有效尺度') from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'{floor}: 单层图片尚未保存有效尺度')
        verify(item['source_image'], item['source_sha256'])
    verify_dependencies(dependencies)
    # Only small request-owned records leave the cache. Unrelated layouts are not copied.
    return deepcopy({f: entry[f] for f in expected}), deepcopy({key: config[key] for key in ('perf_mode', 'capture', 'asset_contract') if key in config})



def sources(profile_key, layout, variant_id):
    """Audit the current published records; runtime uses FrozenSourceSnapshot."""
    with resources.resource_lock:
        config = _read_config()
        root = PROJECT_ROOT.resolve()
        return _source_records(config, profile_floors(profile_key, layout), profile_key, layout, variant_id,
            verify_dependencies=lambda dependencies: resources.verify_fingerprints(
                dependencies, lambda value: source_path(value, root=root)))


class SourceSnapshot:
    """Request-local records; live audits are the default, runtime injects a session.

    Callers hold resource_lock over candidate matching and final validation. Standalone
    reference/pan loads use load() instead so a previous snapshot cannot reuse a pose.
    """

    def __init__(self, profile_key, layout, variant_id, *, _session=None):
        self.key = (profile_key, layout, variant_id)
        self._session = _session
        reader = sources if _session is None else _session.records
        records, self.config = reader(*self.key)
        self.identity = fingerprint(records)
        self.entries = records

    def load(self, floor):
        if floor not in self.entries:
            raise ValueError('尚未取得单层匹配候选')
        with resources.resource_lock:
            return _load_verified(self.entries[floor], self.config, frozen=self._session)

    def validate(self):
        reader = sources if self._session is None else self._session.records
        current, config = reader(*self.key)
        if (fingerprint(current) != self.identity
                or config.get('perf_mode') != self.config.get('perf_mode')
                or config.get('capture') != self.config.get('capture')):
            raise ValueError('匹配期间单层配置已改变，请重新截图匹配')


class FrozenSourceSnapshot(SourceSnapshot):
    """Runtime request view of verified bytes and configuration fixed until restart."""

    def __init__(self, profile_key, layout, variant_id):
        with resources.resource_lock:
            super().__init__(profile_key, layout, variant_id,
                             _session=fixed_session())


def load(profile_key, layout, variant_id, floor, *, expected_fingerprint=None):
    """Load one layer from this process's fixed resources, checking its pose identity."""
    if floor not in FLOORS:
        raise ValueError('尚未取得单层匹配候选')
    with resources.resource_lock:
        snapshot = FrozenSourceSnapshot(profile_key, layout, variant_id)
        if floor not in snapshot.entries:
            raise ValueError('当前难度不包含该楼层')
        identity = fingerprint(snapshot.entries[floor])
        if expected_fingerprint is not None and identity != expected_fingerprint:
            raise ValueError('单层素材或尺度已改变，请重新截图匹配')
        return snapshot.load(floor)


def working_set_fits(records):
    """Estimate verified BGR + BGRA pixels before decoding the active sources."""
    sizes = {(item['source_image'], fingerprint(item)): item['size'][0] * item['size'][1] * 7
             for item in records}
    return len(sizes) <= DECODED_IMAGE_LIMIT and sum(sizes.values()) <= DECODED_IMAGE_BYTES


def retain_working_set(records, *, frozen):
    """Caller holds resource_lock; reference/prewarm reads cannot displace these sources."""
    global _retained_keys
    keys = frozenset((str(source_path(item['source_image']) if frozen is None
                         else frozen.path(item['source_image'], item['source_sha256'])), fingerprint(item))
                     for item in records)
    if len(keys) > DECODED_IMAGE_LIMIT:
        raise ValueError('当前配准工作集超出解码缓存容量')
    _retained_keys = keys


def trim_uncached_templates():
    """After workers join, drop masks whose decoded source was returned uncached."""
    paths = {key[0] for key in _images}
    for key in list(matcher._map_cache):
        if key[0] in ('template', 'gray', 'walls') and key[1] not in paths:
            del matcher._map_cache[key]


def _load_verified(item, config, *, cancelled=lambda: False, frozen=None):
    """Decode/cache pixels from records verified by the owning snapshot."""
    identity = fingerprint(item)
    path = (source_path(item['source_image']) if frozen is None
            else frozen.path(item['source_image'], item['source_sha256']))
    key = (str(path), identity)
    cached = _images.get(key)
    if cached is None:
        payload = path if frozen is None else BytesIO(frozen.read(item['source_image'], item['source_sha256']))
        with Image.open(payload) as source:
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
        cached = (image, rgba)
        _images[key] = cached
        # Keep a bounded working set; changed files must not reuse matcher pixels.
        for cache_key in list(matcher._map_cache):
            if str(path) in cache_key:
                del matcher._map_cache[cache_key]
        while (len(_images) > DECODED_IMAGE_LIMIT
               or sum(image.nbytes + rgba.nbytes for image, rgba in _images.values()) > DECODED_IMAGE_BYTES):
            # When all slots are retained, an unrelated reference is returned
            # to its caller without caching it. A protected image exceeding the
            # byte budget is also returned uncached; request-owned pixels stay valid.
            old_key = next((k for k in _images if k not in _retained_keys), key)
            del _images[old_key]
            for cache_key in list(matcher._map_cache):
                if old_key[0] in cache_key:
                    del matcher._map_cache[cache_key]
    if key in _images:
        _images.move_to_end(key)
    image, rgba = cached
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
        snapshot = FrozenSourceSnapshot(profile_key, layout, variant_id)
        mode = int(snapshot.config.get('perf_mode', 2))
        if not 0 <= mode < len(matcher.PERF_MODES):
            raise ValueError('性能档位无效')
        ds = matcher.PERF_MODES[mode]['ds']
    finally:
        resources.resource_lock.release()
    for floor in snapshot.entries:
        if cancelled() or not resources.resource_lock.acquire(blocking=False):
            return False
        try:
            if cancelled():
                return False
            loaded = _load_verified(snapshot.entries[floor], snapshot.config, cancelled=cancelled,
                                    frozen=snapshot._session)
            if loaded is None or cancelled():
                return False
            item, path, _, image, _ = loaded
            if (str(path), fingerprint(item)) not in _images:
                continue  # Uncached prewarm pixels must not leave orphaned template masks.
            matcher.prepare_map(str(path), ds, map_threshold=item.get('map_threshold', matcher.TEMPLATE_MAP_THRESH),
                                map_image=image, image_fingerprint=fingerprint(item))
            if variant_id == CLEAN_VERSION_ID and not cancelled():
                walls.prepare_map(str(path), ds, map_image=image, image_fingerprint=fingerprint(item))
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
