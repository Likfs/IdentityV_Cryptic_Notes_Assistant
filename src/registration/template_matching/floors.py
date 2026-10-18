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

from src.app_catalog import PROJECT_ROOT, TEMPLATE_VERSION_NAMES, MapCatalog
from . import resources, matcher

FLOORS = ('2nd', '1st', 'b1')
_images = OrderedDict()
_config_cache = {}
_fixed_session = None


def _close_fixed_session():
    global _fixed_session
    with resources.resource_lock:
        if _fixed_session is not None:
            _fixed_session.close()
            _fixed_session = None


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


def _read_config(*, offline=False):
    """Private parsed document; callers must never expose or mutate this object."""
    if not offline:
        from src.runtime_resources import template_config
        config = template_config(PROJECT_ROOT)
        if config is not None:
            return config
    path = PROJECT_ROOT / 'maintenance/configs/template_matching.json'
    # Live audits read current bytes; the runtime session calls this only at creation.
    raw = path.read_bytes()
    if _config_cache.get('path') != path or _config_cache.get('raw') != raw:
        # Decode only changed bytes; keep strict UTF-8/BOM validation unchanged.
        config = json.loads(raw.decode('utf-8'))
        _config_cache.clear()
        _config_cache.update(path=path, raw=raw, config=config)
    return _config_cache['config']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def source_path(value, *, root=None):
    root = PROJECT_ROOT.resolve() if root is None else root
    # Strict resolution rejects missing/looping links and avoids pathlib's extra
    # non-strict stat. Fingerprint verification still checks the resolved file.
    path = (root / value).resolve(strict=True)
    path.relative_to(root)
    return path


def profile_floors(profile_key):
    path = PROJECT_ROOT / 'configs/map_catalog.json'
    if not path.exists():
        return FLOORS  # Standalone legacy configurations use all three floors.
    return MapCatalog(path).profile(*profile_key.split('/')).floors


def _source_records(config, expected, profile_key, layout, variant_id, *, require_scale, verify_dependencies):
    if variant_id not in TEMPLATE_VERSION_NAMES:
        raise ValueError('模板模式不支持所选版本')
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
        pending = item.get('scale_status') == 'pending' and item.get('saved_scale') is None
        if pending:
            if require_scale:
                raise ValueError(f'{floor}: 屏幕尺度待标定，请补充可靠保存尺度或实机截图')
        else:
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
        reference_floor = item.get('reference_floor_variant')
        if reference_floor:
            variants = config['floor_variants'][profile_key][layout]
            source = variants.get(reference_floor.get('variant_id'), {}).get(floor)
            if (reference_floor.get('floor') != floor or source is None
                    or fingerprint(source) != reference_floor.get('sha256')):
                raise ValueError('底图版对应的原底图尺度记录已改变，请重新导入')
    verify_dependencies(dependencies)
    # Only small request-owned records leave the cache. Unrelated layouts are not copied.
    return deepcopy({f: entry[f] for f in expected}), deepcopy({key: config[key] for key in ('perf_mode', 'capture') if key in config})



def sources(profile_key, layout, variant_id, *, require_scale=True, offline=False):
    """Live maintenance audit; runtime uses FrozenSourceSnapshot instead."""
    with resources.resource_lock:
        config = _read_config(offline=offline)
        root = PROJECT_ROOT.resolve()
        return _source_records(config, profile_floors(profile_key), profile_key, layout, variant_id,
            require_scale=require_scale,
            verify_dependencies=lambda dependencies: resources.verify_fingerprints(
                dependencies, lambda value: source_path(value, root=root)))


class SourceSnapshot:
    """Request-local records; live audits are the default, runtime injects a session.

    Callers hold resource_lock over candidate matching and final validation. Standalone
    reference/pan loads use load() instead so a previous snapshot cannot reuse a pose.
    """

    def __init__(self, profile_key, layout, variant_id, *, allow_pending=False, _session=None):
        self.key = (profile_key, layout, variant_id)
        self.allow_pending = allow_pending
        self._session = _session
        reader = sources if _session is None else _session.records
        records, self.config = reader(*self.key, require_scale=not allow_pending)
        self.identity = fingerprint(records)
        self.pending = [f for f, item in records.items()
                        if item.get('scale_status') == 'pending' and item.get('saved_scale') is None]
        self.entries = {f: item for f, item in records.items() if f not in self.pending}
        if not self.entries:
            raise ValueError('屏幕尺度待标定，请补充可靠保存尺度或实机截图')

    def load(self, floor):
        if floor not in self.entries:
            raise ValueError('尚未取得单层匹配候选')
        with resources.resource_lock:
            return _load_verified(self.entries[floor], self.config, frozen=self._session)

    def validate(self):
        reader = sources if self._session is None else self._session.records
        current, config = reader(*self.key, require_scale=not self.allow_pending)
        if (fingerprint(current) != self.identity
                or config.get('perf_mode') != self.config.get('perf_mode')
                or config.get('capture') != self.config.get('capture')):
            raise ValueError('匹配期间单层配置已改变，请重新截图匹配')


class FrozenSourceSnapshot(SourceSnapshot):
    """Runtime request view of verified bytes and configuration fixed until restart."""

    def __init__(self, profile_key, layout, variant_id, *, allow_pending=False):
        with resources.resource_lock:
            super().__init__(profile_key, layout, variant_id, allow_pending=allow_pending,
                             _session=fixed_session())


def load(profile_key, layout, variant_id, floor, *, expected_fingerprint=None):
    """Load one layer from this process's fixed resources, checking its pose identity."""
    if floor not in FLOORS:
        raise ValueError('尚未取得单层匹配候选')
    with resources.resource_lock:
        snapshot = FrozenSourceSnapshot(profile_key, layout, variant_id, allow_pending=True)
        if floor not in snapshot.entries:
            raise ValueError('当前难度不包含该楼层')
        identity = fingerprint(snapshot.entries[floor])
        if expected_fingerprint is not None and identity != expected_fingerprint:
            raise ValueError('单层素材或尺度已改变，请重新截图匹配')
        return snapshot.load(floor)


def _load_verified(item, config, *, cancelled=lambda: False, frozen=None):
    """Decode/cache pixels from records verified by the owning snapshot."""
    identity = fingerprint(item)
    path = (source_path(item['source_image']) if frozen is None
            else frozen.path(item['source_image'], item['source_sha256']))
    key = (str(path), identity)
    if key not in _images:
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
        snapshot = FrozenSourceSnapshot(profile_key, layout, variant_id, allow_pending=True)
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


def check(profile_key, layout, variant_id, *, require_scale=True):
    """Validate fingerprints and dimensions without constructing all RGBA canvases."""
    with resources.resource_lock:
        entries, _ = sources(profile_key, layout, variant_id, require_scale=require_scale)
        for item in entries.values():
            with Image.open(source_path(item['source_image'])) as image:
                if list(image.size) != item['size']:
                    raise ValueError('单层图片尺寸已改变，请重新导入')
        return len(entries)
