"""Minimal published records for matcher boundary tests, without maintenance formats.

These tests own low-level records (including deliberately invalid ones). Asset import
and same-canvas geometry are exercised separately through aligned_runtime.publish.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.runtime_resources import BUNDLE_PATH, SCALE_PATH, digest


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def split_records(config):
    template = {key: deepcopy(config[key]) for key in ('perf_mode', 'capture', 'asset_contract') if key in config}
    template.setdefault('perf_mode', 2)
    template.setdefault('asset_contract', 'aligned_floor_images')
    groups, images, scales = {}, {}, {}
    for profile, layouts in config['floor_variants'].items():
        groups[profile] = {}
        map_id, difficulty = profile.split('/')
        entry = scales.setdefault(map_id, {}).setdefault(difficulty, dict(minimum_scale=None, layout_minimum_scales={}))
        layout_scales = {}
        for layout, variants in layouts.items():
            groups[profile][layout] = {}
            for variant, floors in variants.items():
                groups[profile][layout][variant] = {}
                for floor, source in floors.items():
                    record = deepcopy(source)
                    path, scale = record.pop('source_image'), record.pop('saved_scale', None)
                    if path in images and images[path] != record:
                        raise ValueError('同一图片存在冲突处理参数')
                    if layout in layout_scales and repr(layout_scales[layout]) != repr(scale):
                        raise ValueError('同一布局必须共用尺度')
                    layout_scales[layout] = scale
                    images[path] = record
                    groups[profile][layout][variant][floor] = path
        if layout_scales:
            entry['minimum_scale'] = next(iter(layout_scales.values()))
            entry['layout_minimum_scales'] = {layout: value for layout, value in layout_scales.items()
                                               if value != entry['minimum_scale']}
    return dict(template, groups=groups, images=images), dict(maps=scales)


def write_bundle(root, payload):
    root = Path(root)
    directory = root/'configs'
    directory.mkdir(parents=True, exist_ok=True)
    data = deepcopy(payload)
    data['template'], scales = split_records(data['template'])
    raw = json_bytes(scales)
    (root/SCALE_PATH).write_bytes(raw)
    data.setdefault('files', {})[SCALE_PATH] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (root/BUNDLE_PATH).write_bytes(json_bytes(dict(schema_version=2, payload=data, payload_sha256=digest(data))))
    return root/BUNDLE_PATH


def write_template(root, config):
    """Publish test-owned template records, creating the catalog only on first use."""
    root = Path(root)
    path = root/'configs/map_catalog.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        maps = {}
        for key, layouts in config['floor_variants'].items():
            map_id, difficulty_id = key.split('/')
            versions = list(dict.fromkeys(v for items in layouts.values() for v in items))
            floors = [floor for floor in ('2nd', '1st', 'b1')
                      if any(floor in records for items in layouts.values() for records in items.values())]
            definition = dict(id=difficulty_id, name=difficulty_id, clean_root=f'assets/maps/{key}/clean',
                floors=floors, asset_contract='aligned_floor_images', asset_status='ready',
                template_versions=versions, asset_versions=[dict(id=v, name=v,
                    kind='clean' if v == 'clean' else 'route', source_root=f'assets/maps/{key}/clean') for v in versions])
            maps.setdefault(map_id, dict(id=map_id, name=map_id, difficulties=[]))['difficulties'].append(definition)
        path.write_bytes(json_bytes(dict(schema_version=2, maps=list(maps.values()))))
    return write_bundle(root, dict(template=config, recognition={}, route_status={}, files={}))
