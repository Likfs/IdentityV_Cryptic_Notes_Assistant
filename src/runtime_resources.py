"""Published runtime data. Detailed calibration evidence belongs to offline export."""
import hashlib
import json
from pathlib import Path

from src.paths import PROJECT_ROOT, published_resource_path
from src.scale_config import SCALE_PATH, load_scales, profile_settings, profile_keys, validate_scales

BUNDLE_PATH = 'configs/runtime_resources.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def read_bundle(root=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    path = root/BUNDLE_PATH
    if not path.exists():
        return None  # Startup discovery may publish a new bundle.
    data = json.loads(path.read_text(encoding='utf-8'))
    payload = data.get('payload')
    if (data.get('schema_version') != 2 or not isinstance(payload, dict)
            or digest(payload) != data.get('payload_sha256')):
        raise ValueError('运行资源配置不完整或已改变，请重新导出')
    return payload


def expand_template(template, scales):
    """Resolve grouped user scales to the matcher's existing per-image records."""
    validate_scales(scales)
    images, groups = template['images'], template['groups']
    if profile_keys(scales) != set(groups):
        raise ValueError('尺度配置与地图／难度清单不一致，请重新发布')
    used = {path for layouts in groups.values() for versions in layouts.values()
            for floors in versions.values() for path in floors.values()}
    if set(images) != used:
        raise ValueError('图片清单与地图分组不一致，请重新发布')
    config = {k: template[k] for k in ('perf_mode', 'capture', 'asset_contract') if k in template}
    resolved, assigned = {}, {}
    for profile, layouts in groups.items():
        map_id, difficulty = profile.split('/')
        entry = profile_settings(scales, map_id, difficulty, ready=bool(layouts))
        resolved[profile] = {}
        for layout, versions in layouts.items():
            scale = entry.get('layout_minimum_scales', {}).get(layout, entry['minimum_scale'])
            resolved[profile][layout] = {}
            for version, floors in versions.items():
                records = {}
                for floor, path in floors.items():
                    metadata = images[path]
                    if 'saved_scale' in metadata or 'source_image' in metadata or 'scale' in metadata:
                        raise ValueError('自动资源清单不能重复保存图片路径或尺度')
                    if path in assigned and assigned[path] != scale:
                        raise ValueError('同一图片引用了冲突尺度，请检查地图分组')
                    assigned[path] = scale
                    records[floor] = dict(metadata, source_image=path, saved_scale=scale)
                resolved[profile][layout][version] = records
    config['floor_variants'] = resolved
    return config


def _published_scales(root, bundle):
    path = (root/SCALE_PATH).resolve(strict=True)
    path.relative_to(root.resolve())
    raw = path.read_bytes()
    record = bundle['files'][SCALE_PATH]
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('尺度配置与已发布资源不一致，请正常重启同步或重新导出')
    return load_scales(root, raw=raw)


def template_config(root=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    bundle = read_bundle(root)
    if bundle is None:
        return None
    if bundle['template'].get('asset_contract') != 'aligned_floor_images':
        raise ValueError('运行资源必须使用 aligned_floor_images，请重新发布素材')
    return expand_template(bundle['template'], _published_scales(root, bundle))


def recognition_profile(clean_root):
    try:
        relative = Path(clean_root).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return None
    if not relative.startswith('assets/maps/'):
        return None
    bundle = read_bundle()
    if bundle is None:
        return None
    profiles = bundle['recognition']
    if relative not in profiles:
        raise ValueError('运行资源包未包含当前识别配置')
    source = profiles[relative]
    if not source.get('source_images') or 'profile_key' not in source:
        raise ValueError('识别资源缺少当前地图／难度分组，请重新发布')
    scales = _published_scales(PROJECT_ROOT, bundle)
    map_id, difficulty = source['profile_key'].split('/')
    entry = profile_settings(scales, map_id, difficulty, ready=bool(source['targets']))
    targets = {}
    for name, metadata in source['targets'].items():
        if 'scales' in metadata or 'scale_modes' in metadata:
            raise ValueError('自动识别资源清单不能重复保存尺度，请重新发布')
        layout = name.split('/')[0]
        value = entry.get('layout_minimum_scales', {}).get(layout, entry['minimum_scale'])
        targets[name] = dict(metadata, scales=[value])
    return dict(source, targets=targets)


def runtime_files(root=None):
    """Validate the published package only; never follow offline evidence paths."""
    root = (Path(root) if root is not None else PROJECT_ROOT).resolve()
    bundle = read_bundle(root)
    if bundle is None:
        raise ValueError('缺少运行资源配置，请先执行离线导出')
    files = []
    for name, record in bundle['files'].items():
        path = published_resource_path(root, name)
        if (not path.is_file() or path.stat().st_size != record['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
            raise ValueError(f'运行资源指纹不一致：{name}')
        files.append(path)
    return sorted([*files, root/BUNDLE_PATH])
