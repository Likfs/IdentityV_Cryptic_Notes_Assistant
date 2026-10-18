"""Published runtime data. Detailed calibration evidence belongs to offline export."""
import hashlib
import json
import math
from pathlib import Path

from src.paths import PROJECT_ROOT

BUNDLE_PATH = 'configs/runtime_resources.json'
SCALE_PATH = 'configs/scale_config.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def read_bundle(root=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    path = root/BUNDLE_PATH
    if not path.exists():
        return None  # Standalone maintenance/test inputs use the evidence format.
    data = json.loads(path.read_text(encoding='utf-8'))
    payload = data.get('payload')
    if (data.get('schema_version') != 2 or not isinstance(payload, dict)
            or digest(payload) != data.get('payload_sha256')):
        raise ValueError('运行资源配置不完整或已改变，请重新导出')
    return payload


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'尺度配置存在重复键：{key}')
        result[key] = value
    return result


def expand_template(template, scales):
    """Adapt the flat public scale table to unchanged matcher records, without rounding."""
    if not isinstance(scales, dict) or set(scales) != {'maps'} or not isinstance(scales['maps'], dict):
        raise ValueError('尺度配置必须为 maps 下每张图片一个 scale')
    images = template['images']
    groups = template['groups']
    used = {path for layouts in groups.values() for versions in layouts.values()
            for floors in versions.values() for path in floors.values()}
    if set(scales['maps']) != set(images) or set(images) != used:
        raise ValueError('尺度配置与图片清单不一致，请重新导出')
    records = {}
    for path, metadata in images.items():
        entry = scales['maps'][path]
        if not isinstance(entry, dict) or set(entry) != {'scale'}:
            raise ValueError('每张图片只能配置一个 scale')
        if 'saved_scale' in metadata or 'source_image' in metadata:
            raise ValueError('自动资源清单不能重复保存图片路径或尺度')
        scale = entry['scale']
        if scale is None:
            if metadata.get('scale_status') != 'pending':
                raise ValueError('未标为待标定的图片缺少 scale')
        elif type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('scale 必须是有限正数；待标定图片使用 null')
        records[path] = dict(metadata, source_image=path, saved_scale=scale)
    config = {k: template[k] for k in ('perf_mode', 'capture') if k in template}
    config['floor_variants'] = {
        profile: {layout: {version: {floor: dict(records[path]) for floor, path in floors.items()}
                           for version, floors in versions.items()} for layout, versions in layouts.items()}
        for profile, layouts in groups.items()}
    return config


def template_config(root=None):
    root = Path(root) if root is not None else PROJECT_ROOT
    bundle = read_bundle(root)
    if bundle is None:
        return None
    path = (root/SCALE_PATH).resolve(strict=True)
    path.relative_to(root.resolve())
    raw = path.read_bytes()
    record = bundle['files'][SCALE_PATH]
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('尺度配置与已发布资源不一致，请审核后重新导出')
    scales = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_keys)
    return expand_template(bundle['template'], scales)


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
    return profiles[relative]


def runtime_files(root=None):
    """Validate the published package only; never follow offline evidence paths."""
    root = (Path(root) if root is not None else PROJECT_ROOT).resolve()
    bundle = read_bundle(root)
    if bundle is None:
        raise ValueError('缺少运行资源配置，请先执行离线导出')
    files = []
    for name, record in bundle['files'].items():
        path = (root/name).resolve(strict=True)
        path.relative_to(root)
        if (not path.is_file() or path.stat().st_size != record['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
            raise ValueError(f'运行资源指纹不一致：{name}')
        files.append(path)
    return sorted([*files, root/BUNDLE_PATH])
