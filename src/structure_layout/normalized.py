"""Validated, isotropically normalized recognition assets; originals stay authoritative."""
import hashlib
import json
import math
import struct
from pathlib import Path
from src.maintenance_paths import profile_root, evidence_path

MINIMUM_SCALE = 1.5
# Fixed 30.8% preset validated in the offline coarse-sampling comparison.
# Keep its exact value so runtime candidates match the reviewed demo.
COARSE_RESIZE = 0.30792678925763023
RENDER_METHOD = 'gaussian_antialias_affine_linear_pixel_centers_v1'


def png_size(data):
    if data[:8] != b'\x89PNG\r\n\x1a\n' or len(data) < 24:
        raise ValueError('识别小图或原图不是有效 PNG')
    return list(struct.unpack('>II', data[16:24]))


def geometry(size, original_minimum):
    factor = MINIMUM_SCALE / original_minimum
    offset = (factor - 1) / 2
    return dict(size=[math.ceil(n * factor) for n in size],
                original_to_index=[[factor, 0., offset], [0., factor, offset], [0., 0., 1.]])


def read_normalized_index(clean_root, overrides, index_root=None, *, source_snapshot=None, offline=False):
    """Validate hashes AND the source-derived geometry, including source dimensions."""
    root = Path(index_root) if index_root is not None else clean_root.parent/'recognition'
    manifest = root/'index.json'
    if index_root is None and not root.exists():
        return {}, {}
    data = json.loads(manifest.read_text(encoding='utf-8'))
    from src.runtime_resources import recognition_profile
    runtime = None if offline else recognition_profile(clean_root)
    if runtime is not None:
        if (data.get('schema_version') != 1 or data.get('render_method') != RENDER_METHOD
                or hashlib.sha256(manifest.read_bytes()).hexdigest() != runtime.get('index_sha256')):
            raise ValueError('识别小图配置或尺度依赖已改变，请重新导出')
        factor = runtime['factor_to_minimum']
    else:
        config = profile_root(clean_root)/'recognition_scales.json'
        if (not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('render_method') != RENDER_METHOD
                or data.get('source_scales_sha256') != hashlib.sha256(config.read_bytes()).hexdigest()):
            raise ValueError('识别小图配置或尺度依赖已改变，请重新生成')
        scale_data = json.loads(config.read_text(encoding='utf-8'))
        zoom = scale_data.get('zoom_derivation', {})
        zoom_path = evidence_path(clean_root, zoom.get('path', ''))
        factor = json.loads(zoom_path.read_text(encoding='utf-8'))['factor_to_minimum']
    modes = {'minimum':MINIMUM_SCALE, '50_percent':MINIMUM_SCALE*factor}
    if data.get('scale_modes') != modes:
        raise ValueError('识别小图两档尺度与截图标定不一致')
    records = data.get('targets')
    sources = {p.relative_to(clean_root).as_posix() for p in clean_root.glob('layout_*/*.png')}
    if not isinstance(records, dict) or set(records) != sources or set(records) != set(overrides):
        raise ValueError('识别小图必须完整覆盖当前底图，不能混用缺失或旧布局')
    images = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            raise ValueError('识别小图记录格式无效')
        snapshot = (source_snapshot or {}).get(name)
        if snapshot is None:
            source = (clean_root/name).read_bytes()
            digest = hashlib.sha256(source).hexdigest()
        else:
            source, digest = snapshot
        override = overrides[name]
        if (record.get('source_sha256') != override['source_sha256']
                or digest != override['source_sha256']):
            raise ValueError(f'识别小图原图指纹已改变：{name}')
        expected = geometry(png_size(source), override['scale_modes']['minimum'])
        if (record.get('source_size') != png_size(source)
                or record.get('size') != expected['size']
                or record.get('original_to_index') != expected['original_to_index']
                or record.get('path') != name):
            raise ValueError(f'识别小图几何换算不一致：{name}')
        path = (root/name).resolve()
        path.relative_to(root.resolve())
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.get('sha256') or png_size(content) != expected['size']:
            raise ValueError(f'识别小图指纹或尺寸不一致：{name}')
        images[name] = content
    return data, images


def original_pose(fit, target):
    """Expose legacy original-image coordinates, never a projection registration."""
    matrix = target.get('original_to_index')
    if matrix is None:
        return dict(fit)
    k, tx, ty = matrix[0][0], matrix[0][2], matrix[1][2]
    return dict(fit, scale=fit['scale']/k, x=(fit['x']-tx)/k, y=(fit['y']-ty)/k,
                index_pose={key:fit[key] for key in ('scale', 'x', 'y')},
                original_to_index=matrix, index_coordinate_space='normalized_capture_roi_to_recognition_image')
