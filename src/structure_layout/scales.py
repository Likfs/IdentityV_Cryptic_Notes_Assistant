"""Per-source, evidence-backed fixed recognition scales; never inferred at runtime."""
import json
import math
import hashlib
from src.maintenance_paths import profile_root, evidence_path


def derivation_files(clean_root, derivation, *, source_hashes=None):
    """Validate every input of an offline scale transfer on each request boundary."""
    if not isinstance(derivation, dict) or not isinstance(derivation.get('dependencies'), list):
        raise ValueError('识别尺度推算依赖格式无效')
    paths = []
    for dependency in [derivation, *derivation['dependencies']]:
        if not isinstance(dependency, dict) or not isinstance(dependency.get('path'), str):
            raise ValueError('识别尺度推算依赖路径无效')
        path = evidence_path(clean_root, dependency['path'])
        digest = (source_hashes or {}).get(path)
        if digest is None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dependency.get('sha256'):
            raise ValueError(f'识别尺度推算依赖指纹不一致：{dependency["path"]}')
        paths.append(path)
    return paths


def read_scale_overrides(clean_root, *, source_hashes=None, offline=False):
    if not offline:
        from src.runtime_resources import recognition_profile
        profile = recognition_profile(clean_root)
        if profile is not None:
            for name, record in profile['targets'].items():
                parts = name.split('/')
                scales = record.get('scales', [])
                if (len(parts) != 2 or not parts[0].startswith('layout_')
                        or parts[1] not in [f'{parts[0]}_{f}.png' for f in ('2nd','1st','b1')]
                        or not (clean_root/name).is_file()
                        or not isinstance(scales, list) or not 1 <= len(scales) <= 2
                        or any(type(s) not in (int,float) or not math.isfinite(s) or s <= 0 for s in scales)
                        or len(set(scales)) != len(scales)
                        or not isinstance(record.get('source_sha256'), str) or len(record['source_sha256']) != 64):
                    raise ValueError('运行识别尺度记录无效，请重新导出')
                modes = record.get('scale_modes')
                if modes is not None:
                    factor = profile.get('factor_to_minimum')
                    if (type(factor) not in (int,float) or not math.isfinite(factor) or not .1 < factor < .9
                            or set(modes) != {'minimum','50_percent'}
                            or scales != [modes['minimum'],modes['50_percent']]
                            or modes['50_percent'] != round(modes['minimum']*factor,6)):
                        raise ValueError('运行识别两档尺度不一致，请重新导出')
            return profile['targets']
    path = profile_root(clean_root) / 'recognition_scales.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('targets'), dict):
        raise ValueError('识别尺度配置格式无效')
    derivation = data.get('derivation')
    if derivation is not None:
        derivation_files(clean_root, derivation, source_hashes=source_hashes)
    zoom = data.get('zoom_derivation')
    zoom_audit = None
    if zoom is not None:
        paths = derivation_files(clean_root, zoom, source_hashes=source_hashes)
        zoom_audit = json.loads(paths[0].read_text(encoding='utf-8'))
        factor = zoom_audit.get('factor_to_minimum')
        if (derivation is None or zoom_audit.get('base_derivation_sha256') != derivation['sha256']
                or type(factor) not in (int, float) or not math.isfinite(factor) or not .1 < factor < .9
                or not isinstance(zoom_audit.get('minimum_scales'), dict)):
            raise ValueError('第二档识别尺度与最小尺度标定不一致')
    for name, record in data['targets'].items():
        if not isinstance(record, dict):
            raise ValueError('识别尺度记录格式无效')
        parts = name.split('/')
        if (len(parts) != 2 or not parts[0].startswith('layout_')
                or parts[1] not in [f'{parts[0]}_{f}.png' for f in ('2nd', '1st', 'b1')]
                or not (clean_root / name).is_file()):
            raise ValueError('识别尺度引用了无效底图')
        scales = record.get('scales', [])
        if (not isinstance(scales, list) or not 1 <= len(scales) <= 2
                or any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in scales)
                or len(set(scales)) != len(scales)):
            raise ValueError('每张底图须提供一至两个有效固定识别尺度')
        if not isinstance(record.get('source_sha256'), str) or len(record['source_sha256']) != 64:
            raise ValueError('识别尺度缺少底图指纹')
        modes = record.get('scale_modes')
        if modes is not None or zoom_audit is not None:
            if zoom_audit is None or not isinstance(modes, dict) or set(modes) != {'minimum', '50_percent'}:
                raise ValueError('第二档识别尺度缺少对应的截图几何证据')
            minimum = zoom_audit['minimum_scales'].get(name)
            if (type(minimum) not in (int,float) or not math.isfinite(minimum) or minimum <= 0
                    or modes != {'minimum':minimum, '50_percent':round(minimum*factor,6)}
                    or scales != [modes['minimum'], modes['50_percent']]):
                raise ValueError('第二档识别尺度与截图几何比例不一致')
        evidence = record.get('evidence', {})
        if evidence.get('method') == 'shared_geometry_scale_transfer':
            if derivation is None or evidence.get('derivation_sha256') != derivation['sha256']:
                raise ValueError('识别尺度缺少对应的几何推算记录')
    return data['targets']


def matched_scale_mode(modes, scale):
    """Describe the winning fixed candidate, not an independent game zoom detector."""
    return next((name for name, value in modes.items()
                 if math.isclose(value, scale, rel_tol=1e-6)), None)


def check_scale_overrides(clean_root):
    """Cheap resource audit without constructing the recognition index."""
    result = dict(configured_targets=0, errors=[])
    try:
        records = read_scale_overrides(clean_root)
        for name, record in records.items():
            if hashlib.sha256((clean_root/name).read_bytes()).hexdigest() != record['source_sha256']:
                raise ValueError(f'识别尺度底图指纹不一致：{name}')
        result['configured_targets'] = len(records)
        from .normalized import read_normalized_index
        normalized, _ = read_normalized_index(clean_root, records)
        result['normalized_targets'] = len(normalized.get('targets', {}))
        result['normalized_scale_modes'] = normalized.get('scale_modes', {})
    except (OSError, ValueError) as exc:
        result['errors'].append(str(exc))
    return result
