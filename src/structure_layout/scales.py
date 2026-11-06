"""Fixed per-image scales from the published source-image contract."""
import hashlib
import math
from src.floor_definitions import FLOORS


def read_scale_overrides(clean_root):
    from src.runtime_resources import recognition_profile
    profile = recognition_profile(clean_root)
    if profile is None or not profile.get('source_images'):
        raise ValueError('缺少已发布的干净底图识别配置，请重新发布素材')
    for name, record in profile['targets'].items():
        parts = name.split('/')
        scales = record.get('scales', [])
        if (len(parts) != 2 or not parts[0].startswith('layout_')
                or parts[1] not in [f'{parts[0]}_{f}.png' for f in FLOORS]
                or not (clean_root/name).is_file()
                or not isinstance(scales, list) or len(scales) != 1
                or type(scales[0]) not in (int, float) or not math.isfinite(scales[0]) or scales[0] <= 0
                or not isinstance(record.get('source_sha256'), str) or len(record['source_sha256']) != 64):
            raise ValueError('运行识别尺度记录无效，请重新导出')
    return profile['targets']


def check_scale_overrides(clean_root):
    result = dict(configured_targets=0, errors=[])
    try:
        records = read_scale_overrides(clean_root)
        for name, record in records.items():
            if hashlib.sha256((clean_root/name).read_bytes()).hexdigest() != record['source_sha256']:
                raise ValueError(f'识别尺度底图指纹不一致：{name}')
        result['configured_targets'] = len(records)
    except (OSError, ValueError) as exc:
        result['errors'].append(str(exc))
    return result
