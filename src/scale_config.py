"""The single editable source of default and per-layout minimum scales."""
import json
import math
from pathlib import Path
import re

SCALE_PATH = 'configs/scale_config.json'


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'尺度配置存在重复键：{key}')
        result[key] = value
    return result


def validate_scales(data):
    if not isinstance(data, dict) or set(data) != {'maps'} or not isinstance(data['maps'], dict):
        raise ValueError('scale_config.json 必须按 maps／地图 ID／难度 ID 配置尺度')
    for map_id, difficulties in data['maps'].items():
        if (not isinstance(map_id, str) or not map_id or '/' in map_id or '\\' in map_id
                or not isinstance(difficulties, dict)):
            raise ValueError('尺度配置的地图 ID 或难度分组无效')
        for difficulty, entry in difficulties.items():
            if (not isinstance(difficulty, str) or not difficulty or '/' in difficulty or '\\' in difficulty
                    or not isinstance(entry, dict) or 'minimum_scale' not in entry
                    or set(entry) - {'minimum_scale', 'layout_minimum_scales'}):
                raise ValueError(f'{map_id}/{difficulty}：尺度配置字段无效')
            value = entry['minimum_scale']
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{map_id}/{difficulty}：minimum_scale 必须为有限正数，待配置难度可用 null')
            overrides = entry.get('layout_minimum_scales', {})
            if (not isinstance(overrides, dict) or any(
                    not isinstance(layout, str) or re.fullmatch(r'layout_\d+', layout) is None
                    or type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
                    for layout, scale in overrides.items())):
                raise ValueError(f'{map_id}/{difficulty}：layout_minimum_scales 必须为布局编号到有限正数的映射')
    return data


def load_scales(root, *, raw=None):
    raw = (Path(root)/SCALE_PATH).read_bytes() if raw is None else raw
    return validate_scales(json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_keys))


def profile_settings(data, map_id, difficulty_id, *, ready=False):
    try:
        entry = data['maps'][map_id][difficulty_id]
    except KeyError as exc:
        raise ValueError(f'scale_config.json 缺少 {map_id}/{difficulty_id} 的尺度配置') from exc
    if ready and entry['minimum_scale'] is None:
        raise ValueError(f'{map_id}/{difficulty_id}：已配置难度必须提供有效 minimum_scale')
    return entry


def profile_keys(data):
    return {f'{map_id}/{difficulty}' for map_id, difficulties in data['maps'].items() for difficulty in difficulties}
