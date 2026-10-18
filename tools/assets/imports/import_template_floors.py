"""Stage independent floor sources and scales without resizing supplied images."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common.route_calibration import verify_identity_geometry
from tools.assets.calibration.rectify_single_floor_routes import rectify
from src.registration.template_matching.floors import FLOORS, fingerprint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片：{path}')
    return image


def verify_padded_clean(clean, target, *, border=128):
    """Verify the declared border geometrically, including clipped blank margins."""
    h, w = target.shape[:2]
    ch, cw = clean.shape[:2]
    if h < ch or w < cw:
        raise ValueError('单层图片不能容纳干净底图，须重新标定')
    canvas = np.zeros_like(target)
    x, y = w-cw-border, h-ch-border
    if min(x, y) < 0:
        raise ValueError('单层图片边距无法解释，须重新标定')
    th, tw = min(ch, h-y), min(cw, w-x)
    # Cropping blank margins is allowed; missing actual structures is not.
    mask = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) > 48
    if np.any(mask[th:]) or np.any(mask[:, tw:]):
        raise ValueError('单层图片截断底图结构，须重新标定')
    canvas[y:y+th, x:x+tw] = clean[:th, :tw]
    # These sources reuse clean-map pixels. Distributed exact boundary matches
    # are stronger evidence than fitting SIFT on flat rooms obscured by routes.
    same = np.max(np.abs(target[y:y+ch, x:x+cw].astype(np.int16)-clean), axis=2) <= 3
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(same & edge)
    ey, ex = np.nonzero(edge)
    foreground_agreement = float(np.count_nonzero(same & mask) / max(1, mask.sum()))
    edge_agreement = float(len(xs) / max(1, len(ex)))
    span = (np.ptp(np.column_stack((xs, ys)), axis=0) /
            np.maximum(1, np.ptp(np.column_stack((ex, ey)), axis=0))) if len(xs) else np.zeros(2)
    fy, fx = np.nonzero(same & mask)
    foreground_span = (np.ptp(np.column_stack((fx, fy)), axis=0) /
        np.maximum(1, np.ptp(np.column_stack((ex, ey)), axis=0))) if len(fx) else np.zeros(2)
    if (foreground_agreement >= .7 and edge_agreement > .5 and
            np.all(span >= .5) and np.all(foreground_span >= .8)):
        return dict(clean_to_source=[[1, 0, x], [0, 1, y], [0, 0, 1]],
                    method='distributed_clean_pixel_identity', foreground_agreement=foreground_agreement,
                    edge_agreement=edge_agreement, edge_span_fraction=span.tolist(), foreground_span_fraction=foreground_span.tolist())
    geometry = verify_identity_geometry(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), {'floor': [0, 0, w, h]})['floor']
    return dict(clean_to_source=[[1, 0, x], [0, 1, y], [0, 0, 1]], **geometry)


def clean_scale_overrides(config, clean_scales):
    """Convert reviewed screenshot-to-clean scales to the legacy reference units.

    The importer then derives every route version in its own pixel coordinates.
    Runtime matching still reads only the resulting saved per-floor scales.
    """
    result = {}
    for key, value in clean_scales.items():
        profile, layout, floor = key
        if floor not in FLOORS or not np.isfinite(value) or value <= 0:
            raise ValueError('干净底图尺度必须为有限正数且楼层有效')
        candidates = config['original_variants'][profile][layout]['route_clean']['candidates']
        candidate = next(item for item in candidates if item['floor'] == floor)
        reference = config['entries'][candidate['reference_entry_key']]
        baseline = float(candidate['saved_scale'])
        reference_scale = float(reference['saved_scale'])
        if not np.isfinite([baseline, reference_scale]).all() or min(baseline, reference_scale) <= 0:
            raise ValueError('原尺度推算记录无效')
        result[key] = float(value) / baseline * reference_scale
    return result


def import_floors(config, root=ROOT, overrides=None, clean_scales=None):
    config = copy.deepcopy(config)
    previous = config.get('floor_variants', {})
    # Profiles imported directly as single-floor assets have no legacy montage
    # records. Rebuilding nightmare floors must not remove those profiles.
    result = copy.deepcopy(previous)
    overrides = dict(overrides or {})
    clean_scales = clean_scales or {}
    if set(overrides) & set(clean_scales):
        raise ValueError('同一楼层不能同时指定参考尺度和干净底图尺度')
    overrides.update(clean_scale_overrides(config, clean_scales))
    for profile, layouts in config['original_variants'].items():
        result.setdefault(profile, {})
        base = root / 'assets/maps' / profile
        offline = root/'maintenance/maps'/profile
        rectified_path = offline / 'routes/route_lhp_solo_rectified/rectification.json'
        rectified = json.loads(rectified_path.read_text(encoding='utf-8')) if rectified_path.exists() else None
        for layout, variants in layouts.items():
            out = result[profile][layout] = {}
            clean_records = {c['floor']: c for c in variants['route_clean']['candidates']}
            lhp_records = {c['floor']: c for c in variants['route_lhp_solo']['candidates']}
            calibrations = {}
            for version in ('route_zhanshi_fast', 'route_lhp_solo'):
                path = offline / 'routes' / version / layout / 'calibration.json'
                calibrations[version] = (path, json.loads(path.read_text(encoding='utf-8')))
            for variant in variants:
                records = out[variant] = {}
                for floor in FLOORS:
                    ref_key = clean_records[floor]['reference_entry_key']
                    reference = config['entries'][ref_key]
                    for dep in clean_records[floor]['dependencies']:
                        if sha(root / dep['path']) != dep['sha256']:
                            raise ValueError(f'{layout}/{floor}: 推算依赖已改变')
                    old = previous.get(profile, {}).get(layout, {}).get(variant, {}).get(floor, {})
                    ref_scale = overrides.get((profile, layout, floor),
                        old.get('reference_saved_scale', reference['saved_scale']))
                    if not np.isfinite(ref_scale) or ref_scale <= 0:
                        raise ValueError('参考尺度必须为有限正数')
                    correction = ref_scale / float(reference['saved_scale'])
                    source_scale = clean_records[floor]['saved_scale'] * correction
                    clean_path = root / ref_key
                    clean = read(clean_path)
                    dependencies = copy.deepcopy(clean_records[floor]['dependencies'])
                    geometry = None
                    if variant.endswith('_clean') and variant != 'route_clean':
                        paths = list((base / 'routes' / (variant + '_single_floor') /
                                      layout.removeprefix('layout_')).glob(f'*_{floor}.png'))
                        if len(paths) != 1:
                            raise ValueError(f'{layout}/{variant}/{floor}: 必须有且只有一张单层图片')
                        path = paths[0]
                        image = read(path)
                        geometry = verify_padded_clean(clean, image)
                    elif variant == 'route_clean':
                        path, image = clean_path, clean
                    else:
                        cal_path, calibration = calibrations[variant]
                        part = calibration['floors'][floor]
                        if part['status'] != 'PASS':
                            raise ValueError(f'{layout}/{floor}: 单层标定未通过')
                        path = root / part['asset']
                        image = read(path)
                        dependencies.append(dict(path=cal_path.relative_to(root).as_posix(), sha256=sha(cal_path)))
                        if variant == 'route_zhanshi_fast':
                            if image.shape != clean.shape or not np.allclose(part['asset_to_clean_homography'], np.eye(3)):
                                raise ValueError('展十分层素材不是干净底图坐标，须重新标定')
                        else:
                            source_scale = lhp_records[floor]['saved_scale'] * correction
                            if list(image.shape[1::-1]) != part['source_crop_xywh'][2:]:
                                raise ValueError('凉哈皮分层图片尺寸与原图裁剪不一致')
                            if rectified is not None:
                                normalized = rectified['layouts'][layout][floor]
                                for dep in normalized['dependencies']:
                                    if sha(root/dep['path']) != dep['sha256']:
                                        raise ValueError(f'{layout}/{floor}: 校正素材的原始依赖已改变，须重新生成')
                                path = root/normalized['source_image']
                                if sha(path) != normalized['source_sha256']:
                                    raise ValueError('校正素材指纹已改变，须重新生成')
                                image = read(path)
                                if list(image.shape[1::-1]) != normalized['size']:
                                    raise ValueError('校正素材尺寸与生成记录不一致')
                                original = cv2.imdecode(np.fromfile(root/part['asset'], np.uint8), cv2.IMREAD_UNCHANGED)
                                expected, transform, translation = rectify(original,
                                    part['asset_to_clean_homography'], clean.shape[1::-1])
                                actual = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
                                if (not np.array_equal(actual, expected)
                                        or not np.allclose(normalized['clean_to_source'], translation)
                                        or not np.allclose(normalized['original_to_source'], transform)):
                                    raise ValueError('校正图片不符合原始标定几何，须重新生成')
                                dependencies.extend(normalized['dependencies'])
                                dependencies.append(dict(path=rectified_path.relative_to(root).as_posix(),
                                                         sha256=sha(rectified_path)))
                                source_scale = clean_records[floor]['saved_scale'] * correction
                                geometry = dict(method='calibrated_rectification',
                                                clean_to_source=normalized['clean_to_source'])
                    record = dict(source_image=path.relative_to(root).as_posix(), source_sha256=sha(path),
                        size=list(image.shape[1::-1]), saved_scale=source_scale, map_threshold=48,
                        preserve_colored_routes=True, reference_saved_scale=ref_scale,
                        reference_entry_key=ref_key, reference_entry_sha256=fingerprint(reference),
                        dependencies=dependencies, scale_provenance='derived_single_floor_scale')
                    if geometry is not None:
                        record['geometry_verification'] = geometry
                    # Do not discard a manually tuned scale when its geometry and reference are unchanged.
                    if ((profile, layout, floor) not in clean_scales and
                        old.get('source_sha256') == record['source_sha256'] and
                        old.get('reference_saved_scale') == ref_scale and
                        old.get('dependencies') == dependencies and
                        old.get('reference_entry_sha256') == record['reference_entry_sha256']):
                        record['saved_scale'] = old['saved_scale']
                    records[floor] = record
            print(f'Verified {profile}/{layout}', flush=True)
    config['floor_variants'] = result
    config['schema_version'] = 3
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'maintenance/configs/template_matching.json')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/configs/template_matching_floors.json')
    parser.add_argument('--reference-scale', action='append', default=[], metavar='PROFILE:LAYOUT:FLOOR=SCALE')
    parser.add_argument('--clean-scale', action='append', default=[], metavar='PROFILE:LAYOUT:FLOOR=SCALE',
                        help='已离线核对的截图到干净底图尺度；同步换算五个版本，不直接复制数值')
    args = parser.parse_args()
    if args.output.resolve() == args.config.resolve():
        parser.error('请先生成暂存配置，核验后再发布')
    overrides, clean_scales = {}, {}
    for values, target in ((args.reference_scale, overrides), (args.clean_scale, clean_scales)):
        for item in values:
            key, value = item.rsplit('=', 1)
            profile, layout, floor = key.split(':')
            if floor not in FLOORS:
                parser.error('无效楼层')
            target[(profile, layout, floor)] = float(value)
    result = import_floors(json.loads(args.config.read_text(encoding='utf-8')),
                          overrides=overrides, clean_scales=clean_scales)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
