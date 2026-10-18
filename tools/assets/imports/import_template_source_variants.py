"""Verify and stage fixed-scale source variants while preserving existing records."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registration.template_matching.direct import entry_fingerprint, saved_scale_value
from tools.common.route_calibration import verify_identity_geometry

SOURCE_VARIANTS = {'route_lhp_solo_clean': 'route_lhp_solo',
                   'route_zhanshi_fast_clean': 'route_zhanshi_fast'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floor_regions(calibration, root, image_shape):
    regions = {}
    for floor, item in calibration['floors'].items():
        if item['status'] != 'PASS':
            raise ValueError(f'{floor}: 对应原图标定未通过')
        if 'source_crop_xywh' in item:
            region = item['source_crop_xywh']
        else:
            clean = cv2.imdecode(np.fromfile(root/item['template'], np.uint8), cv2.IMREAD_GRAYSCALE)
            if clean is None:
                raise ValueError(f'无法读取底图：{item["template"]}')
            height, width = clean.shape
            corners = np.float64([[[0, 0], [width, 0], [0, height], [width, height]]])
            corners = cv2.perspectiveTransform(corners, np.asarray(item['clean_to_route_homography'], np.float64))[0]
            low = np.maximum(np.floor(corners.min(axis=0)).astype(int)-12, 0)
            high = np.minimum(np.ceil(corners.max(axis=0)).astype(int)+12, image_shape[:2][::-1])
            region = [*low.tolist(), *(high-low).tolist()]
        x, y, width, height = map(int, region)
        if min(x, y) < 0 or min(width, height) <= 0 or x+width > image_shape[1] or y+height > image_shape[0]:
            raise ValueError(f'{floor}: 对应原图搜索区域无效')
        regions[floor] = [x, y, width, height]
    if set(regions) != {'2nd', '1st', 'b1'}:
        raise ValueError('对应原图标定缺少三层记录')
    return regions


def import_source_variants(config, root=ROOT):
    config = json.loads(json.dumps(config))
    imported = 0
    for profile_key, layouts in config.get('original_variants', {}).items():
        profile_root = root/'maintenance/maps'/profile_key
        for variant_id, reference_id in SOURCE_VARIANTS.items():
            source_root = profile_root/'sources'/variant_id
            if not source_root.is_dir():
                continue
            for layout, variants in layouts.items():
                reference = variants[reference_id]
                source = root/reference['source_image']
                target = source_root/source.name
                if not target.is_file():
                    raise ValueError(f'{layout}/{variant_id}: 缺少与原图同名的底图版：{target.name}')
                if sha(source) != reference['source_sha256']:
                    raise ValueError(f'{layout}/{reference_id}: 对应原图指纹已改变')
                calibration_path = profile_root/'routes'/reference_id/layout/'calibration.json'
                calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
                if calibration['layout'] != layout or calibration['route'] != reference['source_image']:
                    raise ValueError(f'{layout}: 原图标定对应关系不一致')
                original = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_GRAYSCALE)
                replacement = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_GRAYSCALE)
                if original is None or replacement is None:
                    raise ValueError(f'{layout}: 无法读取对应原图或底图版')
                regions = floor_regions(calibration, root, original.shape)
                try:
                    geometry = verify_identity_geometry(original, replacement, regions)
                except ValueError as exc:
                    raise ValueError(f'{layout}/{variant_id}: {exc}') from exc
                entry = json.loads(json.dumps(reference))
                entry.update(source_image=target.relative_to(root).as_posix(), source_sha256=sha(target),
                             reference_variant_id=reference_id, reference_variant_sha256=entry_fingerprint(reference),
                             scale_provenance='inherited_verified_source_geometry', geometry_verification=geometry,
                             preserve_colored_routes=reference_id == 'route_lhp_solo')
                for candidate in entry['candidates']:
                    saved_scale_value(candidate)
                    dependencies = candidate.setdefault('dependencies', [])
                    for path, fingerprint in ((reference['source_image'], reference['source_sha256']),
                                              (calibration_path.relative_to(root).as_posix(), sha(calibration_path))):
                        if not any(d['path'] == path for d in dependencies):
                            dependencies.append(dict(path=path, sha256=fingerprint))
                    for dependency in dependencies:
                        if sha(root/dependency['path']) != dependency['sha256']:
                            raise ValueError(f'{layout}: 固定尺度依赖已改变：{dependency["path"]}')
                variants[variant_id] = entry
                imported += 1
    if not imported:
        raise ValueError('没有找到可导入的底图版原图目录')
    return config, imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'maintenance/configs/template_matching.json')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/configs/template_matching_source_variants.json')
    args = parser.parse_args()
    if args.config.resolve() == args.output.resolve():
        parser.error('请先生成暂存配置，核验后再复制到正式配置')
    config, imported = import_source_variants(json.loads(args.config.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Staged {imported} verified source variants: {args.output}')


if __name__ == '__main__':
    main()
