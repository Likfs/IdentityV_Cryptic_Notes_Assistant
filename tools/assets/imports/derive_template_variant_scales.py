"""Derive fixed original-image scales from existing route calibrations; stage by default."""
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

from tools.common.route_calibration import fit_fixed_scale
from src.registration.template_matching.direct import entry_fingerprint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_config(config, root=ROOT):
    """Preserve legacy floor entries; add version-specific full-original records."""
    config = json.loads(json.dumps(config))
    originals = config.setdefault('original_variants', {})
    for path in sorted((root/'maintenance/maps').glob('*/*/routes/route_lhp_solo/*/calibration.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        profile_key = '/'.join(path.relative_to(root/'maintenance/maps').parts[:2])
        source = root/data['route']
        source_image = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f'无法读取原图：{source}')
        candidates = []
        clean_candidates = []
        clean_parts = []
        clean_top = 0
        reference_entries = []
        reference_identities = set()
        for floor in ('2nd', '1st', 'b1'):
            item = data['floors'][floor]
            if item['status'] != 'PASS':
                raise ValueError(f'{data["layout"]}/{floor}: 标定未通过')
            reference = config['entries'][item['template']]
            reference_entries.append(dict(key=item['template'], sha256=entry_fingerprint(reference)))
            reference_identities.add((reference['source_image'], reference['source_sha256'], reference['saved_scale']))
            dependencies = []
            for name, expected in ((item['template'], reference['clean_sha256']),
                                   (reference['source_image'], reference['source_sha256']),
                                   (reference['calibration'], reference['calibration_sha256']),
                                   (path.relative_to(root).as_posix(), sha(path))):
                if sha(root/name) != expected:
                    raise ValueError(f'推算依赖指纹不一致：{name}')
                dependencies.append(dict(path=name, sha256=expected))
            clean = cv2.imdecode(np.fromfile(root/item['template'], np.uint8), cv2.IMREAD_GRAYSCALE)
            if clean is None:
                raise ValueError(f'无法读取底图：{item["template"]}')
            ys, xs = np.nonzero(clean[::16, ::16] > 48)
            points = np.column_stack((xs*16, ys*16)).astype(np.float64)
            if len(points) > 4096:
                points = points[np.linspace(0, len(points)-1, 4096, dtype=int)]
            reference_points = cv2.perspectiveTransform(points[None], np.asarray(reference['clean_to_original'], np.float64))[0]
            target_points = cv2.perspectiveTransform(points[None], np.asarray(item['clean_to_route_homography'], np.float64))[0]
            saved_scale, diagnostics = fit_fixed_scale(reference_points/float(reference['saved_scale']), target_points)
            region = [int(v) for v in item['source_crop_xywh']]
            x, y, w, h = region
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > source_image.shape[1] or y+h > source_image.shape[0]:
                raise ValueError(f'{data["layout"]}/{floor}: 原图楼层区域无效')
            candidates.append(dict(floor=floor, saved_scale=saved_scale, search_region=region,
                                   reference_entry_key=item['template'], reference_entry_sha256=entry_fingerprint(reference),
                                   dependencies=dependencies, derivation=diagnostics))
            clean_scale, clean_diagnostics = fit_fixed_scale(
                reference_points/float(reference['saved_scale']), points)
            clean_region = [0, clean_top, clean.shape[1], clean.shape[0]]
            clean_parts.append(dict(floor=floor, path=item['template'],
                                    sha256=reference['clean_sha256'], region=clean_region))
            clean_candidates.append(dict(floor=floor, saved_scale=clean_scale, search_region=clean_region,
                reference_entry_key=item['template'], reference_entry_sha256=entry_fingerprint(reference),
                dependencies=dependencies[:3], derivation=clean_diagnostics))
            clean_top += clean.shape[0] + 64
        if len(reference_identities) != 1:
            raise ValueError('同布局展十原图或保存尺度不一致')
        reference_source, reference_sha, reference_scale = reference_identities.pop()
        variants = originals.setdefault(profile_key, {}).setdefault(data['layout'], {})
        variants['route_zhanshi_fast'] = dict(source_image=reference_source, source_sha256=reference_sha,
                                             scale_provenance='imported_saved_scale', candidates=[dict(saved_scale=reference_scale,
                                                 reference_entries=reference_entries)])
        variants['route_lhp_solo'] = dict(source_image=data['route'], source_sha256=sha(source), map_threshold=48,
                                         scale_provenance='derived_from_route_calibrations', candidates=candidates)
        variants['route_clean'] = dict(source_image=(root/data['floors']['2nd']['template']).parent.relative_to(root).as_posix(),
            source_sha256=entry_fingerprint(clean_parts), source_parts=clean_parts, map_threshold=48,
            scale_provenance='derived_from_clean_calibrations', candidates=clean_candidates)
    config['schema_version'] = 2
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'maintenance/configs/template_matching.json')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/maintenance/configs/template_matching.json')
    args = parser.parse_args()
    if args.output.resolve() == args.config.resolve():
        parser.error('请先输出暂存配置，核验后再复制到正式配置')
    config = derive_config(json.loads(args.config.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    variants = [v for layouts in config['original_variants'].values() for items in layouts.values() for v in items.values()]
    print(f'Staged {len(variants)} originals; {sum(len(v["candidates"]) for v in variants)} fixed candidates: {args.output}')


if __name__ == '__main__':
    main()
