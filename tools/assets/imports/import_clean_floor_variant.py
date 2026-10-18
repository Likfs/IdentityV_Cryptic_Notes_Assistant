"""Stage a clean-backed single-floor version after verifying every source geometry."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.app_catalog import MapCatalog
from src.registration.template_matching.floors import fingerprint
from tools.assets.imports.import_template_floors import read, sha, verify_padded_clean


def import_variant(config, profile, source_root, variant, *, border, root=ROOT):
    """Inherit minimum scales only after a checked, unscaled translation."""
    root = Path(root).resolve()
    source_root = Path(source_root).resolve()
    source_root.relative_to(root)
    if variant == 'route_clean' or not variant.endswith('_clean'):
        raise ValueError('底图版版本必须以 _clean 结尾，且不能覆盖 route_clean')
    if not isinstance(border, int) or border < 0:
        raise ValueError('边距必须为非负整数')
    layouts = config['floor_variants'][profile.key]
    expected = {source_root/layout/f'{layout}_{floor}.png'
                for layout in layouts for floor in profile.floors}
    if set(source_root.rglob('*.png')) != expected:
        raise ValueError('底图版编号／楼层文件集合与当前难度不一致')
    output = copy.deepcopy(config)
    verified = set()
    for layout, versions in layouts.items():
        records = {}
        for floor in profile.floors:
            reference = versions['route_clean'][floor]
            scale = reference.get('saved_scale')
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{layout}/{floor}: 原底图最小尺度尚未核对')
            dependencies = copy.deepcopy(reference.get('dependencies', []))
            clean_dep = dict(path=reference['source_image'], sha256=reference['source_sha256'])
            if clean_dep not in dependencies:
                dependencies.append(clean_dep)
            for dep in dependencies:
                path = (root/dep['path']).resolve();path.relative_to(root)
                key = (path, dep['sha256'])
                if key not in verified:
                    if sha(path) != dep['sha256']:
                        raise ValueError(f'{layout}/{floor}: 参考底图或尺度依赖已改变')
                    verified.add(key)
            clean = read(root/reference['source_image'])
            if list(clean.shape[1::-1]) != reference['size']:
                raise ValueError('参考底图尺寸与尺度记录不一致')
            path = source_root/layout/f'{layout}_{floor}.png'
            image = read(path)
            if image.shape[:2] != tuple(n+2*border for n in clean.shape[:2]):
                raise ValueError(f'{layout}/{floor}: 底图版尺寸不符合声明边距')
            geometry = verify_padded_clean(clean, image, border=border)
            records[floor] = dict(source_image=path.relative_to(root).as_posix(),
                source_sha256=sha(path), size=list(image.shape[1::-1]),
                saved_scale=float(scale), scale_status='derived_geometry',
                scale_provenance='verified_clean_translation', map_threshold=48,
                preserve_colored_routes=True, dependencies=dependencies,
                geometry_verification=geometry,
                reference_floor_variant=dict(variant_id='route_clean', floor=floor,
                                             sha256=fingerprint(reference)),
                scale_evidence=dict(method='verified_unscaled_clean_translation',
                    reference_scale_status=reference.get('scale_status'),
                    direct_capture_verified=False))
        output['floor_variants'][profile.key][layout][variant] = records
        print(f'Verified {layout}: {list(records)}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'maintenance/configs/template_matching.json')
    parser.add_argument('--profile', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--border', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.config.resolve() or args.output.exists():
        parser.error('请使用新的暂存输出文件，核对后再发布')
    profile = MapCatalog().profile(*args.profile.split('/'))
    result = import_variant(json.loads(args.config.read_text(encoding='utf-8')),
        profile, args.source_root, args.variant, border=args.border)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
