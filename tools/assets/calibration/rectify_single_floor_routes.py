"""Stage route floors in clean-map coordinates using existing verified calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common.layered_assets import RouteOverlayAsset, warp_route_overlay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectify(rgba, asset_to_clean, clean_size, padding=128):
    matrix = np.asarray(asset_to_clean, np.float64)
    if (rgba.ndim != 3 or rgba.shape[2] != 4 or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError('单层素材或标定矩阵无效')
    width, height = map(int, clean_size)
    if min(width, height) <= 0 or padding < 0:
        raise ValueError('校正画布尺寸无效')
    translation = np.float64([[1, 0, padding], [0, 1, padding], [0, 0, 1]])
    asset = RouteOverlayAsset('', '', '', rgba, matrix, Path('.'), Path('.'))
    layer, transform = warp_route_overlay(asset, translation,
        (width + 2*padding, height + 2*padding), opacity=1.)
    return layer, transform, translation


def stage(config, output, root=ROOT):
    output = Path(output)
    if output.exists():
        raise ValueError('请使用新的暂存目录，避免覆盖已有素材')
    manifests = {}
    for profile, layouts in config['floor_variants'].items():
        relative = Path('assets/maps')/profile/'routes/route_lhp_solo_rectified'
        target = output/relative
        manifest = dict(schema_version=1, method='existing_calibration_to_clean_pixels', layouts={})
        for layout, variants in layouts.items():
            calibration_path = root/'maintenance/maps'/profile/'routes/route_lhp_solo'/layout/'calibration.json'
            calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
            records = manifest['layouts'][layout] = {}
            for floor, entry in variants['route_clean'].items():
                # Source geometry must still match the template dependencies.
                for dependency in variants['route_lhp_solo'][floor]['dependencies']:
                    if sha(root/dependency['path']) != dependency['sha256']:
                        raise ValueError(f'{layout}/{floor}: 标定依赖已改变')
                part = calibration['floors'][floor]
                if part['status'] != 'PASS':
                    raise ValueError(f'{layout}/{floor}: 标定未通过')
                source = root/part['asset']
                rgba = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_UNCHANGED)
                layer, transform, translation = rectify(rgba, part['asset_to_clean_homography'], entry['size'])
                destination = target/layout/f'{floor}.png'
                destination.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode('.png', layer)[1].tofile(destination)
                records[floor] = dict(source_image=(relative/layout/f'{floor}.png').as_posix(),
                    source_sha256=sha(destination), size=list(layer.shape[1::-1]),
                    clean_to_source=translation.tolist(), original_to_source=transform.tolist(),
                    dependencies=[dict(path=part['asset'], sha256=sha(source)),
                        dict(path=calibration_path.relative_to(root).as_posix(), sha256=sha(calibration_path)),
                        dict(path=entry['source_image'], sha256=sha(root/entry['source_image']))])
            print(f'Rectified {profile}/{layout}', flush=True)
        manifest_relative = Path('maintenance/maps')/profile/'routes/route_lhp_solo_rectified/rectification.json'
        manifest_path = output/manifest_relative
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        manifests[profile] = manifest_relative.as_posix()
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'maintenance/configs/template_matching.json')
    parser.add_argument('--output', type=Path, required=True, help='新的暂存目录；核对后另行发布')
    args = parser.parse_args()
    stage(json.loads(args.config.read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
