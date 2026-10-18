"""Build staged recognition-only images; publishing never changes original assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.structure_layout.normalized import MINIMUM_SCALE, RENDER_METHOD, geometry, read_normalized_index
from src.structure_layout.scales import read_scale_overrides
from src.maintenance_paths import profile_root, evidence_path
from tools.common.calibration_assets import project_relative


def render(image, transform):
    matrix = np.array(transform['original_to_index'], np.float64)
    k = matrix[0,0]
    sigma = .5*np.sqrt(max(0, 1/k**2-1))
    filtered = cv2.GaussianBlur(image, (0,0), sigma) if sigma > 0 else image
    return cv2.warpAffine(filtered, matrix[:2], tuple(transform['size']),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build(clean_root, output):
    overrides = read_scale_overrides(clean_root, offline=True)
    if not overrides or any('scale_modes' not in r for r in overrides.values()):
        raise ValueError('需要已核对的最小与 50% 识别尺度')
    output.mkdir(parents=True, exist_ok=False)
    config = profile_root(clean_root)/'recognition_scales.json'
    data = json.loads(config.read_text(encoding='utf-8'))
    audit = json.loads(evidence_path(clean_root, data['zoom_derivation']['path']).read_text(encoding='utf-8'))
    result = dict(schema_version=1, render_method=RENDER_METHOD,
                  source_root=project_relative(clean_root),
                  source_scales_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  scale_modes={'minimum':MINIMUM_SCALE, '50_percent':MINIMUM_SCALE*audit['factor_to_minimum']}, targets={})
    for name, record in overrides.items():
        raw = (clean_root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['source_sha256']:
            raise ValueError(f'原图已改变：{name}')
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        size = list(image.shape[1::-1])
        transform = geometry(size, record['scale_modes']['minimum'])
        reduced = render(image, transform)
        content = cv2.imencode('.png', reduced)[1].tobytes()
        path = output/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(content)
        result['targets'][name] = dict(path=name, source_sha256=record['source_sha256'],
            source_size=size, sha256=hashlib.sha256(content).hexdigest(), **transform)
    (output/'index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    read_normalized_index(clean_root, overrides, output, offline=True)
    return result


def publish(clean_root, staged):
    destination = clean_root.parent/'recognition'
    if destination.exists():
        raise ValueError('正式识别小图目录已存在，请先核对并保留旧版本，不能直接覆盖')
    data, images = read_normalized_index(clean_root, read_scale_overrides(clean_root, offline=True), staged, offline=True)
    # Reproduce geometry from the actual source pixels before publishing. A new
    # hash alone must never legitimize an unrelated or incorrectly resized image.
    for name, record in data['targets'].items():
        source = cv2.imdecode(np.frombuffer((clean_root/name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        actual = cv2.imdecode(np.frombuffer(images[name], np.uint8), cv2.IMREAD_COLOR)
        if not np.array_equal(actual, render(source, record)):
            raise ValueError(f'识别小图像素与原图几何生成结果不一致：{name}')
    shutil.copytree(staged, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-root', type=Path, default=ROOT/'assets/maps/a_bringer_of_doom/hard/clean')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/hard_recognition_v1')
    parser.add_argument('--publish-from', type=Path, help='发布已回放核对的暂存小图；不修改原图或模板尺度')
    args = parser.parse_args()
    if args.publish_from:
        print(publish(args.clean_root, args.publish_from))
    else:
        result = build(args.clean_root, args.output)
        print(json.dumps(dict(output=str(args.output), targets=len(result['targets']), scales=result['scale_modes'])))


if __name__ == '__main__':
    main()
