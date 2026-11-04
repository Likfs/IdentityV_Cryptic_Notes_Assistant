"""Audit same-canvas floor images and export a self-contained runtime manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime_resources import BUNDLE_PATH, SCALE_PATH, digest

def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def generate_package():
    from tools.assets.imports.export_aligned_assets import generate
    return generate(ROOT)


def generate():
    return generate_package()[0]


def publish_package(output, data, scales, *, preserve_scales=False):
    """Export a pair, or publish only the manifest beside the editable scale source."""
    output = output.resolve()
    sidecar = output.parent/Path(SCALE_PATH).name
    if output == sidecar:
        raise ValueError('运行清单输出不能命名为 scale_config.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    old_scales = sidecar.read_bytes() if sidecar.exists() else None
    if preserve_scales and old_scales != scales:
        raise ValueError('审核后尺度配置发生变化，请重新导出')
    staged = []
    def stage(raw):
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as f:
            path = Path(f.name)
            staged.append(path)
            f.write(raw)
        return path
    try:
        scale_temp, bundle_temp = stage(scales), stage(json_bytes(data))
        if not preserve_scales:
            os.replace(scale_temp, sidecar)
        try:
            os.replace(bundle_temp, output)
        except OSError:
            if preserve_scales:
                raise
            if old_scales is None:
                sidecar.unlink()
            else:
                os.replace(stage(old_scales), sidecar)
            raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/runtime_resources.json')
    parser.add_argument('--publish', action='store_true', help='审核通过后原子替换正式运行清单')
    parser.add_argument('--check', action='store_true', help='完整审核同画布素材并拒绝陈旧运行清单')
    args = parser.parse_args()
    data, scales = generate_package()
    if args.check:
        if (json.loads((ROOT/BUNDLE_PATH).read_text(encoding='utf-8')) != data
                or (ROOT/SCALE_PATH).read_bytes() != scales):
            raise ValueError('运行资源清单已过期，请重新导出')
        print('完整离线审核通过，运行清单与同画布素材一致')
        return
    output = ROOT/BUNDLE_PATH if args.publish else args.output.resolve()
    publish_package(output, data, scales, preserve_scales=args.publish)
    print(json.dumps(dict(output=str(output), files=len(data['payload']['files']),
                         bytes=sum(r['bytes'] for r in data['payload']['files'].values())), ensure_ascii=False))


if __name__ == '__main__':
    main()
