"""Audit complete maintenance inputs and export a self-contained runtime manifest."""
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

from src.maintenance_paths import profile_root, evidence_path
from src.runtime_resources import BUNDLE_PATH, SCALE_PATH, digest, expand_template

FLOOR_FIELDS = ('source_image', 'source_sha256', 'size', 'saved_scale', 'scale_status',
                'map_threshold', 'preserve_colored_routes', 'scale_provenance')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_template(config):
    result = {key: config[key] for key in ('perf_mode', 'capture') if key in config}
    result['floor_variants'] = {
        profile: {layout: {version: {floor: {key: record[key] for key in FLOOR_FIELDS if key in record}
                   for floor, record in records.items()} for version, records in versions.items()}
                  for layout, versions in layouts.items()}
        for profile, layouts in config['floor_variants'].items()}
    return result


def split_template(config):
    """Keep one public scalar per image; the generated manifest owns grouping/metadata."""
    compact = compact_template(config)
    template = {k: compact[k] for k in ('perf_mode', 'capture') if k in compact}
    groups, images, maps = {}, {}, {}
    for profile, layouts in compact['floor_variants'].items():
        groups[profile] = {}
        for layout, versions in layouts.items():
            groups[profile][layout] = {}
            for version, floors in versions.items():
                groups[profile][layout][version] = {}
                for floor, record in floors.items():
                    path = record['source_image']
                    value = {'scale': record['saved_scale']}
                    metadata = {k: v for k, v in record.items() if k not in ('source_image', 'saved_scale')}
                    if path in maps and (maps[path] != value or images[path] != metadata):
                        raise ValueError(f'同一图片存在冲突尺度或处理参数：{path}')
                    maps[path], images[path] = value, metadata
                    groups[profile][layout][version][floor] = path
    template.update(groups=groups, images=images)
    scales = {'maps': maps}
    if expand_template(template, scales) != compact:
        raise ValueError('简化配置不能改变原有匹配记录')
    return template, scales


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def validate_dimensions(template, root):
    from PIL import Image
    checked = set()
    for layouts in template['floor_variants'].values():
        for variants in layouts.values():
            for records in variants.values():
                for record in records.values():
                    key = (record['source_image'], tuple(record['size']))
                    if key not in checked:
                        with Image.open(root/record['source_image']) as image:
                            if list(image.size) != record['size']:
                                raise ValueError('单层图片尺寸与运行记录不一致，请重新导入')
                        checked.add(key)


def generate_package():
    import cv2
    import numpy as np
    from tools.release.build_windows_exe import maintenance_files
    from src.app_catalog import MapCatalog, VERSION_NAMES
    from src.structure_layout.scales import read_scale_overrides
    from src.structure_layout.normalized import read_normalized_index
    from tools.assets.imports.build_recognition_index import render

    # Validate complete old dependency chains, including originals and references.
    audited = maintenance_files()
    before = {p: sha(p) for p in audited}
    config = json.loads((ROOT/'maintenance/configs/template_matching.json').read_text(encoding='utf-8'))
    compact = compact_template(config)
    validate_dimensions(compact, ROOT)
    template, scales = split_template(config)
    needed = {ROOT/'configs/map_catalog.json', ROOT/'configs/app_defaults.json',
              ROOT/'src/registration/template_matching/provenance.json'}
    needed.update(p for p in audited if '/assets/app_ui/' in p.as_posix()
                  or '/assets/layout_retrieval/' in p.as_posix())
    for layouts in compact['floor_variants'].values():
        for versions in layouts.values():
            for records in versions.values():
                for record in records.values():
                    needed.add(ROOT/record['source_image'])
    recognition, statuses = {}, {}
    catalog = MapCatalog(ROOT/'configs/map_catalog.json')
    for profile in catalog.profiles.values():
        needed.update(profile.clean_root.glob('layout_*/*.png'))
        if profile.entrance_root:
            needed.update(profile.entrance_root.glob('*.png'))
        targets = read_scale_overrides(profile.clean_root, offline=True)
        normalized, images = read_normalized_index(profile.clean_root, targets, offline=True)
        compact = {name: {key: record[key] for key in ('source_sha256', 'scales', 'scale_modes') if key in record}
                   for name, record in targets.items()}
        entry = dict(targets=compact)
        if normalized:
            scale_data = json.loads((profile_root(profile.clean_root)/'recognition_scales.json').read_text(encoding='utf-8'))
            audit = json.loads(evidence_path(profile.clean_root, scale_data['zoom_derivation']['path']).read_text(encoding='utf-8'))
            entry['factor_to_minimum'] = audit['factor_to_minimum']
            index = profile.clean_root.parent/'recognition/index.json'
            entry['index_sha256'] = sha(index)
            needed.add(index)
            for name, record in normalized['targets'].items():
                original = cv2.imdecode(np.frombuffer((profile.clean_root/name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                actual = cv2.imdecode(np.frombuffer(images[name], np.uint8), cv2.IMREAD_COLOR)
                if not np.array_equal(actual, render(original, record)):
                    raise ValueError(f'识别小图像素与原图几何生成结果不一致：{name}')
                needed.add(index.parent/record['path'])
        recognition[profile.clean_root.relative_to(ROOT).as_posix()] = entry
        statuses[profile.key] = {}
        order = {v['id']: index for index, v in enumerate(profile.asset_versions)}
        for layout in catalog.layouts(profile):
            variants = []
            for path in sorted(profile.calibration_root.glob(f'route_*/{layout}/calibration.json')):
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('layout') != layout:
                    continue
                variant = data['variant_id']
                variants.append(dict(variant_id=variant, source_name=Path(data['route']).name,
                    floor_status={f: data['floors'][f]['status'] for f in profile.floors},
                    display_name=next((v['name'] for v in profile.asset_versions if v['id']==variant),
                                      VERSION_NAMES.get(variant, str(data.get('display_name','')))),
                    asset_kind=str(data.get('asset_kind','route'))))
            statuses[profile.key][layout] = sorted(variants, key=lambda v: order.get(v['variant_id'],len(order)))
    files = {}
    for path in sorted(needed):
        path = path.resolve(strict=True)
        name = path.relative_to(ROOT).as_posix()
        files[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    scale_bytes = json_bytes(scales)
    files[SCALE_PATH] = dict(bytes=len(scale_bytes), sha256=hashlib.sha256(scale_bytes).hexdigest())
    # Detect edits during export; never publish a mixed old/new audit.
    if any(sha(path) != expected for path, expected in before.items()):
        raise ValueError('导出期间维护资源发生变化，请重新审核导出')
    payload = dict(template=template, recognition=recognition, route_status=statuses, files=files,
                   maintenance_revision=digest({p.relative_to(ROOT).as_posix(): value for p,value in before.items()}))
    return dict(schema_version=2, payload_sha256=digest(payload), payload=payload), scale_bytes


def generate():
    return generate_package()[0]


def publish_package(output, data, scales):
    """Stage both files first; publish manifest last, rolling back scales on failure."""
    output = output.resolve()
    sidecar = output.parent/Path(SCALE_PATH).name
    if output == sidecar:
        raise ValueError('运行清单输出不能命名为 scale_config.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    old_scales = sidecar.read_bytes() if sidecar.exists() else None
    staged = []
    def stage(raw):
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as f:
            path = Path(f.name)
            staged.append(path)
            f.write(raw)
        return path
    try:
        scale_temp, bundle_temp = stage(scales), stage(json_bytes(data))
        os.replace(scale_temp, sidecar)
        try:
            os.replace(bundle_temp, output)
        except OSError:
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
    parser.add_argument('--check', action='store_true', help='完整审核维护资料并拒绝陈旧运行清单')
    args = parser.parse_args()
    data, scales = generate_package()
    if args.check:
        if (json.loads((ROOT/BUNDLE_PATH).read_text(encoding='utf-8')) != data
                or (ROOT/SCALE_PATH).read_bytes() != scales):
            raise ValueError('运行资源清单已过期，请重新导出')
        print('完整离线审核通过，运行清单与维护资料一致')
        return
    output = ROOT/BUNDLE_PATH if args.publish else args.output.resolve()
    publish_package(output, data, scales)
    print(json.dumps(dict(output=str(output), files=len(data['payload']['files']),
                         bytes=sum(r['bytes'] for r in data['payload']['files'].values())), ensure_ascii=False))


if __name__ == '__main__':
    main()
