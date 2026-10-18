"""Audit numbered Zhanshi sources and stage profile-specific floor assets without inventing screen scales."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
from PIL import Image
from src.app_catalog import MapCatalog
from src.maintenance_paths import profile_root
from tools.assets.calibration.calibrate_zhanshi_routes import calibrate
from tools.common.calibration_assets import leading_number


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(clean_root, source_root, floors):
    sources = {}
    for path in sorted(source_root.iterdir()):
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            number = leading_number(path)
            if number is None:
                raise ValueError(f'路线原图缺少编号：{path.name}')
            sources.setdefault(number, []).append(path)
    layouts = {}
    for path in sorted(clean_root.glob('layout_*')):
        if path.is_dir():
            number = int(path.name.removeprefix('layout_'))
            if number in layouts:
                raise ValueError(f'底图编号重复：{number}')
            layouts[number] = path
    rows = []
    for number in sorted(layouts.keys() | sources.keys()):
        directory = layouts.get(number)
        images = sources.get(number, [])
        missing = [f for f in floors if directory is None or not (directory/f'{directory.name}_{f}.png').is_file()]
        rows.append(dict(number=number, layout=f'layout_{number:02d}',
                         source=images[0].relative_to(ROOT).as_posix() if len(images) == 1 else None,
                         source_count=len(images), missing_floors=missing,
                         ready=not missing and len(images) == 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='地图ID/难度ID，须先注册 map_catalog')
    parser.add_argument('--variant', default='route_zhanshi_full')
    parser.add_argument('--publish', action='store_true', help='发布审核通过的暂存图片、标定和待标定尺度记录')
    parser.add_argument('--reuse', action='store_true', help='复用源图、底图与标定工具指纹均未改变的暂存结果')
    parser.add_argument('--layouts', nargs='+', type=int, help='仅暂存指定编号，供分批核对')
    args = parser.parse_args()
    if args.publish and args.layouts:
        parser.error('发布前请使用完整编号清单进行复核')
    profile = MapCatalog().profile(*args.profile.split('/'))
    if args.variant not in profile.template_versions:
        parser.error('当前难度未声明所选路线版本')
    base = ROOT/'assets/maps'/profile.key
    stage = ROOT/'workspace/asset_build'/base.relative_to(ROOT)
    offline = profile_root(profile.clean_root)
    metadata_stage = ROOT/'workspace/asset_build'/offline.relative_to(ROOT)
    reports = ROOT/'workspace/reports'/f'{profile.map_id}_{profile.difficulty_id}_calibration'
    reports.mkdir(parents=True, exist_ok=True)
    rows = audit(profile.clean_root, offline/'sources'/args.variant, profile.floors)
    suffix = ''
    if args.layouts:
        rows = [r for r in rows if r['number'] in args.layouts]
        suffix = '_' + '_'.join(map(str, args.layouts))
    report_file = reports/f'inventory{suffix}.json'
    tool_digest = sha(ROOT/'tools/assets/calibration/calibrate_zhanshi_routes.py') + sha(ROOT/'tools/common/route_calibration.py')
    config_path = ROOT/'maintenance/configs/template_matching.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    variants = config.setdefault('floor_variants', {}).setdefault(profile.key, {})
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260926)
    for row in rows:
        if not row['ready']:
            row['status'] = 'MISSING_INPUT'
            continue
        layout = row['layout']
        source = ROOT/row['source']
        route_cal = metadata_stage/'routes'/args.variant/layout/'calibration.json'
        input_hashes = {f: sha(profile.clean_root/layout/f'{layout}_{f}.png') for f in profile.floors}
        cached = json.loads(route_cal.read_text(encoding='utf-8')) if args.reuse and route_cal.exists() else {}
        try:
            if not (cached.get('input_sha256') == input_hashes and cached.get('source_sha256') == sha(source)
                    and cached.get('tool_sha256') == tool_digest
                    and all(i['status'] == 'PASS' for i in cached.get('floors', {}).values())
                    and set(cached.get('floors', {})) == set(profile.floors)):
                cached = calibrate(layout, source, clean_root=profile.clean_root, floors=profile.floors,
                                   asset_root=stage/'routes', metadata_root=metadata_stage/'routes', output_root=reports, variant=args.variant, working_max_size=1400)
                cached.update(input_sha256=input_hashes, tool_sha256=tool_digest)
                # Run-time asset references must point to the final project-relative location.
                for part in cached['floors'].values():
                    for key in ('asset', 'structure_mask'):
                        if part.get(key):
                            part[key] = part[key].removeprefix('workspace/asset_build/')
                route_cal.write_text(json.dumps(cached, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            row['floors'] = {f: dict(status=i['status'], quality=i['quality']) for f, i in cached['floors'].items()}
            if any(i['status'] != 'PASS' for i in cached['floors'].values()):
                row['status'] = 'REVIEW'
                continue
            clean_cal = metadata_stage/'routes/route_clean'/layout/'calibration.json'
            clean_cal.parent.mkdir(parents=True, exist_ok=True)
            clean_payload = dict(layout=layout, variant_id='route_clean', asset_kind='clean',
                                 display_name='原版干净底图', route=(base/'clean'/layout).relative_to(ROOT).as_posix(), floors={})
            for f in profile.floors:
                path = (profile.clean_root/layout/f'{layout}_{f}.png').relative_to(ROOT).as_posix()
                clean_payload['floors'][f] = dict(status='PASS', template=path, asset=path,
                    asset_to_clean_homography=[[1,0,0],[0,1,0],[0,0,1]],
                    diagnostics=dict(method='identity_reference', independent_route_evidence=False))
            clean_cal.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            records = variants.setdefault(layout, {})
            for version, calibration in (('route_clean', clean_cal), (args.variant, route_cal)):
                items = {}
                for floor in profile.floors:
                    clean = profile.clean_root/layout/f'{layout}_{floor}.png'
                    image = clean if version == 'route_clean' else stage/'routes'/version/layout/f'{floor}.png'
                    destination = clean if version == 'route_clean' else base/'routes'/version/layout/image.name
                    dependencies = [dict(path=(offline/'routes'/version/layout/'calibration.json').relative_to(ROOT).as_posix(), sha256=sha(calibration)),
                                    dict(path=clean.relative_to(ROOT).as_posix(), sha256=sha(clean))]
                    if version != 'route_clean':
                        dependencies.append(dict(path=source.relative_to(ROOT).as_posix(), sha256=sha(source)))
                    item = dict(source_image=destination.relative_to(ROOT).as_posix(), source_sha256=sha(image),
                                size=list(Image.open(image).size), saved_scale=None, scale_status='pending',
                                scale_provenance='awaiting_screen_scale_evidence', dependencies=dependencies,
                                map_threshold=48 if version == 'route_clean' else 3, preserve_colored_routes=True)
                    previous = records.get(version, {}).get(floor)
                    previous_dependencies = previous.get('dependencies', []) if previous else []
                    same_geometry = (previous and previous.get('source_sha256') == item['source_sha256']
                        and all(d in previous_dependencies for d in dependencies)
                        and all((ROOT/d['path']).is_file() and sha(ROOT/d['path']) == d['sha256']
                                for d in previous_dependencies))
                    if same_geometry:
                        item['dependencies'] = previous_dependencies
                        for key in ('saved_scale', 'scale_status', 'scale_provenance', 'scale_evidence'):
                            if key in previous:
                                item[key] = previous[key]
                    items[floor] = item
                records[version] = items
            row['status'] = 'PASS'
            print(layout, row['status'], flush=True)
        except (ValueError, OSError, cv2.error) as exc:
            row.update(status='ERROR', error=str(exc))
            print(layout, 'ERROR', str(exc), flush=True)
        report_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    failed = any(r['status'] in ('ERROR', 'REVIEW') for r in rows)
    if args.publish and failed:
        raise SystemExit('存在未通过的几何校验，未发布。请检查：' + str(report_file))
    if args.publish:
        for row in rows:
            if row['status'] != 'PASS':
                continue
            for version in ('route_clean', args.variant):
                for staged_root, final_root in ((stage, base), (metadata_stage, offline)):
                    source_dir = staged_root/'routes'/version/row['layout']
                    if not source_dir.exists():
                        continue
                    destination = final_root/'routes'/version/row['layout']
                    destination.mkdir(parents=True, exist_ok=True)
                    for p in source_dir.iterdir():
                        if p.is_file():
                            shutil.copy2(p, destination/p.name)
    target = config_path if args.publish else ROOT/f'workspace/asset_build/configs/template_matching_profile{suffix}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(report_file), counts={s: sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})})))
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
