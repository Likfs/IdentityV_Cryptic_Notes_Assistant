"""Offline, audited scale transfer from one labelled capture through shared map geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.app_catalog import MapCatalog
from src.maintenance_paths import profile_root
from tools.common.route_calibration import fit_fixed_scale

PARAMETERS = dict(feature_max_size=1800, features=10000, lowe_ratio=.72,
                  angle_difference=5, minimum_inliers=8, minimum_unique_points=6,
                  minimum_short_span=120, minimum_long_span=350, max_rotation=.1,
                  max_p95=2.5, max_leave_one_out_range=.003, max_cycle_error=.004)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def features(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    factor = min(1., PARAMETERS['feature_max_size']/max(image.shape))
    small = cv2.resize(image, None, fx=factor, fy=factor)
    points, descriptors = cv2.SIFT_create(nfeatures=PARAMETERS['features'], contrastThreshold=.01).detectAndCompute(small, None)
    return dict(points=np.float32([k.pt for k in points])/factor, desc=descriptors,
                angles=np.float32([k.angle for k in points]), shape=np.array(image.shape[::-1]))


def match_link(a, b):
    if a['desc'] is None or b['desc'] is None:
        return None
    matcher = cv2.BFMatcher()
    reverse = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(b['desc'], a['desc'], k=2)
               if m.distance < PARAMETERS['lowe_ratio']*n.distance}
    matches = [m for m,n in matcher.knnMatch(a['desc'], b['desc'], k=2)
               if m.distance < PARAMETERS['lowe_ratio']*n.distance
               and reverse.get(m.trainIdx) == m.queryIdx
               and abs((a['angles'][m.queryIdx]-b['angles'][m.trainIdx]+180)%360-180) < PARAMETERS['angle_difference']]
    if len(matches) < PARAMETERS['minimum_inliers']:
        return None
    x = np.float32([a['points'][m.queryIdx] for m in matches])
    y = np.float32([b['points'][m.trainIdx] for m in matches])
    matrix, mask = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC,
        ransacReprojThreshold=4, maxIters=10000, confidence=.999)
    if matrix is None or int(mask.sum()) < PARAMETERS['minimum_inliers']:
        return None
    x, y = x[mask.ravel()>0], y[mask.ravel()>0]
    rotation = float(np.degrees(np.arctan2(matrix[1,0], matrix[0,0])))
    if (abs(rotation) > PARAMETERS['max_rotation']
            or min(len(np.unique(np.round(x), axis=0)), len(np.unique(np.round(y), axis=0))) < PARAMETERS['minimum_unique_points']):
        return None
    scale, details = fit_fixed_scale(x, y)
    sx, sy = np.ptp(x, axis=0), np.ptp(y, axis=0)
    if (min(sx.min(), sy.min()) < PARAMETERS['minimum_short_span']
            or min(sx.max(), sy.max()) < PARAMETERS['minimum_long_span']
            or details['residual_p95_original_pixels'] > PARAMETERS['max_p95']):
        return None
    estimates = [fit_fixed_scale(np.delete(x,i,0), np.delete(y,i,0))[0] for i in range(len(x))]
    stability = float(np.ptp(estimates)/scale)
    if stability > PARAMETERS['max_leave_one_out_range']:
        return None
    return dict(ratio=scale, inliers=len(x), matches=len(matches), rotation=rotation,
        span_a=sx.tolist(), span_b=sy.tolist(), leave_one_out_relative_range=stability,
        points_a=x.tolist(), points_b=y.tolist(), **details)


def solve_scales(nodes, links, anchor, anchor_scale):
    """Keep the capture anchor exact; reject disconnected or inconsistent evidence."""
    index = {n:i for i,n in enumerate(n for n in nodes if n != anchor)}
    active = list(links)
    discarded = []
    for _ in range(5):
        rows, values = [], []
        for edge in active:
            row = np.zeros(len(index))
            weight = np.sqrt(min(edge['inliers'],100))*max(edge['span_b'])/max(edge['residual_p95_original_pixels'],.5)
            for name, sign in ((edge['a'],-1), (edge['b'],1)):
                if name != anchor:
                    row[index[name]] = sign*weight
            rows.append(row)
            values.append(np.log(edge['ratio'])*weight)
        if not rows:
            raise ValueError('没有可验证的尺度关系')
        solution, _, rank, _ = np.linalg.lstsq(rows, values, rcond=None)
        if rank != len(index):
            raise ValueError('存在未连接到实机锚点的底图，不能发布全量尺度')
        scales = {anchor:anchor_scale, **{n:float(anchor_scale*np.exp(solution[i])) for n,i in index.items()}}
        errors = [abs(float(np.log(scales[e['b']]/scales[e['a']]/e['ratio']))) for e in active]
        bad = [e for e,err in zip(active,errors) if err > PARAMETERS['max_cycle_error']]
        if not bad:
            for node in nodes:
                incident = [e for e in active if node in (e['a'],e['b'])]
                # A leaf must have broad, high-count shared geometry, not just one small corner.
                if node != anchor and len(incident) == 1:
                    edge = incident[0]
                    if edge['inliers'] < 30 or min(edge['span_a']+edge['span_b']) < 800:
                        raise ValueError(f'{node}: 单一局部关系不足，需要另一组几何证据')
            return scales, active, discarded, max(errors, default=0.)
        discarded.extend(bad)
        active = [e for e,err in zip(active,errors) if err <= PARAMETERS['max_cycle_error']]
    raise ValueError('多路径尺度不一致，未发布')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--anchor', default='layout_05_1st')
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    profile = MapCatalog().profile(*args.profile.split('/'))
    base = profile_root(profile.clean_root)
    if json.loads((base/'recognition_scales.json').read_text(encoding='utf-8')).get('zoom_derivation'):
        raise ValueError('已有第二档识别尺度；请在暂存副本重新标定最小尺度，再运行 add_recognition_zoom，避免覆盖旧证据')
    config_path = ROOT/'maintenance/configs/template_matching.json'
    config_hash = sha(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    layouts = config['floor_variants'][profile.key]
    anchor_layout, anchor_floor = args.anchor.rsplit('_',1)
    anchor = layouts[anchor_layout]['route_clean'][anchor_floor]
    if anchor.get('scale_status') != 'verified_capture' or anchor['scale_evidence']['screenshot_sha256'] != sha(args.screenshot):
        raise ValueError('锚点必须有与本次截图指纹相符的直接标定尺度')
    paths = sorted(p for p in profile.clean_root.glob('layout_*/*.png') if p.stem.rsplit('_',1)[-1] in profile.floors)
    source_hashes = {p.stem:sha(p) for p in paths}
    signature = dict(parameters=PARAMETERS, sources=source_hashes, opencv=cv2.__version__,
                     tool_sha256=sha(Path(__file__)), shared_tool_sha256=sha(ROOT/'tools/common/route_calibration.py'))
    cache = args.output/'links.json'
    cached = json.loads(cache.read_text(encoding='utf-8')) if args.reuse and cache.exists() else {}
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    if cached.get('signature') == signature:
        links = cached['links']
    else:
        prepared = {p.stem:features(p) for p in paths}
        links = []
        for i,a in enumerate(sorted(prepared)):
            for b in sorted(prepared):
                if a >= b:
                    continue
                link = match_link(prepared[a], prepared[b])
                if link:
                    links.append(dict(a=a,b=b,**link))
            print(f'{i+1}/{len(paths)} {a}: {len(links)} geometry links', flush=True)
        write(cache, dict(signature=signature, links=links))
    scales, active, discarded, error = solve_scales(sorted(source_hashes), links, args.anchor, anchor['saved_scale'])
    audit = dict(schema_version=1, profile=profile.key, signature=signature,
        anchor=dict(name=args.anchor, scale=anchor['saved_scale'], evidence=anchor['scale_evidence']),
        status='derived_from_shared_geometry', all_floors_ingame_verified=False,
        scales=scales, links=active, rejected_links=discarded, max_cycle_log_error=error)
    audit_path = base/'sources/screen_scale_calibration.json'
    staged_audit = args.output/'screen_scale_calibration.json'
    write(staged_audit, audit)
    dependencies = [dict(path=p.relative_to(ROOT).as_posix(), sha256=source_hashes[p.stem]) for p in paths]
    dependencies.append(dict(path=audit_path.relative_to(ROOT).as_posix(), sha256=sha(staged_audit)))
    existing = json.loads((base/'recognition_scales.json').read_text(encoding='utf-8'))
    recognition = dict(schema_version=1, derivation=dict(path='sources/screen_scale_calibration.json',
        sha256=sha(staged_audit), dependencies=[dict(path=p.relative_to(ROOT).as_posix(), sha256=source_hashes[p.stem]) for p in paths]), targets={})
    for path in paths:
        name = path.stem
        layout, floor = name.rsplit('_',1)
        scale = round(scales[name], 6)
        evidence = dict(method='shared_geometry_scale_transfer', anchor=args.anchor,
                        derivation_sha256=sha(staged_audit), direct_capture_verified=False)
        key = path.relative_to(profile.clean_root).as_posix()
        recognition['targets'][key] = (existing['targets'][key] if name == args.anchor else
            dict(source_sha256=source_hashes[name], scales=[scale], evidence=evidence))
        for version in profile.template_versions:
            item = layouts[layout][version][floor]
            # Route layers must already be in these exact clean-image coordinates.
            cal_path = base/'routes'/version/layout/'calibration.json'
            calibration = json.loads(cal_path.read_text(encoding='utf-8'))['floors'][floor]
            if calibration['status'] != 'PASS' or not np.allclose(calibration['asset_to_clean_homography'],np.eye(3),atol=1e-9):
                raise ValueError(f'{layout}/{floor}/{version}: 必须先完成底图坐标标定')
            if sha(ROOT/item['source_image']) != item['source_sha256']:
                raise ValueError(f'{layout}/{floor}/{version}: 分层资源已改变')
            for dependency in item.get('dependencies', []):
                if sha(ROOT/dependency['path']) != dependency['sha256']:
                    raise ValueError(f"依赖已改变，须先重新核对几何：{dependency['path']}")
            if name != args.anchor:
                item.update(saved_scale=scale, scale_status='derived_geometry',
                            scale_provenance='shared_structure_from_verified_capture', scale_evidence=evidence)
                merged = {d['path']:d for d in item.get('dependencies', [])}
                merged.update({d['path']:d for d in dependencies})
                item['dependencies'] = list(merged.values())
    write(args.output/'recognition_scales.json', recognition)
    write(args.output/'template_matching.json', config)
    if args.publish:
        if sha(config_path) != config_hash:
            raise ValueError('推算期间正式配置已改变，未发布')
        if any(sha(p) != source_hashes[p.stem] for p in paths):
            raise ValueError('推算期间底图已改变，未发布')
        # Preserve the exact previous formal configuration in this build directory.
        previous = args.output/'template_matching_before_publish.json'
        if not previous.exists():
            previous.write_bytes(config_path.read_bytes())
        write(audit_path, audit)
        write(base/'recognition_scales.json', recognition)
        write(config_path, config)
    print(json.dumps(dict(floors=len(scales), links=len(active), rejected_links=len(discarded),
                         scale_range=[min(scales.values()),max(scales.values())], published=args.publish)))


if __name__ == '__main__':
    main()
