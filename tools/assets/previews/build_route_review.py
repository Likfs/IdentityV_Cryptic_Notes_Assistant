"""Build a standalone manual review page for asset warnings and registration samples."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from src.app_catalog import MapCatalog, FLOOR_NAMES
from src.projection.route_overlay import load_route_overlay_asset, warp_route_overlay, warp_route_structure
from src.registration.runtime_registration import prepare_registration_template, runtime_registration_config
from src.registration.runtime_registration import register_screenshot as legacy_register
from src.registration.multiscale_structure.runtime import register_screenshot as current_register
from src.registration_core.color_segmenter import segment_map
from tools.evaluation.benchmark_first_registration import pose_delta


def read(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'无法读取：{path}')
    return image


def uri(image):
    ratio = min(1., 1600/max(image.shape[:2]))
    if ratio < 1:
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', image)[1]).decode('ascii')


def fingerprint(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def image_layer(image, matrix, size):
    alpha = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 48)*255
    return cv2.warpAffine(np.dstack((image, alpha)), np.asarray(matrix, np.float64)[:2], size)


def asset_cases(audit):
    catalog = MapCatalog()
    profile = catalog.profile('a_bringer_of_doom', 'nightmare')
    config = runtime_registration_config()
    cases = []
    for warning in audit['review']:
        layout, floor = warning['layout'], warning['floor']
        path = catalog.clean_template(profile, layout, floor)
        clean = read(path)[:, :, :3]
        variant = next(v for v in catalog.variants(profile, layout) if v.variant_id == warning['version'])
        asset = load_route_overlay_asset(variant.calibration_path, floor)
        size = clean.shape[:2][::-1]
        layer, _ = warp_route_overlay(asset, np.eye(3), size, 1.)
        reference = segment_map(clean, config, infer_known_background=False, include_stairs=False).map_mask > 0
        structure = warp_route_structure(asset, np.eye(3), size)
        route = (structure > 127) if structure is not None else (
            (segment_map(layer[:, :, :3], config, infer_known_background=False, include_stairs=False).map_mask > 0)
            & (layer[:, :, 3] > 0))
        coverage = round(np.count_nonzero(reference & route)/max(1, np.count_nonzero(reference)), 4)
        distances = cv2.distanceTransform(np.uint8(~route), cv2.DIST_L2, 3)[reference]
        p95 = round(float(np.percentile(distances, 95)), 2) if distances.size else None
        if coverage != warning['coverage'] or p95 != warning['distance_p95_px']:
            raise ValueError(f'{layout}/{floor}: 资源与审计指标已不一致，请重新运行 audit_route_assets.py')
        difference = np.zeros((*reference.shape, 3), np.uint8)
        difference[reference & route] = (180, 180, 180)
        difference[reference & ~route] = (180, 50, 255)  # magenta: clean-only
        difference[route & ~reference] = (230, 215, 40)  # cyan: route-only
        cases.append(dict(id=f'asset:{layout}:{floor}', group='资产 REVIEW',
            title=f'{layout} · {FLOOR_NAMES[floor]} · 凉哈皮分层路线',
            description='检查墙角、门口、走廊是否系统性偏移。色块缺失也可能来自画法或结构分割；请区分位置偏差与内容差异。',
            metrics=f"覆盖率 {warning['coverage']:.2%} · 距离 P95 {warning['distance_p95_px']:.2f} 底图像素",
            base_name='干净底图', overlay_name='按现有标定投到同一底图的路线图',
            width=size[0], height=size[1], base=uri(clean), overlay=uri(layer), difference=uri(difference),
            evidence=[fingerprint(p) for p in (path, asset.asset_path, asset.calibration_path)], details=warning))
    return cases


def registration_cases(output):
    catalog = MapCatalog()
    profile = catalog.profile('a_bringer_of_doom', 'nightmare')
    cases = []
    definitions = [
        ('layout_11', '稀疏候选冲突：预期接受', [.9116103649139404, 237.01605224609375, -790.8101806640625], [265, 585, 290, 710]),
        ('layout_16', '已复核房间：预期补充接受', None, None),
        ('layout_13', '区域竞争位置：预期拒绝', [1.0274015665054321, 78.7654037475586, -596.2371826171875], [335, 515, 380, 620]),
    ]
    for layout, label, pose, region in definitions:
        path = catalog.clean_template(profile, layout, 'b1')
        prepared = prepare_registration_template(layout, 'b1', path, output/'cache')
        if pose:
            scale, x, y = pose
            truth = np.float64([[scale, 0, x], [0, scale, y]])
            template = prepared.processed
            canvas = np.full_like(template.image, 15)
            canvas[template.corridor_mask > 0] = (150, 142, 126)
            canvas[template.room_mask > 0] = (65, 103, 142)
            processed_to_screen = truth.copy()
            processed_to_screen[:, :2] /= prepared.registration_scale
            processed_to_screen[:, 2] += processed_to_screen[:, :2] @ np.float32(template.crop_offset)
            complete = cv2.warpAffine(canvas, processed_to_screen, (1000, 850), borderValue=(15, 15, 15))
            screenshot = np.full_like(complete, 15)
            y1, y2, x1, x2 = region
            screenshot[y1:y2, x1:x2] = complete[y1:y2, x1:x2]
            result = legacy_register(screenshot, prepared)
            evidence = [fingerprint(path)]
        else:
            fixture = ROOT/'tests/fixtures/registration/multiscale_structure/layout_16_b1_semantic_positive.json'
            screenshot = read(fixture.with_suffix('.png'))[:, :, :3]
            truth = np.float64(json.loads(fixture.read_text(encoding='utf-8'))['clean_to_screen'])
            result = current_register(screenshot, prepared, backend='cpu')
            evidence = [fingerprint(p) for p in (path, fixture, fixture.with_suffix('.png'))]
        diagnostics = result.diagnostics()
        (output/f'{layout}_cpu.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')
        size = screenshot.shape[:2][::-1]
        reference = image_layer(prepared.raw_image, truth, size)
        actual = image_layer(prepared.raw_image, result.clean_to_screen, size) if result.clean_to_screen is not None else None
        delta = pose_delta(result.clean_to_screen, truth, screenshot.shape)
        explanation = (' 当前被拒绝，诊断候选矩阵仅供比较，不是已接受的投影。' if actual is not None and not result.accepted else
                       ' 当前没有返回矩阵，仅显示参考位置。' if actual is None else
                       ' 当前被接受，可比较当前矩阵与参考矩阵。')
        if layout == 'layout_13' and delta is not None and delta < 10:
            explanation += ' 本次矩阵接近合成真值；拒绝断言失败不等于仍存在历史的约 30 像素误配准。'
        cases.append(dict(id=f'regression:{layout}', group='配准回归样例', title=f'{layout} · 地下室 · {label}',
            description=('此页重放旧 CPU 入口的已知变换合成截图。参考矩阵来自测试构造，不是实机真值。' if pose else
                         '此页以 CPU 重放正式多尺度流程。参考矩阵来自固定样例中的历史人工复核记录，仍须结合当前底图检查。')+
                        explanation,
            metrics=' · '.join(filter(None, [('当前接受' if result.accepted else '当前拒绝'),
                              f'与参考矩阵的视口 P95 位移 {delta:.2f} 像素' if delta is not None else None,
                              '；'.join(result.rejection_reasons)])),
            base_name='测试输入截图', overlay_name='参考矩阵投影的底图', width=size[0], height=size[1],
            base=uri(screenshot), overlay=uri(reference), actual=uri(actual) if actual is not None else None,
            actual_name='当前接受矩阵投影的底图' if result.accepted else '已拒绝的诊断候选矩阵投影的底图',
            evidence=evidence, details=dict(accepted=result.accepted, reasons=result.rejection_reasons,
                                           reference_matrix=truth.tolist(), actual_matrix=diagnostics['clean_to_screen'],
                                           reference_viewport_p95_displacement_px=delta,
                                           entry='legacy_cpu' if pose else 'current_multiscale_cpu')))
    return cases


def template_cases(report):
    cases = []
    for label, name in [('lhp_clean', '凉哈皮单人速刷路线图底图版'), ('zhanshi_clean', '展十速刷路线图底图版')]:
        folder = report/label
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if not result['accepted']:
            raise ValueError(f'{label}: 固定样例投影没有通过')
        base = read(folder/'01_screenshot.png')[:, :, :3]
        layer = read(folder/'06_overlay_layer.png')
        # Saved layer retains the actual production opacity.
        cases.append(dict(id=f'template:{label}', group='新增定尺度样例', title=f'layout_05 · {name}',
            description='实际离线投影产物。检查门口、墙角及路线位置；这是固定截图验证，尚不代表游戏内所有布局均已对齐。',
            metrics='固定样例接受 · floor=all · 原图坐标投影', base_name='固定游戏截图', overlay_name='保存的定尺度投影层（含运行不透明度）',
            width=base.shape[1], height=base.shape[0], base=uri(base), overlay=uri(layer),
            evidence=[fingerprint(p) for p in (folder/'result.json', Path(result['asset']))],
            details={key: result[key] for key in ('variant_id', 'accepted', 'floor', 'variant_source')}))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'workspace/reports/template_source_variants_20261003')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/reports/manual_route_review')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    audit = json.loads((args.report/'route_asset_audit.json').read_text(encoding='utf-8'))
    cases = asset_cases(audit)+registration_cases(args.output)+template_cases(args.report)
    payload = dict(cases=cases, audit_summary=audit['summary'])
    template = Path(__file__).with_name('route_review.html').read_text(encoding='utf-8')
    html = template.replace('__REVIEW_DATA__', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/'))
    target = args.output/'index.html'
    target.write_text(html, encoding='utf-8')
    print(f'{len(cases)} review cases: {target}')


if __name__ == '__main__':
    main()
