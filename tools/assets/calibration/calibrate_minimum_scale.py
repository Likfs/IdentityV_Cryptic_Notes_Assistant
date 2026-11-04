"""Offline fixed-scale trial; outputs a candidate overlay for human verification."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from src.registration.template_matching import matcher


def trial(clean, screenshot, scales, *, map_threshold=56, ds=.25):
    best = None
    attempts = []
    for scale in scales:
        try:
            match = matcher._match_template(screenshot, str(clean), float(scale), expand_search=True,
                                            map_threshold=map_threshold, ds=ds)
            attempts.append(dict(scale=float(scale), score=float(match['score'])))
            if best is None or match['score'] > best['score']:
                best = match
        except (ValueError, cv2.error) as exc:
            attempts.append(dict(scale=float(scale), error=str(exc)))
    if best is None:
        raise ValueError(f'没有可匹配的尺度：{attempts}')
    image = matcher.read_image(clean)
    x, y = best['location']
    s = best['scale']
    inverse = np.linalg.inv(np.float64([[s, 0, x], [0, s, y], [0, 0, 1]]))
    layer = cv2.warpPerspective(image, inverse, screenshot.shape[1::-1])
    preview = cv2.addWeighted(screenshot, .5, layer, .5, 0)
    return dict(scale=s, score=float(best['score']), location=list(best['location']),
                scale_direction='screenshot_to_source', reference_size=list(screenshot.shape[1::-1]),
                map_threshold=map_threshold, downsample=ds, verified=False, attempts=attempts), preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--scale', type=float, help='手动试用一个截图到源图的比例')
    parser.add_argument('--min', type=float, default=1.)
    parser.add_argument('--max', type=float, default=4.)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--map-threshold', type=int, choices=range(256), default=56,
                        metavar='0..255', help='底图灰度阈值；同画布深灰背景默认 56')
    parser.add_argument('--downsample', type=float, choices=matcher.TEMPLATE_DS_OPTIONS, default=.25,
                        help='离线匹配分辨率；细搜可使用 0.5 或 1.0')
    parser.add_argument('--reference-width', type=int, default=2560, help='截图时完整游戏画面的物理像素宽度')
    parser.add_argument('--reference-height', type=int, default=1440, help='截图时完整游戏画面的物理像素高度')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/scale_trial')
    args = parser.parse_args()
    values = (args.min, args.max, args.step) if args.scale is None else (args.scale,)
    if any(not np.isfinite(v) or v <= 0 for v in values) or args.max < args.min:
        parser.error('尺度和步长必须为有限正数，max 不小于 min')
    if args.scale is None and (args.max-args.min)/args.step > 2000:
        parser.error('一次最多试算 2001 个尺度，请缩小范围')
    scales = [args.scale] if args.scale is not None else np.arange(args.min, args.max+args.step/2, args.step)
    result, preview = trial(args.clean, matcher.read_image(args.screenshot), scales,
                            map_threshold=args.map_threshold, ds=args.downsample)
    if min(args.reference_width, args.reference_height) <= 0:
        parser.error('参考分辨率必须为正数')
    result['capture_size'] = result.pop('reference_size')
    result['reference_size'] = [args.reference_width, args.reference_height]
    result['minimum_scale'] = result['scale'] * min(args.reference_width/2560, args.reference_height/1440)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', preview)[1].tofile(str(args.output/'preview.png'))
    (args.output/'candidate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('已保存候选尺度与 preview.png；请人工核对，再按参考分辨率换算后导入 minimum_scale。')


if __name__ == '__main__':
    main()
