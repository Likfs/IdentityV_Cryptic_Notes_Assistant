"""Known-pose current multiscale_structure stress test; synthetic occlusion is not real game fog."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template
from src.registration.multiscale_structure.runtime import register_screenshot
from tools.common.paths import TOOL_CACHE_ROOT
from tools.evaluation.benchmark_first_registration import (
    add_backend_arguments, reject_legacy_arguments, pose_delta, valid_pose,
    timing_summary, backend_summary, output_path,
)


def summarize(outcomes):
    accepted = [item for item in outcomes if item["accepted"]]
    def invalid(item):
        return (item["p95_screen_error_px"] is None or not np.isfinite(item["p95_screen_error_px"])
                or not valid_pose(item["clean_to_screen"]))
    return {"cases": len(outcomes), "accepted": len(accepted), "rejected": len(outcomes) - len(accepted),
            "accepted_invalid_pose": sum(invalid(item) for item in accepted),
            "accepted_over_10px": sum(not invalid(item) and item["p95_screen_error_px"] > 10 for item in accepted),
            "accepted_correct": sum(not invalid(item) and item["p95_screen_error_px"] <= 10 for item in accepted),
            "all_timing": timing_summary([item["registration_ms"] for item in outcomes]),
            "accepted_timing": timing_summary([item["registration_ms"] for item in accepted])}


def run_synthetic(args):
    catalog = MapCatalog()
    profile = catalog.profile("a_bringer_of_doom", "nightmare")
    rng = np.random.default_rng(260926)
    rows = []
    pairs = [(layout, floor) for layout in catalog.layouts(profile) for floor in ("b1", "1st", "2nd")]
    if args.limit:
        pairs = pairs[:args.limit]
    if args.case and args.case not in {f"{layout}/{floor}/{level}" for layout, floor in pairs for level in ("narrow", "partial", "spread")}:
        raise ValueError("--case 未匹配任何样例（同时检查 --limit），不能运行空评测")
    if not pairs:
        raise ValueError("没有可评测布局")
    cache_root = TOOL_CACHE_ROOT / "formal_benchmarks" / args.backend / profile.key
    for layout, floor in pairs:
        template_path = catalog.clean_template(profile, layout, floor)
        if not template_path.is_file():
            raise ValueError(f"底图不存在：{template_path}")
        prepared = prepare_registration_template(layout, floor, template_path, cache_root)
        template = prepared.processed
        canvas = np.full_like(template.image, 15)
        canvas[template.corridor_mask > 0] = (150, 142, 126)
        canvas[template.room_mask > 0] = (65, 103, 142)
        yy, xx = np.nonzero(template.room_mask)
        if not len(xx):
            raise ValueError(f"底图缺少可生成样例的房间像素：{layout}/{floor}")
        center_index = int(rng.integers(0, len(xx)))
        anchor = np.float32([xx[center_index], yy[center_index]])
        scale = float(rng.uniform(.85, 1.25))
        width, height = 1000, 850
        transform = np.float32([[scale, 0, anchor[0] - scale * width/2],
                                 [0, scale, anchor[1] - scale * height/2]])
        complete = cv2.warpAffine(canvas, cv2.invertAffineTransform(transform), (width, height),
                                   borderValue=(15, 15, 15))
        to_clean = transform.copy()
        to_clean[:, 2] += np.float32(template.crop_offset)
        to_clean /= prepared.registration_scale
        reference = cv2.invertAffineTransform(to_clean).tolist()
        for level, patch_w, patch_h in (("narrow", 180, 60), ("partial", 240, 180), ("spread", 420, 320)):
            if args.case and args.case != f"{layout}/{floor}/{level}":
                continue
            screenshot = np.full_like(complete, 15)
            left, top = (width-patch_w)//2, (height-patch_h)//2
            screenshot[top:top+patch_h, left:left+patch_w] = complete[top:top+patch_h, left:left+patch_w]
            result = register_screenshot(screenshot, prepared, backend=args.backend)
            matrix = result.clean_to_screen
            pose = matrix.tolist() if valid_pose(matrix) else None
            error = pose_delta(pose, reference, screenshot.shape)
            outcome = {"accepted": result.accepted, "p95_screen_error_px": error,
                       "pose_valid": pose is not None, "registration_ms": result.timings.get("total"),
                       "method": result.timings.get("method"), "reasons": result.rejection_reasons,
                       "validation": result.validation, "clean_to_screen": pose,
                       "backend_info": result.validation.get("registration_backend", {})}
            rows.append({"layout": layout, "floor": floor, "level": level,
                         "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
                         "screenshot_sha256": hashlib.sha256(screenshot.tobytes()).hexdigest(),
                         "reference_clean_to_screen": reference, "outcomes": {args.backend: outcome}})
        print(layout, floor, "done", flush=True)
    if not rows:
        raise ValueError("没有生成可评测样例")
    outcomes = [row["outcomes"][args.backend] for row in rows]
    payload = {"report_version": 2, "algorithm": "multiscale_structure", "backend": args.backend,
               **backend_summary([item["backend_info"] for item in outcomes]),
               "reference": "known synthetic transform, not real-game accuracy", "history_used": False,
               "opencv_threads": cv2.getNumThreads(), "opencv_version": cv2.__version__,
               "summary": summarize(outcomes), "rows": rows}
    output = args.output or output_path("synthetic", args.backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--limit", type=int, default=0, help="限制布局/楼层对数；每对生成三个遮挡样例")
    parser.add_argument("--case", help="layout_XX/floor/level；保持固定 RNG 序列")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reject_legacy_arguments(parser, args)
    cv2.setNumThreads(4)
    try:
        run_synthetic(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
