from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.assets.image_io import read_image as read_color, write_image
from src.projection.route_overlay import homogeneous, load_route_overlay_asset, project_route_overlay  # noqa: E402






def registration_matrix(payload: dict) -> tuple[np.ndarray, str]:
    registration = payload.get("registration")
    source = registration if isinstance(registration, dict) else payload
    if source.get("validation", {}).get("coordinate_space") == "original_source_image":
        raise ValueError("此报告使用完整原图坐标，不能作为干净底图配准矩阵")
    for key in (
        "clean_to_screen",
        "structure_raw_template_to_screenshot_affine",
        "raw_template_to_screenshot_affine",
        "clean_to_screen_homography",
        "clean_to_screen_affine",
    ):
        value = source.get(key)
        if value is not None:
            matrix = homogeneous(np.asarray(value, dtype=np.float64))
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
                raise ValueError("配准矩阵无效")
            return matrix, ("registration." if source is registration else "") + key
    raise KeyError("配准结果中没有干净底图到截图的矩阵")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用离线标定资产，把原路线图投影到游戏截图。")
    parser.add_argument("--layout", required=True, help="例如 layout_05")
    parser.add_argument("--floor", required=True, choices=("2nd", "1st", "b1"))
    parser.add_argument("--variant", default="route_lhp_solo")
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opacity", type=float, default=0.96)
    parser.add_argument("--benchmark-runs", type=int, default=31)
    parser.add_argument("--opencv-threads", type=int, default=4)
    args = parser.parse_args()

    from src.app_catalog import canonical_route_version
    cv2.setNumThreads(max(1, args.opencv_threads))
    calibration = (
        PROJECT_ROOT
        / "assets/maps/a_bringer_of_doom/nightmare/routes"
        / canonical_route_version(args.variant)
        / args.layout
        / "calibration.json"
    )
    asset = load_route_overlay_asset(calibration, args.floor)
    screenshot = read_color(args.screenshot.resolve())
    registration = json.loads(args.registration.resolve().read_text(encoding="utf-8"))
    if registration.get("selected_layout", registration.get("layout", args.layout)) != args.layout:
        parser.error("报告布局与 --layout 不一致")
    if registration.get("floor", args.floor) != args.floor:
        parser.error("报告楼层与 --floor 不一致")
    clean_to_screen, matrix_key = registration_matrix(registration)

    project_route_overlay(screenshot, asset, clean_to_screen, args.opacity)
    timings = []
    projection = alpha = matrix = None
    for _ in range(max(1, args.benchmark_runs)):
        started = time.perf_counter()
        projection, alpha, matrix = project_route_overlay(screenshot, asset, clean_to_screen, args.opacity)
        timings.append((time.perf_counter() - started) * 1000.0)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_image(output / "projection.png", projection)
    write_image(output / "alpha.png", alpha)
    result = {
        "layout": asset.layout,
        "floor": asset.floor,
        "variant": asset.variant,
        "screenshot": str(args.screenshot.resolve()),
        "registration": str(args.registration.resolve()),
        "registration_matrix_key": matrix_key,
        "calibration": str(asset.calibration_path),
        "asset": str(asset.asset_path),
        "asset_to_screen_homography": matrix.tolist(),
        "opacity": args.opacity,
        "opencv_threads": cv2.getNumThreads(),
        "runtime_ms_median": float(np.median(timings)),
        "runtime_ms_p95": float(np.percentile(timings, 95)),
        "benchmark_runs": len(timings),
        "projection": str((output / "projection.png").resolve()),
        "alpha": str((output / "alpha.png").resolve()),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
