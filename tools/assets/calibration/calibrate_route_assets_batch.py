from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.assets.image_io import imread_unicode, imwrite_unicode  # noqa: E402
from tools.common.route_calibration import estimate_template_to_route  # noqa: E402


from tools.common.calibration_assets import (CLEAN_ROOT, OUTPUT_ROOT, FLOORS, calibration_root,
                                             project_relative, leading_number, floor_template)
ROUTE_ROOT = PROJECT_ROOT / "maintenance/maps/a_bringer_of_doom/nightmare/sources/route_lhp_solo"
ASSET_ROOT = calibration_root()


SEARCH_RATIOS = {
    "2nd": (0.00, 0.38),
    "1st": (0.25, 0.43),
    "b1": (0.51, 0.39),
}


@dataclass
class Calibration:
    floor: str
    template_path: Path
    clean_to_route: np.ndarray
    route_to_clean: np.ndarray
    diagnostics: dict
    center_y: float
    route_bbox: tuple[int, int, int, int]
    elapsed_ms: float
    status: str


def route_variants(layout_number: int, route_files: list[Path]) -> list[Path]:
    return sorted((path for path in route_files if leading_number(path) == layout_number), key=lambda path: path.name)


def transformed_geometry(template: np.ndarray, homography: np.ndarray, route_shape: tuple[int, ...]) -> tuple[float, tuple[int, int, int, int]]:
    height, width = template.shape[:2]
    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    transformed = cv2.perspectiveTransform(corners, homography)[0]
    min_xy = np.floor(transformed.min(axis=0)).astype(int)
    max_xy = np.ceil(transformed.max(axis=0)).astype(int)
    route_height, route_width = route_shape[:2]
    x0 = int(np.clip(min_xy[0], 0, route_width - 1))
    y0 = int(np.clip(min_xy[1], 0, route_height - 1))
    x1 = int(np.clip(max_xy[0] + 1, x0 + 1, route_width))
    y1 = int(np.clip(max_xy[1] + 1, y0 + 1, route_height))
    center = cv2.perspectiveTransform(np.float32([[[width / 2.0, height / 2.0]]]), homography)[0, 0]
    return float(center[1]), (x0, y0, x1 - x0, y1 - y0)


def quality_status(diagnostics: dict, center_y: float, search_crop: tuple[int, int, int, int]) -> str:
    refinement = diagnostics.get("feature_refinement", {})
    if not refinement.get("accepted", False):
        return "REVIEW"
    inliers = int(refinement.get("inliers", 0))
    median_error = float(refinement.get("median_inlier_error_px", 999.0))
    p95_error = float(refinement.get("p95_inlier_error_px", 999.0))
    template_score = float(diagnostics.get("score", 0.0))
    # Small basement maps naturally contain fewer repeatable feature points.
    # Accept them only when an independent whole-template score and both error
    # statistics strongly confirm the same transform.
    sparse_structure_confirmed = (
        inliers >= 12
        and template_score >= 0.96
        and median_error <= 3.0
        and p95_error <= 6.0
    )
    if inliers < 20 and not sparse_structure_confirmed:
        return "REVIEW"
    if median_error > 6.0:
        return "REVIEW"
    _, search_y, _, search_h = search_crop
    if not search_y <= center_y <= search_y + search_h:
        return "REVIEW"
    return "PASS"


def make_rgba_asset(route_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(route_crop, cv2.COLOR_BGR2HSV)
    alpha = np.full(route_crop.shape[:2], 145, dtype=np.uint8)
    alpha[hsv[:, :, 2] >= 50] = 255
    alpha[(hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 60)] = 255
    return np.dstack((route_crop, alpha))


def write_alignment_preview(
    destination: Path,
    route: np.ndarray,
    template: np.ndarray,
    clean_to_route: np.ndarray,
    crop: tuple[int, int, int, int],
    status: str,
) -> None:
    warped = cv2.warpPerspective(template, clean_to_route, (route.shape[1], route.shape[0]), flags=cv2.INTER_CUBIC)
    preview = route.copy()
    visible = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 20
    preview[visible] = cv2.addWeighted(preview, 0.38, warped, 0.62, 0)[visible]
    x, y, w, h = crop
    floor_view = preview[y : y + h, x : x + w].copy()
    color = (60, 220, 60) if status == "PASS" else (0, 180, 255)
    cv2.rectangle(floor_view, (0, 0), (floor_view.shape[1] - 1, floor_view.shape[0] - 1), color, 4)
    cv2.putText(floor_view, status, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    imwrite_unicode(destination, floor_view)


def calibrate_floor(
    route: np.ndarray,
    route_path: Path,
    template_path: Path,
    floor: str,
) -> Calibration:
    template = imread_unicode(template_path)
    if template is None:
        raise RuntimeError(f"无法读取 {template_path}")
    start_ratio, height_ratio = SEARCH_RATIOS[floor]
    search_y = int(round(route.shape[0] * start_ratio))
    search_h = min(route.shape[0] - search_y, int(round(route.shape[0] * height_ratio)))
    search_crop = (0, search_y, route.shape[1], search_h)
    started = time.perf_counter()
    clean_to_route, diagnostics = estimate_template_to_route(template, route, search_crop)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    route_to_clean = np.linalg.inv(clean_to_route)
    center_y, route_bbox = transformed_geometry(template, clean_to_route, route.shape)
    status = quality_status(diagnostics, center_y, search_crop)
    refinement = diagnostics.get("feature_refinement", {})
    diagnostics["quality_gate"] = (
        "standard_feature_support"
        if status == "PASS" and int(refinement.get("inliers", 0)) >= 20
        else "sparse_structure_confirmed"
        if status == "PASS"
        else "review"
    )
    diagnostics["route_file"] = project_relative(route_path)
    diagnostics["template_file"] = project_relative(template_path)
    diagnostics["floor"] = floor
    diagnostics["center_y"] = center_y
    diagnostics["route_bbox"] = list(route_bbox)
    return Calibration(
        floor=floor,
        template_path=template_path,
        clean_to_route=clean_to_route,
        route_to_clean=route_to_clean,
        diagnostics=diagnostics,
        center_y=center_y,
        route_bbox=route_bbox,
        elapsed_ms=elapsed_ms,
        status=status,
    )


def floor_bands(route_height: int, calibrations: dict[str, Calibration]) -> dict[str, tuple[int, int, int, int]]:
    centers = [calibrations[floor].center_y for floor in FLOORS]
    boundary_top_middle = int(round((centers[0] + centers[1]) / 2.0))
    boundary_middle_bottom = int(round((centers[1] + centers[2]) / 2.0))
    boxes = {floor: calibrations[floor].route_bbox for floor in FLOORS}

    def bounds(floor: str, start: int, end: int) -> tuple[int, int, int, int]:
        _, box_y, _, box_height = boxes[floor]
        # A midpoint between floor centers can cut through a floor's map.
        top = max(0, min(start, box_y))
        bottom = min(route_height, max(end, box_y + box_height))
        return (0, top, 0, max(1, bottom - top))

    bottom_bbox = boxes["b1"]
    bottom_content = bottom_bbox[1] + bottom_bbox[3] + 90
    bottom_end = min(route_height, max(boundary_middle_bottom + 100, bottom_content))
    return {
        "2nd": bounds("2nd", 0, boundary_top_middle),
        "1st": bounds("1st", boundary_top_middle, boundary_middle_bottom),
        "b1": bounds("b1", boundary_middle_bottom, bottom_end),
    }


def process_variant(layout_dir: Path, route_path: Path, variant_index: int) -> dict:
    layout_number = int(layout_dir.name.split("_")[-1])
    route = imread_unicode(route_path)
    if route is None:
        raise RuntimeError(f"无法读取 {route_path}")
    variant_id = "route_lhp_solo" if variant_index == 1 else f"route_lhp_solo_{variant_index:02d}"
    report_dir = OUTPUT_ROOT / layout_dir.name / variant_id
    asset_dir = ASSET_ROOT / variant_id / layout_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    calibrations: dict[str, Calibration] = {}
    for floor in FLOORS:
        calibration = calibrate_floor(route, route_path, floor_template(layout_dir, floor), floor)
        calibrations[floor] = calibration

    ordered = calibrations["2nd"].center_y < calibrations["1st"].center_y < calibrations["b1"].center_y
    if not ordered:
        for calibration in calibrations.values():
            calibration.status = "REVIEW"
    bands = floor_bands(route.shape[0], calibrations)
    for floor, (x, y, _, h) in bands.items():
        bands[floor] = (0, y, route.shape[1], min(h, route.shape[0] - y))

    floor_payloads = {}
    for floor in FLOORS:
        calibration = calibrations[floor]
        template = imread_unicode(calibration.template_path)
        crop_x, crop_y, crop_w, crop_h = bands[floor]
        route_crop = route[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        asset_path = asset_dir / f"{floor}.png"
        if calibration.status == "PASS":
            rgba = make_rgba_asset(route_crop)
            imwrite_unicode(asset_path, rgba)
            serialized_asset: str | None = project_relative(asset_path)
        else:
            # 上一次批处理可能留下低置信度产物；不能让运行时误用。
            if asset_path.exists():
                asset_path.unlink()
            serialized_asset = None
        preview_path = report_dir / f"{floor}_alignment.png"
        write_alignment_preview(preview_path, route, template, calibration.clean_to_route, bands[floor], calibration.status)
        crop_to_route = np.array(
            [[1.0, 0.0, float(crop_x)], [0.0, 1.0, float(crop_y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        asset_to_clean = calibration.route_to_clean @ crop_to_route
        floor_payloads[floor] = {
            "status": calibration.status,
            "template": project_relative(calibration.template_path),
            "asset": serialized_asset,
            "alignment_preview": project_relative(preview_path),
            "source_crop_xywh": list(bands[floor]),
            "clean_to_route_homography": calibration.clean_to_route.tolist(),
            "route_to_clean_homography": calibration.route_to_clean.tolist(),
            "asset_to_clean_homography": asset_to_clean.tolist(),
            "calibration_ms": calibration.elapsed_ms,
            "diagnostics": calibration.diagnostics,
        }

    payload = {
        "layout": layout_dir.name,
        "layout_number": layout_number,
        "variant_id": variant_id,
        "route": project_relative(route_path),
        "route_size": [route.shape[1], route.shape[0]],
        "floor_order_valid": ordered,
        "floors": floor_payloads,
    }
    (asset_dir / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def write_summary(payloads: list[dict], failures: list[dict]) -> None:
    summary = {
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": payloads,
        "failures": failures,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    pass_count = 0
    review_items = []
    for payload in payloads:
        for floor in FLOORS:
            item = payload["floors"][floor]
            if item["status"] == "PASS":
                pass_count += 1
            else:
                review_items.append(f"{payload['layout']} / {floor}")
            diagnostics = item["diagnostics"].get("feature_refinement", {})
            rows.append(
                "| {layout} | {variant} | {floor} | {status} | {inliers} | {median:.2f} | {elapsed:.0f} |".format(
                    layout=payload["layout"],
                    variant=payload["variant_id"],
                    floor=floor,
                    status=item["status"],
                    inliers=diagnostics.get("inliers", 0),
                    median=float(diagnostics.get("median_inlier_error_px", 999.0)),
                    elapsed=float(item["calibration_ms"]),
                )
            )
    markdown = [
        "# 路线资产批量标定结果",
        "",
        f"- 路线变体：{len(payloads)}",
        f"- 楼层结果：{len(rows)}",
        f"- 可用透明资产：{pass_count}",
        f"- 未生成资产、需要处理：{len(review_items)}",
        f"- 执行失败：{len(failures)}",
        "",
        "| 布局 | 变体 | 楼层 | 状态 | 内点 | 中位误差(px) | 标定耗时(ms) |",
        "|---|---|---:|---|---:|---:|---:|",
        *rows,
    ]
    if review_items:
        markdown.extend(["", "## 待处理", ""])
        for item in review_items:
            markdown.append(f"- `{item}`：自动标定未通过，未输出运行时资产。")
    if failures:
        markdown.extend(["", "## 失败", ""])
        for failure in failures:
            markdown.append(f"- `{failure['layout']}` / `{failure['route']}`：{failure['error']}")
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    global ASSET_ROOT
    parser = argparse.ArgumentParser(description="批量制作路线资产，默认输出到 workspace/asset_build")
    parser.add_argument("--publish", action="store_true", help="明确将结果写入正式资产目录")
    args = parser.parse_args()
    ASSET_ROOT = calibration_root(publish=args.publish)
    route_files = sorted((path for path in ROUTE_ROOT.iterdir() if path.is_file()), key=lambda path: path.name)
    layout_dirs = sorted((path for path in CLEAN_ROOT.iterdir() if path.is_dir()), key=lambda path: path.name)
    payloads = []
    failures = []
    for layout_dir in layout_dirs:
        layout_number = int(layout_dir.name.split("_")[-1])
        variants = route_variants(layout_number, route_files)
        if not variants:
            failures.append({"layout": layout_dir.name, "route": "", "error": "没有对应路线图"})
            continue
        for index, route_path in enumerate(variants, start=1):
            print(f"[{layout_dir.name}] {index}/{len(variants)} {route_path.name}", flush=True)
            try:
                payload = process_variant(layout_dir, route_path, index)
                payloads.append(payload)
                statuses = ", ".join(f"{floor}:{payload['floors'][floor]['status']}" for floor in FLOORS)
                print(f"  {statuses}", flush=True)
            except Exception as exc:  # keep the batch moving and report exact failures
                failures.append({"layout": layout_dir.name, "route": route_path.name, "error": str(exc)})
                print(f"  FAILED: {exc}", flush=True)
    write_summary(payloads, failures)
    print(json.dumps({"variants": len(payloads), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
