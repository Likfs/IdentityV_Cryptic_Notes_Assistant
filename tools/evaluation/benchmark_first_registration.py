"""Current multiscale_structure independent-frame replay; reference poses are not ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from tools.common.paths import TOOL_CACHE_ROOT
from src.app_catalog import MapCatalog
from src.projection_workflow import process_screenshot, save_workflow_result
from src.projection.route_overlay import load_route_overlay_asset, warp_route_overlay, composite_route_layer
from src.registration.runtime_registration import prepare_registration_template

RUNS = ROOT / "runtime/diagnostics/route_projection/app_runs"
OUTPUT = ROOT / "workspace/reports/registration/formal_benchmarks"


def make_manifest(runs: Path = RUNS) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for path in sorted(runs.glob("*/result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            image = path.parent / "01_screenshot.png"
            validation = (data.get("registration") or {}).get("validation") or {}
            pixels = validation.get("visible_structure_pixels", 0)
            layout, floor = data.get("selected_layout"), data.get("floor")
            if not (layout and floor in ("b1", "1st", "2nd") and image.is_file()
                    and 2000 <= pixels <= 65000):
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            groups[(layout, floor)].append({
                "source": path.parent.name, "layout": layout, "floor": floor,
                "sha256": digest, "visible_pixels": pixels,
                "session_id": data.get("session_id"),
                "split": "holdout" if int(layout.split("_")[-1]) % 3 == 0 else "development",
            })
        except (OSError, ValueError, TypeError):
            continue
    selected = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["visible_pixels"])
        indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
        selected.extend(ordered[index] for index in indices)
        for row in group:
            if row["source"] == "20260926_033739_856989_layout_18_1st" and row not in selected:
                selected.append(row)
    return sorted(selected, key=lambda row: (row["layout"], row["floor"], row["source"]))


def valid_pose(matrix) -> bool:
    if matrix is None:
        return False
    try:
        value = np.asarray(matrix, np.float64)
        return value.shape == (2, 3) and bool(np.isfinite(value).all()) and np.linalg.det(value[:, :2]) != 0
    except (ValueError, TypeError):
        return False


def pose_delta(current: list | None, baseline: list | None, shape: tuple) -> float | None:
    """P95 screen displacement over the reference viewport, including scale drift."""
    if not valid_pose(current) or not valid_pose(baseline):
        return None
    a, b = np.asarray(current, np.float64), np.asarray(baseline, np.float64)
    xx, yy = np.meshgrid(np.linspace(0, shape[1] - 1, 15), np.linspace(0, shape[0] - 1, 15))
    screen = np.stack((xx.ravel(), yy.ravel()), axis=1).reshape(-1, 1, 2)
    clean = cv2.transform(screen, cv2.invertAffineTransform(b))
    mapped = cv2.transform(clean, a)
    delta = float(np.percentile(np.linalg.norm((mapped - screen).reshape(-1, 2), axis=1), 95))
    return delta if np.isfinite(delta) else None


def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=("cpu", "auto"), default="cpu",
                        help="cpu：当前正式 CPU；auto：GPU 优先并记录实际后端与回退原因")
    # Recognize old spellings only to provide a precise migration error.
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    for name in ("point-refinement", "candidate-pool", "geometry"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)


def reject_legacy_arguments(parser, args):
    names = ("mode", "point_refinement", "candidate_pool", "geometry", "config")
    if any(getattr(args, name) is not None for name in names):
        parser.error("历史 baseline/optimized、搜索开关和 --config 已停用；请改用 --backend cpu 或 --backend auto 评测当前 multiscale_structure")
    if args.limit < 0:
        parser.error("--limit 不能为负数")


def timing_summary(times):
    measured = [float(value) for value in times if value is not None and np.isfinite(value) and value >= 0]
    return {"timed_cases": len(measured), "missing_timing_cases": len(times) - len(measured),
            "p50_ms": float(np.median(measured)) if measured else None,
            "p95_ms": float(np.percentile(measured, 95)) if measured else None}


def backend_summary(infos):
    return {"effective_backends": dict(Counter(info.get("effective_backend", "not_run") for info in infos)),
            "fallback_reasons": sorted({info["fallback_reason"] for info in infos if info.get("fallback_reason")})}


def output_path(kind, backend):
    return OUTPUT / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{kind}_{backend}.json"


def validate_reference(manifest, report):
    records = report.get("rows", [])
    reference = {row["source"]: row for row in records}
    if not records or len(reference) != len(records) or set(reference) != {row["source"] for row in manifest}:
        raise ValueError("参考清单与当前筛选清单不一致，不能进行部分对比")
    for item in manifest:
        row = reference[item["source"]]
        if any(row.get(key) != item.get(key) for key in ("sha256", "layout", "floor")):
            raise ValueError(f"参考输入不一致：{item['source']}")
    return reference


def run_replay(args):
    manifest = (json.loads(args.manifest.read_text(encoding="utf-8-sig"))
                if args.manifest else make_manifest(args.runs))
    if not isinstance(manifest, list):
        raise ValueError("输入清单必须为 JSON 数组")
    manifest = [row for row in manifest if args.split == "all" or row.get("split") == args.split]
    if args.source:
        manifest = [row for row in manifest if row["source"] == args.source]
    if args.limit:
        manifest = manifest[:args.limit]
    if not manifest:
        raise ValueError("没有可评测样例：请提供存在的 --manifest，或检查诊断目录及筛选条件")
    if len({row["source"] for row in manifest}) != len(manifest):
        raise ValueError("清单包含重复的 source")
    references = validate_reference(manifest, json.loads(args.reference.read_text(encoding="utf-8-sig"))) if args.reference else {}
    catalog = MapCatalog()
    profile = catalog.profile("a_bringer_of_doom", "nightmare")
    images = {}
    for item in manifest:
        path = ROOT / item["image"] if item.get("image") else args.runs / item["source"] / "01_screenshot.png"
        if not path.is_file():
            raise ValueError(f"截图不存在：{path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"截图指纹与清单不一致：{path}")
        if item["floor"] not in ("b1", "1st", "2nd") or item["layout"] not in catalog.layouts(profile):
            raise ValueError(f"布局或楼层无效：{item['source']}")
        if not catalog.clean_template(profile, item["layout"], item["floor"]).is_file():
            raise ValueError(f"底图不存在：{item['source']}")
        images[item["source"]] = path
    prepared_cache, asset_cache, template_hashes, rows = {}, {}, {}, []
    output = args.output or output_path("replay", args.backend)
    cache_root = TOOL_CACHE_ROOT / "formal_benchmarks" / args.backend
    for index, item in enumerate(manifest):
        image_path = images[item["source"]]
        image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"不能解码截图：{image_path}")
        key = (profile.key, item["layout"], item["floor"])
        preparation_ms = 0.0
        if key not in prepared_cache:
            template_path = catalog.clean_template(profile, item["layout"], item["floor"])
            template_hashes[key] = hashlib.sha256(template_path.read_bytes()).hexdigest()
            started = time.perf_counter()
            prepared_cache[key] = prepare_registration_template(item["layout"], item["floor"], template_path, cache_root / profile.key)
            preparation_ms = (time.perf_counter() - started) * 1000
        previous = references.get(item["source"])
        if previous and previous.get("template_sha256") != template_hashes[key]:
            raise ValueError(f"参考底图指纹不一致：{item['source']}")
        result = process_screenshot(
            image, catalog=catalog, template_cache_root=cache_root, profile=profile, model_bundle=None,
            floor=item["floor"], layout_override=item["layout"], route_versions={}, opacity=.96,
            prepared_cache=prepared_cache, asset_cache=asset_cache, registration_backend=args.backend,
        )
        registration = result.registration
        matrix = registration.clean_to_screen if registration else None
        pose = matrix.tolist() if valid_pose(matrix) else None
        row = {**item, "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
               "template_sha256": template_hashes[key], "accepted": result.accepted, "pose_valid": pose is not None,
               "reasons": result.reasons, "timings_ms": result.timings_ms, "template_load_ms": preparation_ms,
               "validation": registration.validation if registration else {},
               "registration_timings": registration.timings if registration else {},
               "overlay_validation": result.overlay_validation, "clean_to_screen": pose,
               "backend_info": registration.validation.get("registration_backend", {}) if registration else {}}
        row["p95_screen_delta_vs_reference"] = (
            pose_delta(pose, previous.get("clean_to_screen"), image.shape)
            if previous and previous["accepted"] and row["accepted"] else None)
        row["reference_accepted"] = previous["accepted"] if previous else None
        rows.append(row)
        if args.save_differences and previous and (row["accepted"] != previous["accepted"] or
                (row["p95_screen_delta_vs_reference"] or 0) > 10 or
                (row["accepted"] and not row["pose_valid"])):
            directory = output.parent / (output.stem + "_differences") / f"case_{index:04d}"
            save_workflow_result(result, directory)
            if valid_pose(previous.get("clean_to_screen")):
                variant = catalog.selected_variant(profile, item["layout"], {})
                asset = load_route_overlay_asset(variant.calibration_path, item["floor"])
                layer, _ = warp_route_overlay(asset, np.asarray(previous["clean_to_screen"], np.float32),
                                               (image.shape[1], image.shape[0]), .75)
                preview, _ = composite_route_layer(image, layer)
                cv2.imencode(".png", preview)[1].tofile(str(directory / "08_reference_projection.png"))
        print(f"{index + 1}/{len(manifest)} {item['source']} accepted={row['accepted']} "
              f"backend={row['backend_info'].get('effective_backend')} delta={row['p95_screen_delta_vs_reference']}", flush=True)
    accepted = [row for row in rows if row["accepted"]]
    summary = {"cases": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted),
               "accepted_invalid_pose": sum(not row["pose_valid"] for row in accepted),
               "all_timing": timing_summary([row["timings_ms"].get("registration") for row in rows]),
               "accepted_timing": timing_summary([row["timings_ms"].get("registration") for row in accepted]),
               "over_10px_vs_reference": [row["source"] for row in rows if (row["p95_screen_delta_vs_reference"] or 0) > 10],
               "acceptance_changes": [row["source"] for row in rows if row["reference_accepted"] is not None and row["reference_accepted"] != row["accepted"]]}
    payload = {"report_version": 2, "algorithm": "multiscale_structure", "backend": args.backend,
               **backend_summary([row["backend_info"] for row in rows]), "history_used": False,
               "opencv_threads": cv2.getNumThreads(), "opencv_version": cv2.__version__,
               "comparison_file": str(args.reference) if args.reference else None,
               "reference": "explicit reference output, not ground truth", "summary": summary, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--manifest", type=Path, help="JSON 数组；可用 image 字段指定固定截图（相对项目根目录或绝对路径）")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source")
    parser.add_argument("--reference", type=Path, help="显式参考报告；须与筛选后的输入清单完全匹配")
    parser.add_argument("--save-differences", action="store_true")
    args = parser.parse_args(argv)
    reject_legacy_arguments(parser, args)
    if args.save_differences and not args.reference:
        parser.error("--save-differences 需要显式 --reference")
    cv2.setNumThreads(4)
    try:
        run_replay(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


