"""Check all clean floors against each calibrated route version without game captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.app_catalog import FLOORS, MapCatalog  # noqa: E402
from src.projection.route_overlay import load_route_overlay_asset, warp_route_overlay, warp_route_structure  # noqa: E402
from src.registration.runtime_registration import runtime_registration_config  # noqa: E402
from src.registration_core.color_segmenter import segment_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'workspace/reports/registration/route_asset_audit.json')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    catalog = MapCatalog()
    config = runtime_registration_config()
    rows = []
    for profile in catalog.profiles.values():
        for layout in catalog.layouts(profile):
            variants = catalog.variants(profile, layout)
            for floor in FLOORS:
                row = {"profile": profile.key, "layout": layout, "floor": floor,
                       "clean": None, "versions": []}
                rows.append(row)
                try:
                    clean_path = catalog.clean_template(profile, layout, floor)
                    clean = cv2.imdecode(np.fromfile(str(clean_path), np.uint8), cv2.IMREAD_COLOR)
                    if clean is None:
                        raise ValueError("干净底图无法解码")
                    row["clean"] = str(clean_path)
                except (RuntimeError, ValueError) as exc:
                    row["error"] = str(exc)
                    continue
                reference = segment_map(clean, config, infer_known_background=False,
                                        include_stairs=False).map_mask > 0
                for variant in variants:
                    item = {"version": variant.variant_id,
                            "status": variant.floor_status.get(floor)}
                    row["versions"].append(item)
                    if item["status"] != "PASS":
                        continue
                    try:
                        asset = load_route_overlay_asset(variant.calibration_path, floor)
                        matrix = asset.asset_to_clean
                        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
                            raise ValueError("路线标定矩阵无效")
                        layer, _ = warp_route_overlay(asset, np.eye(2, 3),
                                                      (clean.shape[1], clean.shape[0]), 1.0)
                        structure = warp_route_structure(asset, np.eye(3), (clean.shape[1], clean.shape[0]))
                        if structure is None:
                            route = segment_map(layer[:, :, :3], config,
                                                infer_known_background=False,
                                                include_stairs=False).map_mask > 0
                            route &= layer[:, :, 3] > 0
                        else:
                            route = structure > 127
                        observed = int(np.count_nonzero(reference))
                        covered = int(np.count_nonzero(reference & route))
                        distances = cv2.distanceTransform((~route).astype(np.uint8),
                                                          cv2.DIST_L2, 3)[reference]
                        item.update({
                            "clean_pixels": observed,
                            "route_pixels": int(np.count_nonzero(route)),
                            "coverage": round(covered / max(1, observed), 4),
                            "distance_p95_px": round(float(np.percentile(distances, 95)), 2)
                            if observed else None,
                            "alpha_coverage": round(float(np.count_nonzero(layer[:, :, 3])) /
                                                    max(1, layer.shape[0] * layer.shape[1]), 4),
                        })
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        item["error"] = str(exc)
    if not rows:
        parser.error("资源目录没有可审计楼层，不能生成空的成功报告")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    issues = [{"layout": row["layout"], "floor": row["floor"],
               "error": row.get("error"), "versions": row["versions"]}
              for row in rows if row.get("error") or not row["versions"]
              or any(item.get("error") or item.get("status") != "PASS" for item in row["versions"])]
    review = [{"layout": row["layout"], "floor": row["floor"],
               "version": item["version"], "coverage": item["coverage"],
               "distance_p95_px": item["distance_p95_px"]}
              for row in rows for item in row["versions"]
              if "coverage" in item and
              (item["coverage"] < 0.80 or item["distance_p95_px"] > 6.0)]
    review.sort(key=lambda item: (item["coverage"], -item["distance_p95_px"]))
    payload = {"summary": {"floors": len(rows),
                           "calibrated_versions": sum(len(row["versions"]) for row in rows),
                           "load_errors": len(issues), "review_count": len(review)},
               "issues": issues, "review": review, "floors": rows}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
