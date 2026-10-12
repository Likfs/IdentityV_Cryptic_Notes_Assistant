"""Calibrate the three independently positioned floors in the Zhanshi montage set."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.common.calibration_assets import (  # noqa: E402
    route_asset_root, layout_numbers, CLEAN_ROOT, FLOORS, OUTPUT_ROOT, floor_template,
    leading_number, project_relative,
)
from tools.common.paths import asset_output
from tools.common.route_calibration import estimate_template_to_route  # noqa: E402
from tools.assets.image_io import imread_unicode, imwrite_unicode  # noqa: E402

ASSET_ROOT = route_asset_root()
VARIANT = "route_zhanshi_fast"
SOURCE_ROOT = CLEAN_ROOT.parent / 'sources/route_zhanshi_fast'


def montage_structure(image: np.ndarray) -> np.ndarray:
    """Extract blue corridors / pink rooms, excluding route lines and captions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (92, 25, 65), (125, 190, 245))
    mask |= cv2.inRange(hsv, (0, 25, 65), (18, 190, 245))
    mask |= cv2.inRange(hsv, (170, 25, 65), (179, 190, 245))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # Stars, numbers and thick route lines can leave enclosed holes in a room.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - mask)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if x > 0 and y > 0 and x+w < mask.shape[1] and y+h < mask.shape[0] and area < 1400:
            mask[labels == label] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 80:
            mask[labels == label] = 0
    return mask


def alignment_quality(reference: np.ndarray, observed: np.ndarray) -> dict:
    ref, obs = reference > 0, observed > 0
    distance = cv2.distanceTransform((~obs).astype(np.uint8), cv2.DIST_L2, 3)
    return {
        "coverage": float(np.count_nonzero(ref & obs) / max(1, np.count_nonzero(ref))),
        "iou": float(np.count_nonzero(ref & obs) / max(1, np.count_nonzero(ref | obs))),
        "distance_p95_px": float(np.percentile(distance[ref], 95)),
    }


def quality_passes(quality: dict, diagnostics: dict) -> bool:
    seed_supported = diagnostics.get("feature_refinement", {}).get("accepted", False)
    if diagnostics.get("selected_seed") == "coarse":
        seed_supported = (diagnostics.get("score", 0) >= .95
                          and diagnostics.get("silhouette_ecc", 0) >= .90 and quality["iou"] >= .90)
    elif diagnostics.get("selected_seed") == "semantic":
        seed_supported = (diagnostics.get("semantic_template_score", 0) >= .85
                          and diagnostics.get("silhouette_ecc", 0) >= .90 and quality["iou"] >= .90)
    return (quality["coverage"] >= .90 and quality["iou"] >= .82
            and quality["distance_p95_px"] <= 4.0 and seed_supported)


def refine_silhouette(h: np.ndarray, reference: np.ndarray, structure: np.ndarray) -> tuple:
    height, width = reference.shape
    warped = cv2.warpPerspective(structure, np.linalg.inv(h), (width, height))
    details = {}
    scale = .5
    adjustment = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, adjustment = cv2.findTransformECC(
            cv2.resize(reference, None, fx=scale, fy=scale),
            cv2.resize(warped, None, fx=scale, fy=scale), adjustment, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6), None, 5,
        )
        scaled = np.diag([scale, scale, 1.0])
        candidate = h @ np.linalg.inv(scaled) @ np.vstack((adjustment, [0, 0, 1])) @ scaled
        after = cv2.warpPerspective(structure, np.linalg.inv(candidate), (width, height))
        if alignment_quality(reference, after)["iou"] > alignment_quality(reference, warped)["iou"]:
            h, warped = candidate, after
        details["silhouette_ecc"] = float(correlation)
    except cv2.error as exc:
        details["silhouette_ecc_error"] = str(exc)
    return h, warped, details


def semantic_seed(reference: np.ndarray, structure: np.ndarray) -> tuple:
    """Global silhouette correlation includes negative space between corridors."""
    downscale = .5
    small = cv2.resize(structure, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    maximum = min(small.shape[1] / reference.shape[1], small.shape[0] / reference.shape[0]) * .98
    best = (-1.0, 0.0, (0, 0))
    for scale in np.linspace(maximum * .15, maximum, 160):
        candidate = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if min(candidate.shape) < 20:
            continue
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(small, candidate, cv2.TM_CCOEFF_NORMED))
        if score > best[0]:
            best = (score, scale, location)
    score, scale, (x, y) = best
    h = np.array([[scale/downscale, 0, x/downscale], [0, scale/downscale, y/downscale], [0, 0, 1.]])
    return h, float(score)


def locate_floor(template: np.ndarray, route: np.ndarray, structure: np.ndarray) -> tuple:
    height, width = template.shape[:2]
    reference = np.uint8(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) > 48) * 255
    envelope = cv2.dilate(reference, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
    route_h, route_w = route.shape[:2]
    tile_w, tile_h = round(route_w * .66), round(route_h * .66)
    windows = [(0, 0, route_w, route_h)] + [
        (x, y, tile_w, tile_h) for y in (0, route_h-tile_h) for x in (0, route_w-tile_w)
    ]
    best = None
    attempts = []
    for window in windows:
        feature_h, base = estimate_template_to_route(template, route, window)
        coarse_h = np.array([[base["scale"], 0, base["translation"][0]],
                             [0, base["scale"], base["translation"][1]], [0, 0, 1]])
        # Repeated stair icons can make a feature fit override a correct global
        # template match. Retain both seeds and require whole-floor evidence.
        for label, seed in (("feature", feature_h), ("coarse", coarse_h)):
            h, warped, refinement = refine_silhouette(seed, reference, structure)
            diagnostics = {**base, **refinement, "selected_seed": label}
            warped[envelope == 0] = 0
            quality = alignment_quality(reference, warped)
            attempts.append({"window": list(window), "seed": label, "quality": quality})
            if best is None or quality["iou"] > best[2]["iou"]:
                best = (h, diagnostics, quality)
            if quality_passes(quality, diagnostics):
                diagnostics["search_attempts"] = attempts
                return h, diagnostics, quality
        if window == windows[0]:
            seed, score = semantic_seed(reference, structure)
            h, warped, refinement = refine_silhouette(seed, reference, structure)
            diagnostics = {**base, **refinement, "selected_seed": "semantic", "semantic_template_score": score}
            warped[envelope == 0] = 0
            quality = alignment_quality(reference, warped)
            attempts.append({"window": list(window), "seed": "semantic", "quality": quality})
            if quality["iou"] > best[2]["iou"]:
                best = (h, diagnostics, quality)
            if quality_passes(quality, diagnostics):
                diagnostics["search_attempts"] = attempts
                return h, diagnostics, quality
        # A small floor may lie below the whole-montage coarse scale range.
        # Overlapping 2D windows retry it without assuming any floor position.
    best[1]["search_attempts"] = attempts
    return best


def write_assets(payload: dict) -> None:
    """Isolate each floor by nearest structure, even where montage tiles touch."""
    source = ROOT / payload["route"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload["source_sha256"]:
        raise ValueError("Source image changed; recalibrate instead of rendering cached matrices")
    original = imread_unicode(source)
    templates = {f: imread_unicode(ROOT / item["template"]) for f, item in payload["floors"].items()}
    masks = {f: np.uint8(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) > 48) * 255 for f, t in templates.items()}
    transforms = {f: np.array(item["route_to_clean_homography"]) for f, item in payload["floors"].items()}
    size = tuple(payload["calibration_size"])
    structure = montage_structure(cv2.resize(original, size, interpolation=cv2.INTER_AREA))
    to_original = np.diag([original.shape[1] / size[0], original.shape[0] / size[1], 1.0])
    for floor, item in payload["floors"].items():
        if item["status"] != "PASS":
            continue
        height, width = templates[floor].shape[:2]
        own_distance = cv2.distanceTransform(255-masks[floor], cv2.DIST_L2, 3)
        # Keep numbers placed outside room walls as well as entrance labels.
        ownership = own_distance <= 120
        for other in FLOORS:
            if other == floor:
                continue
            other_mask = cv2.warpPerspective(masks[other], transforms[floor] @ np.linalg.inv(transforms[other]),
                                            (width, height), flags=cv2.INTER_NEAREST)
            other_distance = cv2.distanceTransform(255-other_mask, cv2.DIST_L2, 3)
            ownership &= own_distance < other_distance
        color = cv2.warpPerspective(original, transforms[floor], (width, height), flags=cv2.INTER_LANCZOS4)
        alpha = np.uint8(ownership & (color.max(axis=2) > 35)) * 255
        color[alpha == 0] = 0
        warped = cv2.warpPerspective(structure, transforms[floor] @ to_original, (width, height))
        warped[~ownership] = 0
        item["quality"] = alignment_quality(masks[floor], warped)
        if not quality_passes(item["quality"], item["diagnostics"]):
            item["status"] = "REVIEW"
            for key in ("asset", "structure_mask"):
                (ROOT / item[key]).unlink(missing_ok=True)
                item[key] = None
            continue
        imwrite_unicode(ROOT / item["asset"], np.dstack((color, alpha)))
        imwrite_unicode(ROOT / item["structure_mask"], warped)
        preview = cv2.addWeighted(templates[floor], .5, color, .5, 0)
        cv2.putText(preview, f"{payload['layout']} {floor} PASS IoU={item['quality']['iou']:.3f}",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 0), 2)
        imwrite_unicode(ROOT / item["alignment_preview"], preview)


def calibrate(layout: str, source: Path) -> dict:
    original = imread_unicode(source)
    size = (1536, round(original.shape[0] * 1536 / original.shape[1]))
    route = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    structure = montage_structure(route)
    report_dir = OUTPUT_ROOT / layout / VARIANT
    asset_dir = ASSET_ROOT / VARIANT / layout
    report_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "layout": layout, "layout_number": int(layout.split("_")[-1]),
        "variant_id": VARIANT, "display_name": "展十速刷路线图",
        "route": project_relative(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "route_size": [original.shape[1], original.shape[0]],
        "calibration_size": list(size), "floors": {},
    }
    for floor in FLOORS:
        started = time.perf_counter()
        template_path = floor_template(CLEAN_ROOT / layout, floor)
        template = imread_unicode(template_path)
        height, width = template.shape[:2]
        reference = np.uint8(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) > 48) * 255
        h, diagnostics, quality = locate_floor(template, route, structure)
        envelope = cv2.dilate(reference, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
        passed = quality_passes(quality, diagnostics)
        status = "PASS" if passed else "REVIEW"
        # Render at clean-template resolution from the original 4K artwork.
        source_scale = np.diag([size[0] / original.shape[1], size[1] / original.shape[0], 1.0])
        source_to_clean = np.linalg.inv(h) @ source_scale
        color = cv2.warpPerspective(original, source_to_clean, (width, height), flags=cv2.INTER_LANCZOS4)
        alpha = np.where((envelope > 0) & (color.max(axis=2) > 35), 255, 0).astype(np.uint8)
        color[alpha == 0] = 0
        asset_path = asset_dir / f"{floor}.png"
        mask_path = asset_dir / f"{floor}_structure.png"
        if not passed:
            asset_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
        preview_path = report_dir / f"{floor}_alignment.jpg"
        preview = cv2.addWeighted(template, .5, color, .5, 0)
        cv2.putText(preview, f"{layout} {floor} {status} IoU={quality['iou']:.3f}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 0) if passed else (0, 0, 255), 2)
        imwrite_unicode(preview_path, preview)
        payload["floors"][floor] = {
            "status": status, "template": project_relative(template_path),
            "asset": project_relative(asset_path) if passed else None,
            "structure_mask": project_relative(mask_path) if passed else None,
            "alignment_preview": project_relative(preview_path),
            "asset_to_clean_homography": np.eye(3).tolist(),
            "clean_to_route_homography": np.linalg.inv(source_to_clean).tolist(),
            "route_to_clean_homography": source_to_clean.tolist(),
            "calibration_ms": (time.perf_counter() - started) * 1000,
            "quality": quality, "diagnostics": diagnostics,
        }
        print(layout, floor, status, quality, flush=True)
    write_assets(payload)
    (asset_dir / "calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return payload


def main() -> None:
    global ASSET_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="明确将结果写入正式资产目录")
    parser.add_argument("--layouts", nargs="+", type=int, help="省略时读取当前底图目录中的全部布局")
    parser.add_argument("--render-only", action="store_true", help="Rebuild floor isolation from unchanged source and saved matrices")
    args = parser.parse_args()
    ASSET_ROOT = route_asset_root(publish=args.publish)
    if args.layouts is None:
        args.layouts = layout_numbers(CLEAN_ROOT)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260926)
    files = list(SOURCE_ROOT.glob("*.jpg"))
    failed = []
    for number in args.layouts:
        matches = [p for p in files if leading_number(p) == number]
        if len(matches) != 1:
            raise ValueError(f"Layout {number}: expected one montage, found {len(matches)}")
        if args.render_only:
            path = ASSET_ROOT / VARIANT / f"layout_{number:02d}" / "calibration.json"
            source_path = CLEAN_ROOT.parent / "routes" / VARIANT / f"layout_{number:02d}" / "calibration.json"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            for item in payload["floors"].values():
                for key in ("asset", "structure_mask"):
                    if item.get(key):
                        item[key] = project_relative(asset_output(ROOT / item[key], publish=args.publish))
            path.parent.mkdir(parents=True, exist_ok=True)
            write_assets(payload)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        else:
            payload = calibrate(f"layout_{number:02d}", matches[0])
        failed.extend(f"{payload['layout']}/{floor}" for floor, item in payload["floors"].items() if item["status"] != "PASS")
    if failed:
        raise SystemExit("Calibration requires review: " + ", ".join(failed))


if __name__ == "__main__":
    main()
