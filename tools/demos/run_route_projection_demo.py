from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.region import CaptureRegion
from tools.common.capture_config import load_capture_config
from tools.demos.projection_service import DemoProjectionService
from src.registration.multiscale_structure.runtime import initialize  # noqa: E402
from src.layout_classifier import load_classifier  # noqa: E402
from tools.demos.layout_recognition_demo import DEFAULT_MODEL_PATH, PRE_RESIZED_MODEL_PATH, model_coverage_text  # noqa: E402
from src.projection.route_overlay import load_route_overlay_asset  # noqa: E402
from tools.common.tk_overlay import RouteOverlayWindow  # noqa: E402
from tools.demos.route_projection_demo import (  # noqa: E402
    CALIBRATION_ROOT,
    CAPTURE_CONFIG,
    CLEAN_ROOT,
    DEMO_CONFIG,
    RUN_ROOT,
    TEMPLATE_CACHE,
    available_asset_layouts,
    available_asset_floors,
    load_demo_config,
    run_demo,
)


def read_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def project_once(screenshot_path: Path, layout: str, floor: str, opacity: float, output: Path, backend="auto") -> dict:
    screenshot = read_color(screenshot_path.resolve())
    _result, payload = DemoProjectionService(backend, TEMPLATE_CACHE).project(
        screenshot, layout, floor, opacity, output.resolve())
    return dict(payload, screenshot=str(screenshot_path.resolve()), output=str(output.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description="布局识别与路线投影 Demo")
    parser.add_argument("--config", type=Path, default=CAPTURE_CONFIG)
    parser.add_argument("--demo-config", type=Path, default=DEMO_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("cpu", "auto"), help="当前正式配准：cpu 或 GPU 优先 auto；默认读取 Demo 设置")
    parser.add_argument("--check", action="store_true", help="检查模型、快捷键配置和路线资产，不打开窗口")
    parser.add_argument("--project-once", type=Path, metavar="SCREENSHOT", help="离线投影一张截图，用于排查")
    parser.add_argument("--layout", default="layout_05")
    parser.add_argument("--floor", choices=("2nd", "1st", "b1"), default="2nd")
    parser.add_argument("--opacity", type=float, default=0.96)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "offline_check")
    args = parser.parse_args()
    default_model = PRE_RESIZED_MODEL_PATH if PRE_RESIZED_MODEL_PATH.is_file() else DEFAULT_MODEL_PATH
    model_path = (args.model or default_model).resolve()

    demo_config = load_demo_config(args.demo_config.resolve())
    backend = args.backend or demo_config["registration_backend"]
    if args.project_once:
        result = project_once(
            args.project_once,
            args.layout,
            args.floor,
            float(np.clip(args.opacity, 0.10, 1.0)),
            args.output.resolve(),
            backend,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["accepted"] else 2

    if args.check:
        capture = load_capture_config(args.config.resolve())
        region = CaptureRegion.from_value(capture["capture_roi"])
        _model, checkpoint, device = load_classifier(model_path)
        demo_config = load_demo_config(args.demo_config.resolve())
        layouts = available_asset_layouts()
        available_floors, review_floors = available_asset_floors()
        loaded_assets = [
            load_route_overlay_asset(CALIBRATION_ROOT / "route_lhp_solo" / layout / "calibration.json", floor)
            for layout, floor in available_floors
        ]
        payload = {
            "ok": True,
            "model": str(model_path),
            "model_labels": checkpoint["labels"],
            "model_coverage": model_coverage_text(checkpoint),
            "device": str(device),
            "registration_backend": backend,
            "registration_backend_info": initialize(backend),
            "registration_algorithm": "multiscale_structure",
            "capture_roi": region.as_list(),
            "hotkeys": {
                "recognize": demo_config["recognize_hotkey"],
                "project": demo_config["project_hotkey"],
            },
            "default_floor": demo_config["floor"],
            "opacity": demo_config["opacity"],
            "asset_layouts": layouts,
            "asset_layout_count": len(layouts),
            "asset_floor_count": len(loaded_assets),
            "review_floors": [{"layout": layout, "floor": floor} for layout, floor in review_floors],
            "all_asset_alpha_nonempty": all(int(asset.rgba[:, :, 3].max()) > 0 for asset in loaded_assets),
            "per_pixel_overlay_supported": RouteOverlayWindow.supported(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    run_demo(model_path, args.config, args.demo_config, backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
