from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.region import CaptureRegion
from tools.common.capture_config import load_capture_config
from src.layout_classifier import load_classifier
from tools.demos.layout_recognition_demo import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    PRE_RESIZED_MODEL_PATH,
    model_coverage_text,
    run_demo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="当前屏幕布局识别可视化 Demo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=Path, help="自定义模型文件；也可在 Demo 窗口中选择")
    model_group.add_argument(
        "--variant",
        choices=("original", "pre_resized"),
        default="original",
        help="启动时选择原图增强或预缩放模型；窗口中仍可切换",
    )
    parser.add_argument("--check", action="store_true", help="只检查模型和配置，不打开窗口")
    args = parser.parse_args()
    model_path = args.model or (
        PRE_RESIZED_MODEL_PATH if args.variant == "pre_resized" else DEFAULT_MODEL_PATH
    )
    if args.check:
        config = load_capture_config(args.config)
        region = CaptureRegion.from_value(config["capture_roi"])
        _model, checkpoint, device = load_classifier(model_path.resolve())
        print(
            json.dumps(
                {
                    "ok": True,
                    "model": str(model_path.resolve()),
                    "labels": checkpoint["labels"],
                    "model_coverage": model_coverage_text(checkpoint),
                    "capture_roi": region.as_list(),
                    "device": str(device),
                    "global_hotkey": "F9",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    run_demo(args.config, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
