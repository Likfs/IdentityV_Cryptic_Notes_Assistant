from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.collector.fixed_roi_collector import (  # noqa: E402
    CaptureRegion,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LABELS_PATH,
    load_capture_config,
    load_category_config,
    run_app,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="固定屏幕区域样本采集器")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="采集配置 JSON 路径")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="地图、难度与布局分类配置 JSON 路径")
    parser.add_argument("--check", action="store_true", help="只检查配置，不打开窗口或截图")
    args = parser.parse_args()
    config = load_capture_config(args.config)
    categories = load_category_config(args.labels)
    if args.check:
        region = CaptureRegion.from_value(config["capture_roi"])
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": str(args.config.resolve()),
                    "labels": str(args.labels.resolve()),
                    "capture_roi": region.as_list(),
                    "image_size": [region.width, region.height],
                    "default_interval_seconds": config["default_interval_seconds"],
                    "voice_prompts_enabled": config["voice_prompts_enabled"],
                    "maps": [{"id": item.id, "name": item.name} for item in categories["maps"]],
                    "difficulties": [
                        {"id": item.id, "name": item.name} for item in categories["difficulties"]
                    ],
                    "default_map_id": categories["default_map_id"],
                    "default_difficulty_id": categories["default_difficulty_id"],
                    "layout_numbering": "dynamic: layout_01, layout_02, ...",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    run_app(args.config, args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
