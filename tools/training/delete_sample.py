"""Permanently delete a sample and update its layout index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.common.sample_store import SampleWriter


def main() -> int:
    parser = argparse.ArgumentParser(description="永久删除样本图片并同步重建 JSONL，不保留副本")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--sample-id", help="图片文件名中的 sample_id")
    target.add_argument("--image", type=Path, help="样本目录内的 PNG 路径")
    args = parser.parse_args()
    sample_id = str(args.sample_id or args.image.stem).strip()
    result = SampleWriter(PROJECT_ROOT).delete_sample(sample_id, image_path=args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
