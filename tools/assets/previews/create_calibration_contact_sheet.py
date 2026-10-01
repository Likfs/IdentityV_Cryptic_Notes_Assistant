from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from tools.assets.image_io import read_image

SUMMARY = PROJECT_ROOT / 'workspace/reports/route_asset_calibration/summary.json'
OUTPUT = PROJECT_ROOT / 'workspace/reports/route_asset_calibration/alignment_contact.png'




def main() -> None:
    parser = argparse.ArgumentParser(description="从标定报告生成预览拼图")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    tiles = []
    for variant in summary["variants"]:
        for floor, item in variant["floors"].items():
            preview_path = Path(item["alignment_preview"])
            if not preview_path.is_absolute():
                preview_path = PROJECT_ROOT / preview_path
            image = read_image(preview_path)
            scale = min(320 / image.shape[1], 380 / image.shape[0])
            resized = cv2.resize(
                image,
                (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            tile = np.full((420, 340, 3), 25, np.uint8)
            x = (tile.shape[1] - resized.shape[1]) // 2
            y = 35 + (380 - resized.shape[0]) // 2
            tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            label = f"{variant['layout']} {floor} {item['status']}"
            cv2.putText(tile, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)

    if not tiles:
        parser.error("标定报告没有预览图片")
    columns = 6
    rows = []
    for index in range(0, len(tiles), columns):
        row = tiles[index : index + columns]
        row.extend(np.full_like(tiles[0], 25) for _ in range(columns - len(row)))
        rows.append(np.hstack(row))
    contact = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", contact)[1].tofile(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
