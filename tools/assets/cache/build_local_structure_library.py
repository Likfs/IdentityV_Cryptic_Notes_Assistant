"""Build the local structure index for every available clean map floor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.registration.runtime_registration import prepare_registration_template
from src.registration_core.local_structure_index import INDEX_VERSION
from src.registration_core.structural_features import FEATURE_NAMES
from tools.assets.image_io import write_image


ROOT = Path(__file__).resolve().parents[3]
CLEAN_ROOT = ROOT / "assets/maps/a_bringer_of_doom/nightmare/clean"
CACHE_ROOT = ROOT / 'workspace/cache/route_projection_templates/a_bringer_of_doom/nightmare'
REPORT = ROOT / 'workspace/reports/route_projection/local_structure_library/build_report.json'
PREVIEW_ROOT = REPORT.parent / "feature_previews"


def _feature_preview(index) -> np.ndarray:
    corridor = index.channels[:, :, 0] > 0.5
    room = index.channels[:, :, 1] > 0.5
    output = np.zeros((*corridor.shape, 3), dtype=np.uint8)
    output[corridor] = (100, 100, 100)
    output[room] = (110, 150, 200)
    features = index.feature_channels > 0.1
    output[features[:, :, 0]] = (230, 190, 80)   # horizontal walls
    output[features[:, :, 1]] = (230, 100, 80)   # vertical walls
    output[np.any(features[:, :, 2:6], axis=2)] = (80, 220, 100)  # oriented corners
    output[np.any(features[:, :, 6:8], axis=2)] = (40, 240, 240)  # room contacts
    output[features[:, :, 8]] = (210, 80, 210)  # turns
    output[features[:, :, 9]] = (190, 60, 240)  # T junctions
    output[features[:, :, 10]] = (70, 70, 255)  # crosses
    return cv2.resize(output, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=CLEAN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    clean_root, cache_root, report = args.clean_root, args.cache_root, args.output
    preview_root = report.parent / "feature_previews"
    paths = [path for path in sorted(clean_root.glob("layout_*/*.png"))
             if path.stem.removeprefix(path.parent.name + "_") in {"b1", "1st", "2nd"}]
    if not paths:
        parser.error("没有可用底图")
    cv2.setNumThreads(4)
    items = []
    started = perf_counter()
    preview_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        layout = path.parent.name
        floor = path.stem.removeprefix(layout + "_")
        if floor not in {"b1", "1st", "2nd"}:
            continue
        prepared = prepare_registration_template(layout, floor, path, cache_root)
        index = prepared.local_structure_index
        preview_path = preview_root / f"{layout}_{floor}.png"
        write_image(preview_path, _feature_preview(index))
        items.append({
            "layout": layout,
            "floor": floor,
            "source": str(path.resolve()),
            "source_sha256": index.source_sha256,
            "index": str(index.cache_path.resolve()),
            "index_shape": list(index.channels.shape),
            "index_build_or_load_ms": round(prepared.local_structure_index_ms, 2),
            "feature_counts": {
                name: int(np.count_nonzero(index.feature_channels[:, :, channel] > 0.1))
                for channel, name in enumerate(FEATURE_NAMES)
            },
            "room_connections": [list(row) for row in index.room_connections],
            "preview": str(preview_path.resolve()),
        })
        print(f"{layout} {floor}: {index.channels.shape[1]}x{index.channels.shape[0]}")
    output = {
        "report_version": 2,
        "index_version": INDEX_VERSION,
        "item_count": len(items),
        "layout_count": len({item["layout"] for item in items}),
        "elapsed_ms": round((perf_counter() - started) * 1000.0, 2),
        "items": items,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} indexes to {report}")


if __name__ == "__main__":
    main()
