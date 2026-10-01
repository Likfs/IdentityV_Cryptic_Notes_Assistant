"""Index the three display sets and register clean templates as identity assets."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.common.paths import asset_output
from src.app_catalog import FLOORS, MapCatalog


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="整理资产清单，默认生成到 workspace/asset_build")
    parser.add_argument("--publish", action="store_true", help="明确更新正式资产清单与标定描述")
    args = parser.parse_args()
    catalog = MapCatalog()
    inventory = {"schema_version": 1, "profiles": []}
    for profile in catalog.profiles.values():
        for layout in catalog.layouts(profile):
            directory = asset_output(profile.calibration_root / "route_clean" / layout, publish=args.publish)
            directory.mkdir(parents=True, exist_ok=True)
            payload = {"layout": layout, "variant_id": "route_clean", "display_name": "原版干净底图",
                       "asset_kind": "clean", "route": relative(profile.clean_root / layout), "floors": {}}
            for floor in FLOORS:
                template = relative(catalog.clean_template(profile, layout, floor))
                payload["floors"][floor] = {"status": "PASS", "template": template, "asset": template,
                    "asset_to_clean_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "diagnostics": {"method": "identity_reference", "independent_route_evidence": False}}
            (directory / "calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        sets = []
        for definition in profile.asset_versions:
            entries = []
            for layout in catalog.layouts(profile):
                variant = next(v for v in catalog.variants(profile, layout) if v.variant_id == definition["id"])
                payload = json.loads(variant.calibration_path.read_text(encoding="utf-8"))
                payload.update(display_name=definition["name"], asset_kind=definition["kind"],
                               source_collection=definition["source_root"])
                destination = asset_output(variant.calibration_path, publish=args.publish)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                entries.append({"layout": layout, "source": payload["route"],
                                "calibration": relative(variant.calibration_path), "floors": {
                                    f: {k: item[k] for k in ("status", "asset", "structure_mask") if k in item}
                                    for f, item in payload["floors"].items()}})
            sets.append({**definition, "layout_count": len(entries), "floor_count": len(entries)*3, "entries": entries})
        inventory["profiles"].append({"profile": profile.key, "versions": sets})
    output = asset_output(ROOT / "assets/maps/inventory.json", publish=args.publish)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Indexed {sum(v['floor_count'] for p in inventory['profiles'] for v in p['versions'])} floor assets: {output}")


if __name__ == "__main__":
    main()
