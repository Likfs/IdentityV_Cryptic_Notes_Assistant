from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.assets.image_io import read_image, write_image
from tools.common.layered_assets import load_route_overlay_asset, project_route_overlay  # noqa: E402


CLEAN_ROOT = PROJECT_ROOT / "assets/maps/a_bringer_of_doom/nightmare/clean"
CALIBRATION_ROOT = PROJECT_ROOT / "maintenance/maps/a_bringer_of_doom/nightmare/routes"
FLOORS = ("b1", "1st", "2nd")


def make_contact_sheet(files: list[Path], destination: Path, floor: str) -> None:
    columns = 4
    tile_width, tile_height = 420, 370
    rows = math.ceil(len(files) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#e7ecf2")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(files):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((tile_width - 16, tile_height - 42), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y + 6))
        draw.text((x + 9, y + tile_height - 28), f"{path.stem}  [{floor}]", fill="#111827")
    canvas.save(destination, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="将全部路线图按标定矩阵叠加到干净底图，集中输出预览")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / 'workspace/reports/route_asset_calibration/registration_gallery',
    )
    parser.add_argument("--opacity", type=float, default=0.72)
    args = parser.parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity 必须在 0 到 1 之间")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[str] = []
    floor_files: dict[str, list[Path]] = {floor: [] for floor in FLOORS}
    for layout_dir in sorted(CLEAN_ROOT.glob("layout_*")):
        if not layout_dir.is_dir():
            continue
        layout = layout_dir.name
        calibration_files = sorted(CALIBRATION_ROOT.glob(f"route_*/{layout}/calibration.json"))
        if not calibration_files:
            errors.append(f"{layout}：没有路线标定文件")
            continue
        for calibration_file in calibration_files:
            payload = json.loads(calibration_file.read_text(encoding="utf-8"))
            route_path = Path(payload["route"])
            if not route_path.is_absolute():
                route_path = PROJECT_ROOT / route_path
            for floor in FLOORS:
                try:
                    clean_files = list(layout_dir.glob(f"*_{floor}.png"))
                    if len(clean_files) != 1:
                        raise RuntimeError(f"干净底图数量为 {len(clean_files)}，预期为 1")
                    clean = read_image(clean_files[0])
                    item = payload["floors"][floor]
                    asset = load_route_overlay_asset(calibration_file, floor)
                    preview, _alpha, _matrix = project_route_overlay(
                        clean, asset, np.eye(3, dtype=np.float64), opacity=args.opacity
                    )
                    filename = f"{layout}_{payload['variant_id']}_{floor}.png"
                    preview_path = destination / filename
                    write_image(preview_path, preview)
                    floor_files[floor].append(preview_path)
                    refinement = item["diagnostics"].get("feature_refinement", {})
                    rows.append(
                        {
                            "layout": layout,
                            "variant": payload["variant_id"],
                            "floor": floor,
                            "status": item["status"],
                            "inliers": refinement.get("inliers", 0),
                            "median_inlier_error_px": refinement.get("median_inlier_error_px"),
                            "clean": str(clean_files[0].resolve()),
                            "route": str(route_path.resolve()),
                            "preview": str(preview_path),
                        }
                    )
                except Exception as exc:
                    errors.append(f"{layout} / {payload['variant_id']} / {floor}：{exc}")

    for floor, files in floor_files.items():
        if files:
            make_contact_sheet(files, destination / f"00_overview_{floor}.jpg", floor)
    summary = {"output": str(destination), "count": len(rows), "errors": errors, "items": rows}
    (destination / "index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 全部干净底图与路线图配准预览",
        "",
        f"- 完成图片：{len(rows)} 张",
        f"- 错误：{len(errors)} 项",
        f"- 路线图透明度：{args.opacity:.0%}",
        "- `PASS` 表示自动特征标定通过；仍需看图确认局部房间与走廊是否重合。",
        "- 单张 PNG 是路线图投到对应干净底图的半透明叠加；`00_overview_*.jpg` 供快速浏览。",
        "",
        "| 布局 | 路线版本 | 楼层 | 自动状态 | 内点 | 中位误差(px) | 配准图 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        median = row["median_inlier_error_px"]
        median_text = f"{median:.2f}" if median is not None else "—"
        name = Path(row["preview"]).name
        lines.append(
            f"| {row['layout']} | {row['variant']} | {row['floor']} | {row['status']} | "
            f"{row['inliers']} | {median_text} | [{name}]({name}) |"
        )
    if errors:
        lines.extend(["", "## 未生成", "", *(f"- {error}" for error in errors)])
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "count": len(rows), "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
