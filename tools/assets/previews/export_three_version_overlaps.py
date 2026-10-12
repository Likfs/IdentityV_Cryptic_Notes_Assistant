"""Export registered pair and triple overlaps, with an offline comparison gallery."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.app_catalog import FLOORS, FLOOR_NAMES, MapCatalog
from tools.common.layered_assets import load_route_overlay_asset, composite_route_layer, warp_route_overlay

PAIRS = [("zhanshi_clean", "展十速刷 + 原版干净底图"), ("zhanshi_lianghapi", "展十速刷 + 凉哈皮"),
         ("all_three", "底图 + 凉哈皮 + 展十速刷")]


def font(size):
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'workspace/reports/three_version_overlaps')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    catalog = MapCatalog()
    profile = catalog.profile("a_bringer_of_doom", "nightmare")
    rows = []
    sheets = []
    for floor in FLOORS:
        tiles = []
        for layout in catalog.layouts(profile):
            path = catalog.clean_template(profile, layout, floor)
            clean = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            rendered = {}
            variants = {v.variant_id: v for v in catalog.variants(profile, layout)}
            for version in ("route_lhp_solo", "route_zhanshi_fast"):
                asset = load_route_overlay_asset(variants[version].calibration_path, floor)
                layer, _ = warp_route_overlay(asset, np.eye(3), (clean.shape[1], clean.shape[0]), 1.)
                rendered[version], _ = composite_route_layer(clean, layer)
            liang, zhan = rendered["route_lhp_solo"], rendered["route_zhanshi_fast"]
            blends = [cv2.addWeighted(zhan, .5, clean, .5, 0), cv2.addWeighted(zhan, .5, liang, .5, 0),
                      np.round((clean.astype(np.float32)+liang+zhan)/3).astype(np.uint8)]
            files = []
            card = Image.new("RGB", (1260, 452), "#111b29")
            draw = ImageDraw.Draw(card)
            for index, ((slug, title), bgr) in enumerate(zip(PAIRS, blends)):
                directory = out / slug
                directory.mkdir(exist_ok=True)
                filename = directory / f"{layout}_{floor}.jpg"
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                image.save(filename, quality=96, subsampling=0)
                files.append(filename.relative_to(out).as_posix())
                thumb = ImageOps.contain(image, (402, 370), Image.Resampling.LANCZOS)
                card.paste(thumb, (index*420+(420-thumb.width)//2, 61+(370-thumb.height)//2))
                draw.text((index*420+16, 12), f"{layout} · {FLOOR_NAMES[floor]}", font=font(17), fill="#9caec4")
                draw.text((index*420+16, 35), title, font=font(19), fill="white")
            card_path = out / f"comparison_{layout}_{floor}.jpg"
            card.save(card_path, quality=94, subsampling=0)
            tiles.append(card)
            rows.append({"layout": layout, "floor": floor, "floor_name": FLOOR_NAMES[floor], "images": files,
                         "comparison": card_path.name})
        for start in range(0, len(tiles), 7):
            subset = tiles[start:start+7]
            sheet = Image.new("RGB", (1260, 452*len(subset)), "#111b29")
            for i, tile in enumerate(subset):
                sheet.paste(tile, (0, i*452))
            filename = f"overview_{floor}_{start//7+1}.jpg"
            sheet.save(out / filename, quality=92, subsampling=0)
            sheets.append(filename)
    data = {"coordinate_system": "clean template pixels", "pair_weights": [.5, .5],
            "triple_weights": [1/3, 1/3, 1/3], "pairs": [p[1] for p in PAIRS],
            "image_count": len(rows)*3, "items": rows, "overview_sheets": sheets}
    (out / "index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三套地图重叠对照</title>
<style>body{font:16px "Microsoft YaHei",sans-serif;margin:32px;background:#0c1420;color:#e5edf7}h1{font-size:28px}
p{color:#aebed1;line-height:1.8}select,a{font:inherit}select{background:#213248;color:white;padding:10px;border-radius:8px;margin:0 18px 20px 8px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}article{background:#172437;padding:16px;border-radius:12px}
img{width:100%;background:#000;object-fit:contain}a{color:#75d9ce}h2{font-size:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>三套地图 · 配准重叠对照</h1><p>原版干净底图 / 凉哈皮路线图 / 展十速刷路线图<br>
全部图片对齐到同一干净底图坐标。两套各占 50%；三套各占 1/3。点击图片查看原尺寸。路线不同不代表配准错位，请核对房间墙壁与走廊边缘。</p>
<label>布局<select id="layout"></select></label><label>楼层<select id="floor"><option value="1st">1 楼</option><option value="2nd">2 楼</option><option value="b1">地下室</option></select></label>
<div class="grid" id="images"></div><p>共 21 个布局 × 3 个楼层 × 3 种叠加方式 = 189 张原尺寸重叠图。</p>
<details><summary>分组总览（每页 7 个布局）</summary><ul id="sheets"></ul></details>
<script>const data=__DATA__;const layout=document.querySelector('#layout'),floor=document.querySelector('#floor');
[...new Set(data.items.map(x=>x.layout))].forEach(x=>{const o=new Option(x,x);layout.add(o)});
function show(){const row=data.items.find(x=>x.layout===layout.value&&x.floor===floor.value);
document.querySelector('#images').replaceChildren(...row.images.map((src,i)=>{const a=document.createElement('article');
const h=document.createElement('h2');h.textContent=data.pairs[i];const link=document.createElement('a');link.href=src;link.target='_blank';
const img=document.createElement('img');img.src=src;img.alt=h.textContent;link.append(img);a.append(h,link);return a}));}
layout.onchange=floor.onchange=show;show();data.overview_sheets.forEach(src=>{const li=document.createElement('li'),a=document.createElement('a');a.href=src;a.textContent=src;li.append(a);document.querySelector('#sheets').append(li)});
</script></html>'''.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"images": data["image_count"], "comparison_cards": len(rows), "overview_sheets": len(sheets), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
