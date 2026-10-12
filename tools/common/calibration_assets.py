"""Shared paths and layout helpers for offline calibration commands."""
from pathlib import Path
import re
from src.paths import PROJECT_ROOT
from src.app_catalog import FLOORS
from tools.common.paths import asset_output

CLEAN_ROOT = PROJECT_ROOT / 'assets/maps/a_bringer_of_doom/nightmare/clean'
OUTPUT_ROOT = PROJECT_ROOT / 'workspace/reports/route_asset_calibration'


def route_asset_root(*, publish=False):
    return asset_output(CLEAN_ROOT.parent / 'routes', publish=publish)


def layout_numbers(clean_root=CLEAN_ROOT):
    return sorted(int(p.name.removeprefix('layout_')) for p in clean_root.glob('layout_*')
                  if p.is_dir() and p.name.removeprefix('layout_').isdigit())


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def leading_number(path: Path) -> int | None:
    match = re.match(r"^(\d+)", path.name)
    return int(match.group(1)) if match else None


def floor_template(layout_dir: Path, floor: str) -> Path:
    matches = list(layout_dir.glob(f"*_{floor}.png"))
    if len(matches) != 1:
        raise RuntimeError(f"{layout_dir.name} 的 {floor} 干净底图数量不是1：{len(matches)}")
    return matches[0]
