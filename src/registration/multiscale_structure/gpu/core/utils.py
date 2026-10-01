from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[5]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式无效: {path}")
    return data


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    with path.open("wb") as f:
        f.write(encoded.tobytes())
    return True


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if is_dataclass(data):
        data = asdict(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def crop_roi(image: np.ndarray, roi: list[int] | tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return image.copy()
    x, y, w, h = [int(v) for v in roi]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError(f"ROI 越界或非法: {roi}, image={image.shape[:2]}")
    return image[y : y + h, x : x + w].copy()


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def draw_label(image: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
