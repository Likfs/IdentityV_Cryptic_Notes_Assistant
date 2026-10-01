"""Capture defaults and personal settings shared by the collector and Demos."""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

from src.capture.region import CaptureRegion
from tools.common.paths import TOOL_CONFIG_ROOT, TOOL_STATE_ROOT

CAPTURE_DEFAULTS_PATH = TOOL_CONFIG_ROOT / "capture_defaults.json"
LEGACY_CONFIG_PATH = TOOL_CONFIG_ROOT / "capture.json"
DEFAULT_CONFIG_PATH = TOOL_STATE_ROOT / "capture.json"
MIN_INTERVAL_SECONDS = 0.1


def capture_settings_path(path: Path) -> Path:
    """Saving bundled defaults or the old default must target personal settings."""
    path = Path(path)
    if path.resolve() in (LEGACY_CONFIG_PATH.resolve(), CAPTURE_DEFAULTS_PATH.resolve()):
        return DEFAULT_CONFIG_PATH
    return path


def load_capture_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path)
    defaults = json.loads(CAPTURE_DEFAULTS_PATH.read_text(encoding="utf-8-sig"))
    if path.resolve() == LEGACY_CONFIG_PATH.resolve() and not path.exists():
        path = DEFAULT_CONFIG_PATH
    if path.resolve() == DEFAULT_CONFIG_PATH.resolve() and not path.exists():
        path = LEGACY_CONFIG_PATH if LEGACY_CONFIG_PATH.exists() else CAPTURE_DEFAULTS_PATH
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("采集配置的根节点必须是对象")
    data = {**defaults, **loaded}
    data["capture_roi"] = CaptureRegion.from_value(data.get("capture_roi")).as_list()
    interval = float(data.get("default_interval_seconds", 1.2))
    if not math.isfinite(interval) or interval < MIN_INTERVAL_SECONDS:
        raise ValueError(f"默认采样间隔必须为有限数且不能小于 {MIN_INTERVAL_SECONDS:.1f} 秒")
    data["default_interval_seconds"] = interval
    for name in ("skip_exact_duplicates", "voice_prompts_enabled", "show_capture_border", "minimize_on_start"):
        data[name] = bool(data.get(name, False))
    return data


def save_capture_config(config: dict, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path = capture_settings_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
