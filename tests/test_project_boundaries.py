"""Formal settings and tool output boundaries survive directory migration."""
import ast
import json
from pathlib import Path

from src.app_catalog import MapCatalog, load_settings
from src.paths import PROJECT_ROOT, SETTINGS_PATH
from tools.common.paths import ASSET_BUILD_ROOT, asset_output


def test_formal_settings_do_not_read_tool_defaults(tmp_path, monkeypatch):
    read_text = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        reads.append(path)
        assert "tools" not in path.parts and "workspace" not in path.parts
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    saved = tmp_path / "settings.json"
    saved.write_text(json.dumps({"map_capture_delay_ms": 135, "opacity": .43}), encoding="utf-8")
    settings = load_settings(MapCatalog(), saved)
    assert settings["map_capture_delay_ms"] == 135 and settings["opacity"] == .43
    assert settings["registration_backend"] == "template" and settings["recognize_hotkey"] == "F8"
    assert any(p.name == "app_defaults.json" for p in reads)
    assert SETTINGS_PATH == PROJECT_ROOT / "runtime/settings.json"


def test_formal_modules_do_not_import_developer_tools():
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            assert not any(m.split(".")[0] in {"tools", "scripts", "tests", "tkinter"} for m in modules), path


def test_asset_generation_stages_until_explicit_publication():
    target = PROJECT_ROOT / "assets/maps/example/calibration.json"
    assert asset_output(target) == ASSET_BUILD_ROOT / "assets/maps/example/calibration.json"
    assert asset_output(target, publish=True) == target
