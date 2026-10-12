import ast
import json
from pathlib import Path
from src.app_catalog import MapCatalog, load_settings
from src.paths import PROJECT_ROOT, SETTINGS_PATH


def test_formal_settings_do_not_read_tool_defaults(tmp_path, monkeypatch):
    read_text = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        reads.append(path)
        if not path.is_relative_to(tmp_path):
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


def test_active_code_has_no_imports_from_retired_engines():
    forbidden = ('torch', 'retired_features', 'src.layout_classifier', 'src.registration_core',
                 'src.registration.multiscale_structure', 'src.registration.runtime_registration')
    for folder in ('src', 'tools'):
        for path in (PROJECT_ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                assert not any(n == p or n.startswith(p + '.') for n in names for p in forbidden), path


def test_tests_do_not_import_other_test_modules():
    """Shared fixtures belong to conftest/support, so files work independently."""
    for path in (PROJECT_ROOT / 'tests').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            assert not any(part.startswith('test_') for module in modules for part in module.split('.')), path


def test_pure_modules_import_without_loading_the_gui():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(('PyQt6', 'src.route_projection_app', 'torch')), fullname
sys.meta_path.insert(0, NoGui())
from src import app_catalog, route_projection_core
from src.capture import mouse_pan
from src.registration.template_matching import screen
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
