from src.paths import PROJECT_ROOT
from tools.common.paths import ASSET_BUILD_ROOT, asset_output


def test_asset_generation_stages_until_explicit_publication():
    target = PROJECT_ROOT / "assets/maps/example/calibration.json"
    assert asset_output(target) == ASSET_BUILD_ROOT / "assets/maps/example/calibration.json"
    assert asset_output(target, publish=True) == target


def test_current_tool_imports_resolve_after_directory_moves():
    """Recovered from the archived mixed tool/collector/Demo test module."""
    import ast
    import importlib.util
    for path in (PROJECT_ROOT / 'tools').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('tools.'):
                assert importlib.util.find_spec(node.module) is not None, (path, node.module)
