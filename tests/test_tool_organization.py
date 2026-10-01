"""Tool restructuring preserves settings and the production workflow boundary."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.common import capture_config as config
from tools.common.paths import PROJECT_ROOT
from tools.demos import projection_service as demo
from tools.assets.previews.project_route_asset import registration_matrix


@pytest.fixture
def capture_paths(tmp_path, monkeypatch):
    defaults = tmp_path / "tools/capture_defaults.json"
    defaults.parent.mkdir()
    defaults.write_bytes(config.CAPTURE_DEFAULTS_PATH.read_bytes())
    personal = tmp_path / "state/capture.json"
    legacy = defaults.with_name("capture.json")
    monkeypatch.setattr(config, "CAPTURE_DEFAULTS_PATH", defaults)
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", personal)
    monkeypatch.setattr(config, "LEGACY_CONFIG_PATH", legacy)
    return defaults, personal, legacy


def test_capture_legacy_fallback_and_personal_precedence(capture_paths):
    defaults, personal, legacy = capture_paths
    legacy.write_text(json.dumps({"capture_roi": [20, 30, 400, 500], "extra": "preserved"}))
    assert config.load_capture_config(personal)["capture_roi"] == [20, 30, 400, 500]
    assert not personal.exists()  # A read/check does not write state.
    personal.parent.mkdir()
    personal.write_text(json.dumps({"capture_roi": [-80, 50, 600, 700]}))
    assert config.load_capture_config(personal)["capture_roi"] == [-80, 50, 600, 700]
    legacy.unlink()
    assert config.load_capture_config(legacy)["capture_roi"] == [-80, 50, 600, 700]


def test_capture_save_redirects_defaults_and_creates_state(capture_paths):
    defaults, personal, legacy = capture_paths
    before = defaults.read_bytes()
    loaded = config.load_capture_config(personal)
    loaded["capture_roi"] = [1, 2, 30, 40]
    config.save_capture_config(loaded, defaults)
    assert defaults.read_bytes() == before
    assert config.load_capture_config(personal)["capture_roi"] == [1, 2, 30, 40]
    assert config.capture_settings_path(legacy) == personal
    assert not list(personal.parent.glob("*.tmp"))


def test_explicit_missing_capture_configuration_is_an_error(capture_paths):
    with pytest.raises(FileNotFoundError):
        config.load_capture_config(capture_paths[0].parent / "typo.json")


def test_dataset_tools_import_without_capture_or_qt():
    code = """
import sys
sys.modules['mss'] = None
sys.modules['PyQt6'] = None
from tools.common.sample_store import SampleWriter
import tools.training.delete_sample
import tools.training.rebuild_layout_jsonl
assert 'tools.collector.fixed_roi_collector' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["cpu", "auto"])
@pytest.mark.parametrize("accepted", [False, True])
def test_demo_uses_production_acceptance_backend_and_isolated_outputs(tmp_path, monkeypatch, backend, accepted):
    captured = {}
    registration = SimpleNamespace(clean_to_screen=np.array([[1., 0., 10.], [0., 1., 20.]]),
                                   validation={"registration_backend": {"effective_backend": "cpu", "fallback_reason": "test"}})
    result = SimpleNamespace(accepted=accepted, registration=registration, variant_id="route_lhp_solo",
                             timings_ms={"projection": 12.}, diagnostics=lambda: {"accepted": accepted})

    def process(screenshot, **kwargs):
        captured.update(kwargs)
        key = (kwargs["profile"].key, "layout_05", "route_lhp_solo", "2nd")
        kwargs["asset_cache"][key] = SimpleNamespace(asset_to_clean=np.eye(3))
        return result

    def save(value, directory):
        directory.mkdir(parents=True)
        return directory

    monkeypatch.setattr(demo, "process_screenshot", process)
    monkeypatch.setattr(demo, "save_workflow_result", save)
    service = demo.DemoProjectionService(backend, tmp_path / "cache")
    actual, payload = service.project(np.zeros((20, 20, 3), np.uint8), "layout_05", "2nd", .96, tmp_path / "run")
    assert actual is result and payload["accepted"] is accepted
    assert captured["registration_backend"] == backend
    assert captured["model_bundle"] is None and captured["layout_override"] == "layout_05"
    assert not {"registration_priors", "registration_evidence", "registration_soft_priors"} & captured.keys()
    assert captured["template_cache_root"] == tmp_path / "cache" / backend
    assert payload["backend_info"]["fallback_reason"] == "test"
    assert (payload["asset_to_screen_homography"] is not None) is accepted
    assert json.loads((tmp_path / "run/result.json").read_text())["accepted"] is accepted


def test_tools_imports_resolve_after_asset_moves():
    for path in (PROJECT_ROOT / "tools").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tools."):
                target = PROJECT_ROOT.joinpath(*node.module.split("."))
                assert target.is_dir() or target.with_suffix(".py").is_file(), (path, node.module)


def test_asset_preview_reads_current_report_and_rejects_wrong_space():
    matrix = [[1., 0., 3.], [0., 1., 4.]]
    value, name = registration_matrix({"registration": {"clean_to_screen": matrix, "validation": {}}})
    assert name == "registration.clean_to_screen"
    assert np.array_equal(value, [[1., 0., 3.], [0., 1., 4.], [0., 0., 1.]])
    with pytest.raises(ValueError, match="完整原图坐标"):
        registration_matrix({"registration": {"clean_to_screen": matrix, "validation": {"coordinate_space": "original_source_image"}}})
    with pytest.raises(ValueError, match="矩阵无效"):
        registration_matrix({"clean_to_screen": [[0., 0., 0.], [0., 0., 0.]]})
