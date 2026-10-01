"""Tests may read fixed inputs but must not write the user's runtime state."""
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_outputs(tmp_path, monkeypatch):
    from src import projection_workflow, route_projection_app, map_presence
    # Unit/Qt tests must not register a real process-wide keyboard listener.
    # Event integration tests explicitly install their own queued source.
    monkeypatch.setattr(route_projection_app, 'GlobalMapKeys', lambda *_args: None)
    for module in (projection_workflow, route_projection_app):
        monkeypatch.setattr(module, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(projection_workflow, "RUN_ROOT", tmp_path / "diagnostics")
    monkeypatch.setattr(route_projection_app, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(map_presence, "MAP_MARKER_REPORT_DIR", tmp_path / "map_presence")
