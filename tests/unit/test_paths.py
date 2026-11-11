"""External data stays beside the EXE regardless of the working directory."""
from pathlib import Path
import runpy
import sys

import pytest

from src import paths


@pytest.mark.parametrize('frozen', [False, True])
def test_application_paths_ignore_working_directory(tmp_path, monkeypatch, frozen):
    executable = tmp_path/'portable/IdentityVRouteAssistant.exe'
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    monkeypatch.setattr(sys, 'executable', str(executable))
    monkeypatch.setattr(sys, '_MEIPASS', str(executable.parent/'_internal'), raising=False)
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(Path(paths.__file__).resolve()))
    root = executable.parent if frozen else Path(paths.__file__).resolve().parents[1]
    assert result['PROJECT_ROOT'] == root
    assert result['CONFIG_ROOT'] == root/'configs'
    assert result['RUNTIME_ROOT'] == root/'runtime'
    assert result['SETTINGS_PATH'] == root/'runtime/settings.json'
    assert result['CACHE_ROOT'] == root/'runtime/cache'
    assert result['LOG_ROOT'] == root/'runtime/logs'
    assert result['DIAGNOSTIC_ROOT'] == root/'runtime/diagnostics'


def test_frozen_manifest_resolves_only_code_metadata_internally(tmp_path, monkeypatch):
    root = tmp_path/'portable'
    internal = root/'_internal'
    monkeypatch.setattr(paths, 'PROJECT_ROOT', root)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(internal), raising=False)
    assert paths.published_resource_path(root, paths.TEMPLATE_PROVENANCE_PATH) == internal/paths.TEMPLATE_PROVENANCE_PATH
    for name in ('assets/maps/example.png', 'configs/runtime_resources.json', 'runtime/settings.json'):
        assert paths.published_resource_path(root, name) == root/name
    other = tmp_path/'offline'
    assert paths.published_resource_path(other, paths.TEMPLATE_PROVENANCE_PATH) == other/paths.TEMPLATE_PROVENANCE_PATH
    with pytest.raises(ValueError):
        paths.published_resource_path(root, '../outside.json')
