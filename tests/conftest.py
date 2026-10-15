"""Shared isolation without importing the application or creating a Qt instance."""
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption('--run-release', action='store_true', help='Run external environment / EXE acceptance')
    parser.addoption('--runtime-python', help='Python in an isolated environment containing requirements.txt only')
    parser.addoption('--release-exe', help='Built IdentityVRouteAssistant.exe to validate')


def pytest_collection_modifyitems(config, items):
    selected, excluded = [], []
    for item in items:
        parts = item.path.relative_to(ROOT / 'tests').parts
        category = parts[0]
        item.add_marker(getattr(pytest.mark, category))
        if category == 'integration' and 'gui' in parts:
            item.add_marker(pytest.mark.gui)
        if category == 'release' and not config.getoption('--run-release'):
            excluded.append(item)
        else:
            selected.append(item)
    if excluded:
        config.hook.pytest_deselected(items=excluded)
        items[:] = selected


@pytest.fixture(autouse=True)
def isolated_runtime_outputs(tmp_path, monkeypatch):
    # Patch already imported modules only. Pure tests never import the Qt app.
    for name in ('src.paths', 'src.layout_retrieval', 'src.projection_workflow',
                 'src.route_projection_app', 'src.__main__'):
        module = sys.modules.get(name)
        if module is None:
            continue
        for attribute, directory in (('CACHE_ROOT', 'cache'), ('LOG_ROOT', 'logs'),
                                     ('RUN_ROOT', 'diagnostics'), ('DIAGNOSTIC_ROOT', 'diagnostics')):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, tmp_path / directory)
    # Shared matcher settings and caches must not leak between test files.
    for name, attributes in {
        'src.registration.template_matching.direct': ('_images',),
        'src.registration.template_matching.floors': ('_images', '_config_cache'),
        'src.registration.template_matching.matcher': ('_map_cache',),
        'src.registration.template_matching.resources': ('verified_fingerprints',),
    }.items():
        module = sys.modules.get(name)
        if module:
            for attribute in attributes:
                monkeypatch.setattr(module, attribute, type(getattr(module, attribute))())
    matcher = sys.modules.get('src.registration.template_matching.matcher')
    if matcher:
        monkeypatch.setattr(matcher, 'TEMPLATE_DS', matcher.TEMPLATE_DS)
