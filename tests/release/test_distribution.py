"""Explicit acceptance: requires supplied paths, never silently skips missing dependencies."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.support.subprocess_checks import ROOT, run_probe, check_projection


@pytest.mark.parametrize('mode', ['check', 'recognize', 'project', 'gui'])
def test_runtime_only_environment(request, tmp_path, mode):
    option = request.config.getoption('--runtime-python')
    assert option, 'Use --runtime-python with a separate environment containing runtime dependencies only'
    python = Path(option).resolve()
    assert python.is_file(), python
    code = '''
import importlib.util, json, sys
assert sys.prefix != sys.base_prefix, 'A separate virtual environment is required'
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('pytest') is None
assert importlib.util.find_spec('PyInstaller') is None
print(json.dumps({'prefix': sys.prefix}))
'''
    check = subprocess.run([str(python), '-I', '-c', code], capture_output=True, encoding='utf-8', timeout=30)
    assert check.returncode == 0, check.stdout + check.stderr
    run_probe(python, mode, tmp_path/mode)


@pytest.fixture(scope='session')
def distribution(request, tmp_path_factory):
    option = request.config.getoption('--release-exe')
    assert option, 'Use --release-exe with a complete built distribution'
    exe = Path(option).resolve()
    assert exe.is_file(), exe
    # Run a disposable distribution, excluding personal state. Runtime resources
    # are read-only; hard links avoid duplicating large assets on the same drive.
    def copy_resource(source, target):
        try:
            return os.link(source, target)
        except OSError:
            return shutil.copy2(source, target)
    def ignore_personal_state(directory, names):
        return ['runtime'] if Path(directory) == exe.parent and 'runtime' in names else []
    target = tmp_path_factory.mktemp('distribution') / 'app'
    shutil.copytree(exe.parent, target, copy_function=copy_resource, ignore=ignore_personal_state)
    return target / exe.name


@pytest.mark.parametrize('mode', ['check', 'project'])
def test_built_executable(distribution, tmp_path, mode):
    exe = distribution
    forbidden = {'torch', 'torchvision', 'retired_features', 'models'}
    for path in exe.parent.rglob('*'):
        assert not forbidden.intersection(path.relative_to(exe.parent).parts), path
        assert path.suffix != '.pt', path
    args = ['--check'] if mode == 'check' else [
        '--project-once', str(ROOT/'tests/fixtures/registration/game2.png'), '--layout', 'layout_05',
        '--route-version', 'route_lhp_solo', '--output', str(tmp_path/'projection')]
    completed = subprocess.run([str(exe), *args], cwd=tmp_path, capture_output=True,
                               encoding='utf-8', timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    if mode == 'check':
        assert result['icon_templates_exist'] and result['profiles']
        assert all(not v['errors'] for p in result['profiles'] for v in p['template_originals'].values())
    else:
        assert result['accepted'] and result['layout'] == 'layout_05'
        check_projection(tmp_path/'projection')
