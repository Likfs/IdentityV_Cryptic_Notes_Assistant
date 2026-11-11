"""Explicit acceptance: requires supplied paths, never silently skips missing dependencies."""
import json
import hashlib
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


@pytest.fixture
def distribution(request, tmp_path):
    option = request.config.getoption('--release-exe')
    assert option, 'Use --release-exe with a complete built distribution'
    exe = Path(option).resolve()
    assert exe.is_file(), exe
    # Run a disposable distribution, excluding personal state. Runtime resources
    # are read-only; hard links avoid duplicating large assets on the same drive.
    def copy_resource(source, target):
        # CreateHardLinkW needs extended paths independently of Python's file I/O.
        # Keep the installation itself deep: only the fixture's copy uses this form.
        if os.name == 'nt':
            source = Path('\\\\?\\' + str(Path(source).resolve()))
            target = Path('\\\\?\\' + str(Path(target).resolve()))
        try:
            return os.link(source, target)
        except OSError:
            return shutil.copy2(source, target)
    def ignore_personal_state(directory, names):
        if Path(directory) != exe.parent:
            return []
        excluded = {'runtime'}
        if request.node.callspec.params['mode'] in ('project', 'client_scale', 'fixed_height'):
            excluded.update(('assets', 'configs'))
        return list(excluded.intersection(names))
    target = tmp_path/'app'
    # Keep the actual EXE/resources in a deep installation, as in the reported failure.
    while len(str(target)) < 145:
        target = target.parent/'nested_installation'/target.name
    shutil.copytree(exe.parent, target, copy_function=copy_resource, ignore=ignore_personal_state)
    if request.node.callspec.params['mode'] in ('project', 'client_scale', 'fixed_height'):
        # Historical screenshot geometry uses independent fixtures, never production assets/scales.
        from tests.support.aligned_runtime import build_nightmare_replay_package, build_phosphorescence_replay_package
        build = (build_phosphorescence_replay_package if request.node.callspec.params['mode'] != 'project'
                 else build_nightmare_replay_package)
        replay = build(tmp_path/'replay')
        for directory in ('assets', 'configs'):
            shutil.copytree(replay/directory, target/directory)
    return target / exe.name


@pytest.mark.parametrize('mode', ['check', 'project', 'client_scale', 'fixed_height', 'gui'])
def test_built_executable(distribution, tmp_path, mode):
    exe = distribution
    feature_example = exe.parent/'runtime/cache/structure_features'/('0'*16)/('0'*16+'.'+'0'*64+'.feat')
    assert len(str(feature_example)) > 260
    assert (exe.parent/'_internal').is_dir()
    assert (exe.parent/'assets/maps').is_dir()
    assert (exe.parent/'assets/app_ui/floor_controls.png').is_file()
    assert (exe.parent/'configs/runtime_resources.json').is_file()
    assert not (exe.parent/'src').exists()
    assert not (exe.parent/'_internal/assets').exists()
    assert not (exe.parent/'_internal/configs').exists()
    assert not list(exe.parent.glob('*.dll'))
    assert not list(exe.parent.glob('*.pyd'))
    forbidden = {'torch', 'torchvision', 'retired_features', 'models'}
    for path in exe.parent.rglob('*'):
        assert not forbidden.intersection(path.relative_to(exe.parent).parts), path
        assert path.suffix != '.pt', path
    cases = json.loads((ROOT/'tests/fixtures/retrieval/cases.json').read_text(encoding='utf-8'))['projection_regressions']
    case = (next(c for c in cases if c['id'] == 'phosphorescence_layout13_b2_windowed')
            if mode in ('client_scale', 'fixed_height') else cases[0])
    screenshot = ROOT/case['screenshot']
    assert hashlib.sha256(screenshot.read_bytes()).hexdigest() == case['sha256']
    client_size = case.get('capture_client_size', [2560, 1440])
    if mode == 'fixed_height':
        import cv2
        # Resampled historical pixels exercise frozen FH scale conversion;
        # this is a synthetic replay, not a native fixed-height game capture.
        image = cv2.imread(str(screenshot))
        ratio = 2/(1.5*(1920000/1779556))
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        screenshot = tmp_path/'fixed_height.png'
        assert cv2.imwrite(str(screenshot), image)
        client_size = [2560, 1334]
    args = ['--check'] if mode == 'check' else ['--smoke'] if mode == 'gui' else [
        '--project-once', str(screenshot), '--layout', case['selected_layout'],
        '--map', case['profile'].split('/')[0], '--difficulty', case['profile'].split('/')[1],
        '--route-version', case['variant_id'], '--capture-client-size',
        *map(str, client_size), '--output', str(tmp_path/'projection')]
    before = {p.name: p.read_bytes() for p in (exe.parent/'configs').iterdir()}
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    completed = subprocess.run([str(exe), *args], cwd=tmp_path, capture_output=True,
                               encoding='utf-8', errors='replace', timeout=120, env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert {p.name: p.read_bytes() for p in (exe.parent/'configs').iterdir()} == before
    assert not (exe.parent/'_internal/runtime').exists()
    assert not (tmp_path/'runtime').exists()
    if mode == 'gui':
        assert (exe.parent/'runtime/cache/asset_audit/png_validation.json').is_file()
        cache_root = exe.parent/'runtime/cache/structure_features'
        if os.name == 'nt':
            cache_root = Path('\\\\?\\'+str(cache_root))
        bundle = json.loads((exe.parent/'configs/runtime_resources.json').read_text(encoding='utf-8'))['payload']
        catalog = json.loads((exe.parent/'configs/map_catalog.json').read_text(encoding='utf-8'))
        selected = catalog['maps'][0]['difficulties'][0]
        expected = len(bundle['recognition'][selected['clean_root']]['targets'])
        assert len(list(cache_root.glob('*/*.feat'))) == expected
        return
    result = json.loads(completed.stdout)
    if mode == 'check':
        assert result['icon_templates_exist'] and result['profiles']
        assert all(not v['errors'] for p in result['profiles'] for v in p['template_originals'].values())
    else:
        assert result['accepted'] and result['layout'] == case['selected_layout']
        report = check_projection(tmp_path/'projection')
        assert report['floor'] == case['expected_floor']
        template = report['registration']['validation']['template']
        assert template['saved_scale'] == case['minimum_scale']
        if mode == 'client_scale':
            assert template['scale'] == 2.
            assert template['capture_client']['width'] == 1920
            assert template['capture_client']['height'] == 1080
        if mode == 'fixed_height':
            assert template['scale'] == pytest.approx(1.5*(1920000/1779556))
            assert template['capture_client']['visible_design_size'] == [1440, 750]
            assert template['capture_client']['resolution_policy'] == 3
