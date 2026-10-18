"""Release integrity and retained maintenance behavior after retiring unused adapters."""
import hashlib
import json
import os
import importlib
import sys

import numpy as np
import pytest

from src.registration.template_matching.resources import verify_fingerprint, verify_module_provenance
from tools.assets.image_io import read_image, write_image, imread_unicode, imwrite_unicode
from tools.common.calibration_assets import layout_numbers


@pytest.mark.parametrize('change', ['modified', 'added', 'removed'])
def test_source_manifest_rejects_changed_code(tmp_path, change):
    source = tmp_path / 'matcher.py'
    source.write_bytes(b'original code')
    (tmp_path / 'provenance.json').write_text(json.dumps({
        'files': {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}))
    verify_module_provenance(tmp_path)
    if change == 'modified':
        source.write_bytes(b'changed code')
    elif change == 'added':
        (tmp_path / 'retired.py').write_bytes(b'unexpected code')
    else:
        source.unlink()
    with pytest.raises(ValueError, match='指纹'):
        verify_module_provenance(tmp_path)


def test_calibration_fingerprint_detects_same_size_same_timestamp_edit(tmp_path):
    source = tmp_path / 'calibration.json'
    source.write_bytes(b'{"scale":1}')
    stamp = source.stat()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    verify_fingerprint(source, expected)
    source.write_bytes(b'{"scale":2}')
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='资源已改变'):
        verify_fingerprint(source, expected)


def test_shared_image_io_preserves_unicode_alpha_and_failure_contracts(tmp_path):
    image = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
    path = tmp_path / '路线.png'
    write_image(path, image)
    np.testing.assert_array_equal(imread_unicode(path, -1), image)
    np.testing.assert_array_equal(read_image(path), image[:, :, :3])
    assert imwrite_unicode(tmp_path / '副本.png', image)
    assert imread_unicode(tmp_path / '缺失.png') is None
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / '缺失.png')
    invalid = tmp_path / '损坏.png'
    invalid.write_bytes(b'not an image')
    assert imread_unicode(invalid) is None
    with pytest.raises(ValueError, match='无法读取'):
        read_image(invalid)


def test_calibration_discovers_new_layout_numbers(tmp_path):
    for name in ('layout_01', 'layout_27', 'layout_28', 'layout_backup', 'other'):
        (tmp_path / name).mkdir()
    (tmp_path / 'layout_29').write_text('not a directory')
    assert layout_numbers(tmp_path) == [1, 27, 28]


@pytest.mark.parametrize('publish', [False, True])
@pytest.mark.parametrize('command', ['calibrate_route_assets_batch', 'calibrate_zhanshi_routes'])
def test_calibration_commands_resolve_output_from_explicit_arguments(tmp_path, monkeypatch, command, publish):
    module = importlib.import_module('tools.assets.calibration.' + command)
    clean = tmp_path / 'clean'
    (clean / 'layout_28').mkdir(parents=True)
    sources = tmp_path / 'sources'
    sources.mkdir()
    (sources / '28route.jpg').write_bytes(b'input handled by calibration boundary')
    stage, official = tmp_path / 'stage', tmp_path / 'official'
    monkeypatch.setattr(module, 'CLEAN_ROOT', clean)
    monkeypatch.setattr(module, 'ASSET_ROOT', stage)
    output_helper = 'calibration_root' if command == 'calibrate_route_assets_batch' else 'route_asset_root'
    monkeypatch.setattr(module, output_helper, lambda *, publish=False: official if publish else stage)
    if command == 'calibrate_zhanshi_routes':
        monkeypatch.setattr(module, 'calibration_root', lambda *, publish=False:
                            (official if publish else stage)/'metadata')
    monkeypatch.setattr(sys, 'argv', [command] + (['--publish'] if publish else []))
    outputs = []

    def calibrated(*args):
        outputs.append(module.ASSET_ROOT)
        return {'layout': 'layout_28', 'floors': {f: {'status': 'PASS'} for f in module.FLOORS}}

    if command == 'calibrate_route_assets_batch':
        monkeypatch.setattr(module, 'ROUTE_ROOT', sources)
        monkeypatch.setattr(module, 'process_variant', calibrated)
        monkeypatch.setattr(module, 'write_summary', lambda payloads, failures: None)
    else:
        monkeypatch.setattr(module, 'SOURCE_ROOT', sources)
        monkeypatch.setattr(module, 'calibrate', calibrated)
    module.main()
    assert outputs == [official if publish else stage]
    if command == 'calibrate_zhanshi_routes':
        assert module.CALIBRATION_ROOT == (official if publish else stage)/'metadata'
