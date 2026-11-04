"""Release integrity and retained maintenance behavior after retiring unused adapters."""
import hashlib
import json
import os

import numpy as np
import pytest

from src.registration.template_matching.resources import verify_fingerprint, verify_module_provenance
from src.registration.template_matching.matcher import read_image
import cv2


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


def test_scale_fingerprint_detects_same_size_same_timestamp_edit(tmp_path):
    source = tmp_path / 'scale_config.json'
    source.write_bytes(b'{"scale":1}')
    stamp = source.stat()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    verify_fingerprint(source, expected)
    source.write_bytes(b'{"scale":2}')
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='资源已改变'):
        verify_fingerprint(source, expected)


def test_current_matcher_reads_unicode_png_and_rejects_missing_or_corrupt_files(tmp_path):
    image = np.arange(48, dtype=np.uint8).reshape(3,4,4)
    path = tmp_path/'路线.png'
    path.write_bytes(cv2.imencode('.png',image)[1].tobytes())
    np.testing.assert_array_equal(read_image(path),image[:,:,:3])
    with pytest.raises(FileNotFoundError):read_image(tmp_path/'缺失.png')
    invalid=tmp_path/'损坏.png';invalid.write_bytes(b'not an image')
    with pytest.raises(OSError):read_image(invalid)
