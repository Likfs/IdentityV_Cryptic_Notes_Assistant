"""Shared locking and content verification for fixed-scale original images."""
from threading import RLock
import hashlib
import json
from pathlib import Path

resource_lock = RLock()
verified_fingerprints = {}


def backend_info():
    """Stable execution metadata consumed by saved diagnostics."""
    return dict(effective_backend='template', device=None, fallback_reason=None, initialization_ms=0.,
                requested_backend='template', version='fixed_scale_template', cpu_double_verification=False)


def verify_fingerprint(path, expected):
    with resource_lock:
        stat = path.stat()
        stamp = (str(path), stat.st_mtime_ns, stat.st_size, expected)
        # Small JSON dependencies must be read even when length and timestamp are unchanged.
        cacheable = path.suffix.lower() != '.json'
        if not cacheable or stamp not in verified_fingerprints:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'尺度对应资源已改变，请重新导入：{path.name}')
            if cacheable:
                verified_fingerprints[stamp] = True


def verify_module_provenance(directory=None):
    """Reject stale or incomplete source fingerprints before creating a release."""
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    manifest = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    expected = manifest['files']
    if set(expected) != {p.name for p in directory.glob('*.py')}:
        raise ValueError('模板模块指纹清单与当前源码文件集合不一致')
    for name, digest in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'模板模块源码指纹不一致：{name}')
