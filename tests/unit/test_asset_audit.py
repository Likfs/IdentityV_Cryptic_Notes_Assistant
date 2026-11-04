import json
import os
from threading import Event, Lock, get_ident

from PIL import Image
import pytest

from src import asset_audit as audit_module
from src.asset_audit import AssetAudit


def png(path, color='red', size=(35, 27)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new('RGB', size, color) as image:
        image.save(path, format='PNG', compress_level=0)
    return path


def test_reuses_full_validation_but_reads_and_hashes_every_startup(tmp_path, monkeypatch):
    path = png(tmp_path/'floor.png')
    with AssetAudit(tmp_path) as audit:
        first = audit.image(path)
        assert audit.stats == dict(hashed=1, decoded=1, reused=0)
    reads = []
    read = audit_module._read
    def tracked(path):
        reads.append(path)
        return read(path)
    monkeypatch.setattr(audit_module, '_read', tracked)
    monkeypatch.setattr(Image.Image, 'load', lambda *_: pytest.fail('Unchanged PNG decoded again'))
    with AssetAudit(tmp_path) as audit:
        assert audit.image(path) == first
        audit.verify({'floor.png': {key: first[key] for key in ('bytes', 'sha256')}})
        assert audit.stats == dict(hashed=1, decoded=0, reused=1)
    assert reads == [path, path]


def test_same_size_and_mtime_replacement_is_revalidated(tmp_path):
    path = png(tmp_path/'floor.png')
    with AssetAudit(tmp_path) as audit:
        first = audit.image(path)
    info = path.stat()
    png(path, 'blue')
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert path.stat().st_size == info.st_size
    with AssetAudit(tmp_path) as audit:
        current = audit.image(path)
        assert current['sha256'] != first['sha256']
        assert audit.stats == dict(hashed=1, decoded=1, reused=0)
        with pytest.raises(ValueError, match='变化'):
            audit.verify({'floor.png': {key: first[key] for key in ('bytes', 'sha256')}})


@pytest.mark.parametrize('damage', ['json', 'checksum', 'dimensions', 'validator'])
def test_invalid_cache_or_changed_validator_forces_full_decode(tmp_path, monkeypatch, damage):
    path = png(tmp_path/'floor.png')
    with AssetAudit(tmp_path) as audit:
        audit.image(path)
        cache = audit.cache_path
    wrapper = json.loads(cache.read_bytes())
    if damage == 'json':
        cache.write_bytes(b'{broken')
    elif damage == 'validator':
        monkeypatch.setattr(audit_module, '_validation_identity', lambda: 'changed-validator')
    else:
        if damage == 'checksum':
            wrapper['sha256'] = 'bad'
        else:
            wrapper['payload']['images'] = {'0'*64: [-1, 27]}
            wrapper['sha256'] = audit_module._digest(wrapper['payload'])
        cache.write_text(json.dumps(wrapper))
    with AssetAudit(tmp_path) as audit:
        assert audit.image(path)['size'] == (35, 27)
        assert audit.stats['decoded'] == 1


def test_corrupt_replacement_is_rejected_without_replacing_cache(tmp_path):
    path = png(tmp_path/'floor.png')
    with AssetAudit(tmp_path) as audit:
        audit.image(path)
        cache = audit.cache_path
    previous = cache.read_bytes()
    raw = path.read_bytes()
    info = path.stat()
    path.write_bytes(b'x'*len(raw))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(OSError):
        with AssetAudit(tmp_path) as audit:
            audit.image(path)
    assert cache.read_bytes() == previous


def test_four_worker_limit_and_cancellation_keep_progress_on_coordinator(tmp_path, monkeypatch):
    paths = [png(tmp_path/f'{index}.png') for index in range(12)]
    coordinator = get_ident()
    lock = Lock()
    release = Event()
    threads = set()
    started = []
    read = audit_module._read
    def blocked(path):
        with lock:
            started.append(path)
            threads.add(get_ident())
        assert release.wait(3)
        return read(path)
    monkeypatch.setattr(audit_module, '_read', blocked)
    class Cancelled(Exception):
        pass
    def progress(stage, done, total):
        assert get_ident() == coordinator and total == 12
        if len(started) == 4:
            release.set()
            raise Cancelled()
    with pytest.raises(Cancelled):
        with AssetAudit(tmp_path) as audit:
            audit.images(paths, progress=progress)
    assert len(threads) == 4 and coordinator not in threads
    assert len(started) == 4
    assert not (tmp_path/'runtime/cache/asset_audit/png_validation.json').exists()


def test_change_between_read_stats_is_rejected(tmp_path, monkeypatch):
    path = png(tmp_path/'floor.png')
    read_bytes = type(path).read_bytes
    def changing(self):
        raw = read_bytes(self)
        if self == path:
            self.write_bytes(raw+b'changed')
        return raw
    monkeypatch.setattr(type(path), 'read_bytes', changing)
    with pytest.raises(ValueError, match='读取期间素材发生变化'):
        with AssetAudit(tmp_path) as audit:
            audit.image(path)


def test_identical_contents_reuse_validation_and_cache_prunes_absent_images(tmp_path):
    first = png(tmp_path/'first.png')
    second = png(tmp_path/'second.png', 'blue')
    with AssetAudit(tmp_path) as audit:
        audit.images([first, second])
    second.unlink()
    duplicate = tmp_path/'duplicate.png'
    duplicate.write_bytes(first.read_bytes())
    with AssetAudit(tmp_path) as audit:
        audit.images([first, duplicate])
        assert audit.stats == dict(hashed=2, decoded=0, reused=2)
        cache = audit.cache_path
    assert len(json.loads(cache.read_bytes())['payload']['images']) == 1
