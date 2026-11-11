"""Bounded startup hashing with content-addressed results of full PNG validation."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
import hashlib
import importlib.util
import json
import logging
import marshal
import os
from pathlib import Path
import tempfile
from threading import Lock

from PIL import Image, ImageFile, __version__ as pillow_version
from src.paths import published_resource_path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validation_identity():
    # A validator or decoder change must not inherit an older validation result.
    code = importlib.util.find_spec(__name__).loader.get_code(__name__)
    return _digest(dict(validator=hashlib.sha256(marshal.dumps(code)).hexdigest(),
                        pillow=pillow_version, max_pixels=Image.MAX_IMAGE_PIXELS,
                        truncated=ImageFile.LOAD_TRUNCATED_IMAGES))


def _read(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino', 'st_dev')
    if len(raw) != after.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f'读取期间素材发生变化：{path}')
    return raw, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class AssetAudit:
    """Only hashes and dimensions survive a job; decoded images never enter the cache."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache_path = self.path('runtime/cache/asset_audit/png_validation.json')
        self.identity = _validation_identity()
        self.validated = self._load()
        self.records = {}
        self.seen = set()
        self.lock = Lock()
        self.stats = dict(hashed=0, decoded=0, reused=0)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='asset-audit')

    def path(self, name):
        path = (self.root/name).resolve()
        path.relative_to(self.root)
        return path

    def _load(self):
        try:
            wrapper = json.loads(self.cache_path.read_bytes())
            payload = wrapper['payload']
            if wrapper['sha256'] != _digest(payload) or payload['identity'] != self.identity:
                return {}
            entries = payload['images']
            if not isinstance(entries, dict):
                return {}
            for fingerprint, size in entries.items():
                if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                        or any(c not in '0123456789abcdef' for c in fingerprint)
                        or not isinstance(size, list) or len(size) != 2
                        or any(type(n) is not int or n <= 0 for n in size)):
                    return {}
            return entries
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.pool.shutdown(wait=True, cancel_futures=True)
        if kind is None:
            self._save()

    def _save(self):
        payload = dict(identity=self.identity, images={key: self.validated[key] for key in sorted(self.seen)})
        raw = json.dumps(dict(payload=payload, sha256=_digest(payload)), sort_keys=True).encode()
        temporary = None
        try:
            if self.cache_path.exists() and self.cache_path.read_bytes() == raw:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cache_path)
        except OSError:
            # Validation already succeeded; an unwritable optional cache only costs time next startup.
            logging.getLogger(__name__).warning('无法保存图片校验缓存，下次将重新解码校验', exc_info=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _image(self, path):
        raw, record = _read(self.path(path))
        fingerprint = record['sha256']
        with self.lock:
            size = self.validated.get(fingerprint)
        reused = size is not None
        if size is None:
            with Image.open(BytesIO(raw)) as image:
                if image.format != 'PNG':
                    raise ValueError(f'楼层图片必须为 PNG：{path}')
                image.load()
                size = list(image.size)
        with self.lock:
            self.validated[fingerprint] = size
            self.seen.add(fingerprint)
            self.stats['hashed'] += 1
            self.stats['reused' if reused else 'decoded'] += 1
        return dict(record, size=tuple(size))

    def _jobs(self, paths, operation, stage, progress):
        """Keep at most four jobs outstanding and poll cancellation on the coordinator."""
        paths = list(paths)
        pending = {}
        results = {}
        iterator = iter(paths)
        try:
            while pending or len(results) < len(paths):
                if progress is not None:
                    progress(stage, len(results), len(paths))
                while len(pending) < 4:
                    path = next(iterator, None)
                    if path is None:
                        break
                    pending[self.pool.submit(operation, path)] = path
                completed, _ = wait(pending, timeout=.05, return_when=FIRST_COMPLETED)
                for future in completed:
                    results[pending.pop(future)] = future.result()
            if progress is not None:
                progress(stage, len(results), len(paths))
            return results
        finally:
            for future in pending:
                future.cancel()
            # A caller may catch an invalid PNG and continue scanning another difficulty.
            # Do not leave those workers mutating the cache during a later operation.
            if pending:
                wait(pending)

    def images(self, paths, *, progress=None, stage='校验素材'):
        paths = list(dict.fromkeys(self.path(path) for path in paths))
        missing = [path for path in paths if path not in self.records]
        self.records.update(self._jobs(missing, self._image, stage, progress))
        return {path: self.records[path] for path in paths}

    def image(self, name):
        path = self.path(name)
        if path not in self.records:
            self.images([path])
        return self.records[path]

    def verify(self, files, *, progress=None, stage='核验已发布资源'):
        """Always read and hash current bytes again, including same-stat replacements."""
        def check(name):
            _, actual = _read(published_resource_path(self.root, name))
            if actual != files[name]:
                raise ValueError(f'启动同步期间文件发生变化：{name}')
        self._jobs(files, check, stage, progress)
