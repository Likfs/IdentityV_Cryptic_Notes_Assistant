"""Verified, process-pinned floor features; map one floor at a time without pickle."""
from contextlib import contextmanager
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
import weakref

import numpy as np

from src.recognition_control import check_cancelled

FORMAT_VERSION = 1
_FOOTER = struct.Struct('<Q8s')
_MAGIC = b'IDVFEAT1'
_NAME = re.compile(r'[0-9a-f]{16}\.[0-9a-f]{64}\.feat')
_STORES = weakref.WeakSet()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def file_digest(stream, cancelled=None):
    stream.seek(0)
    result = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        check_cancelled(cancelled)
        result.update(block)
    return result.hexdigest()


def _pin(path):
    if os.name != 'nt':
        return path.open('rb')
    # A validated Windows cache file must not be rewritten/deleted while its
    # session lives, even between mappings. New generations use different names.
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(fd, 'rb')


def _encode(value, stream):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        padding = (-stream.tell()) % 64
        stream.write(b'\0' * padding)
        offset = stream.tell()
        if array.size:
            stream.write(memoryview(array).cast('B'))
        return {'array': [offset, array.dtype.str, list(array.shape)]}
    if isinstance(value, dict):
        return {key: _encode(item, stream) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, stream) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate(value, limit):
    if isinstance(value, dict):
        if set(value) == {'array'}:
            offset, dtype, shape = value['array']
            dtype = np.dtype(dtype)
            if (dtype not in (np.dtype('float32'), np.dtype('uint8'))
                    or type(offset) is not int or offset < 0 or offset % 64
                    or not isinstance(shape, list) or not 1 <= len(shape) <= 3
                    or any(type(n) is not int or n < 0 for n in shape)
                    or offset + math.prod(shape) * dtype.itemsize > limit):
                raise ValueError('识别特征数组格式或边界无效')
        else:
            for item in value.values():
                _validate(item, limit)
    elif isinstance(value, list):
        for item in value:
            _validate(item, limit)


def _decode(value, buffer):
    if isinstance(value, dict):
        if set(value) == {'array'}:
            offset, dtype, shape = value['array']
            result = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
            result.flags.writeable = False
            return result
        return {key: _decode(item, buffer) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, buffer) for item in value]
    return value


class FeatureFile:
    def __init__(self, path, identity, cancelled=None):
        self.path = path
        self.stream = _pin(path)
        try:
            if file_digest(self.stream, cancelled) != path.name.split('.')[1]:
                raise ValueError('识别特征文件指纹不一致')
            size = self.stream.seek(0, 2)
            if size < _FOOTER.size:
                raise ValueError('识别特征文件不完整')
            self.stream.seek(-_FOOTER.size, 2)
            length, magic = _FOOTER.unpack(self.stream.read(_FOOTER.size))
            if magic != _MAGIC or not 0 < length <= min(1024 * 1024, size - _FOOTER.size):
                raise ValueError('识别特征元数据无效')
            self.stream.seek(size - _FOOTER.size - length)
            data = json.loads(self.stream.read(length))
            if data['format'] != FORMAT_VERSION or data['identity'] != identity:
                raise ValueError('识别特征依赖已改变')
            self.tree = data['target']
            _validate(self.tree, size - _FOOTER.size - length)
        except BaseException:
            self.stream.close()
            raise

    @contextmanager
    def mapped(self):
        buffer = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        target = None
        try:
            target = _decode(self.tree, buffer)
            yield target
        finally:
            # Callers must not retain target arrays beyond this context.
            if target is not None:
                target.clear()
            buffer.close()

    def close(self):
        self.stream.close()


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        # Content-addressed filenames can exceed MAX_PATH even when the cache
        # directory fits. All cache I/O, including CreateFileW pinning, uses this path.
        if os.name == 'nt':
            value = str(self.root)
            prefix = '\\\\?\\'
            if not value.startswith(prefix):
                value = prefix + 'UNC' + value[1:] if value.startswith('\\\\') else prefix + value
                self.root = Path(value)
        self.files = {}
        self.built = 0
        _STORES.add(self)

    def ensure(self, identity, build, *, cancelled=None):
        check_cancelled(cancelled)
        key = digest(identity)
        if key in self.files:
            return key
        for path in sorted(self.root.glob(f'{key[:16]}.*.feat')):
            if not _NAME.fullmatch(path.name) or path.is_symlink():
                continue
            try:
                self.files[key] = FeatureFile(path, identity, cancelled)
                return key
            except (ValueError, OSError, KeyError, TypeError, struct.error):
                pass  # Rebuild this floor only; original PNG remains authoritative.
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            target = build()
            check_cancelled(cancelled)
            with tempfile.NamedTemporaryFile(dir=self.root, prefix='feature-', suffix='.tmp',
                                             mode='w+b', delete=False) as stream:
                temporary = Path(stream.name)
                tree = _encode(target, stream)
                del target
                raw = json.dumps(dict(format=FORMAT_VERSION, identity=identity, target=tree),
                                 ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                stream.write(raw)
                stream.write(_FOOTER.pack(len(raw), _MAGIC))
                stream.flush()
                os.fsync(stream.fileno())
                fingerprint = file_digest(stream, cancelled)
            check_cancelled(cancelled)
            destination = self.root / f'{key[:16]}.{fingerprint}.feat'
            # Another process may already have published the identical generation.
            if destination.exists():
                try:
                    self.files[key] = FeatureFile(destination, identity, cancelled)
                    return key
                except (ValueError, OSError, KeyError, TypeError, struct.error):
                    pass
            os.replace(temporary, destination)
            self.files[key] = FeatureFile(destination, identity, cancelled)
            self.built += 1
            return key
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def mapped(self, key):
        return self.files[key].mapped()

    def prune(self):
        # Only our named immutable artifacts, never arbitrary runtime/user files.
        active = {file.path for store in list(_STORES) for file in list(store.files.values())}
        for path in self.root.glob('*.feat'):
            if _NAME.fullmatch(path.name) and path not in active and not path.is_symlink():
                try:
                    with path.open('rb') as stream:
                        stream.seek(-len(_MAGIC), 2)
                        if stream.read(len(_MAGIC)) != _MAGIC:
                            continue
                    path.unlink()
                except OSError:
                    pass  # A different process may still pin this generation.

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()
