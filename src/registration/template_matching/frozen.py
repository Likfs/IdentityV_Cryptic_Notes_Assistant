"""Process-owned verified resource bytes; source edits take effect after restart."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import tempfile

from src.app_catalog import MapCatalog


class FrozenResources:
    """Caller owns resource_lock. Decoded pixels remain in the bounded floor cache.

    The temporary file retains compressed source bytes, not full-resolution arrays.
    Failed batches never publish partial records or bytes; successful bytes are
    read back from this file even after the original path is replaced or removed.
    """

    def __init__(self, root, config):
        self.origin = root
        self.root = root.resolve()
        self.config = deepcopy(config)
        catalog_path = self.root/'configs/map_catalog.json'
        self.profiles = ({profile.key: profile.floors for profile in MapCatalog(catalog_path).profiles.values()}
                         if catalog_path.exists() else None)
        self._file = None
        self._files = {}
        self._records = {}

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._files.clear()
        self._records.clear()

    def records(self, profile, layout, variant, *, require_scale=True):
        from . import floors
        key = (profile, layout, variant)
        if key not in self._records:
            if self.profiles is not None and profile not in self.profiles:
                raise ValueError(f'未配置地图难度：{profile}')
            expected = floors.FLOORS if self.profiles is None else self.profiles[profile]
            # Pending records are retained and verified, even when they cannot match.
            self._records[key] = floors._source_records(
                self.config, expected, profile, layout, variant,
                require_scale=False, verify_dependencies=self._verify)
        records, config = self._records[key]
        if require_scale and any(item.get('scale_status') == 'pending' and item.get('saved_scale') is None
                                 for item in records.values()):
            raise ValueError('屏幕尺度待标定，请补充可靠保存尺度或实机截图')
        return deepcopy(records), deepcopy(config)

    def _verify(self, dependencies):
        from . import floors
        pending = []
        expected_by_path = {}
        for value, expected in dependencies:
            if value in expected_by_path and expected_by_path[value] != expected:
                raise ValueError('固定资源记录包含冲突指纹，请重新导入并重启')
            expected_by_path[value] = expected
            previous = self._files.get(value)
            if previous is None:
                pending.append((value, expected))
            elif previous[1] != expected:
                raise ValueError('固定资源记录包含冲突指纹，请重新导入并重启')
        if not pending:
            return
        if self._file is None:
            # The OS removes this process-owned file on close/exit, including Windows.
            self._file = tempfile.TemporaryFile(prefix='idv-template-snapshot-', mode='w+b')
        self._file.seek(0, 2)
        start = self._file.tell()
        staged = {}

        def read(pair):
            value, expected = pair
            path = floors.source_path(value, root=self.root)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'尺度对应资源已改变，请重新导入：{path.name}')
            return value, expected, path, data

        try:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix='template-freeze') as pool:
                # Bound pending byte buffers to four files, even for 58-image evidence.
                for offset in range(0, len(pending), 4):
                    for value, expected, path, data in pool.map(read, pending[offset:offset+4]):
                        position = self._file.tell()
                        self._file.write(data)
                        staged[value] = (path, expected, position, len(data))
                        del data
            self._file.flush()
        except BaseException:
            self._file.seek(start)
            self._file.truncate()
            raise
        self._files.update(staged)

    def path(self, value, expected):
        record = self._files[value]
        if record[1] != expected:
            raise ValueError('固定资源指纹不一致，请重启')
        return record[0]

    def read(self, value, expected):
        self.path(value, expected)
        _, _, offset, size = self._files[value]
        if self._file is None:
            raise ValueError('固定资源快照已关闭，请重启')
        self._file.seek(offset)
        data = self._file.read(size)
        if len(data) != size:
            raise OSError('固定资源快照读取不完整，请重启')
        return data
