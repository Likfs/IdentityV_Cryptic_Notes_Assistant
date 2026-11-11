"""Own prepared profile indexes across startup, map switches and game sessions."""
import hashlib
from pathlib import Path
from threading import Event, RLock

from src.layout_retrieval import create_layout_retriever
from src.recognition_control import check_cancelled


class RecognitionResources:
    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)
        self.retrievers = {}
        self.fingerprints = {}
        self._lock = RLock()
        self._closed = Event()

    def get(self, profile, backend):
        key = (profile.key, backend)
        with self._lock:
            check_cancelled(self._closed.is_set)
            if key not in self.retrievers:
                directory = (self.cache_root/'structure_features'/hashlib.sha256(profile.key.encode()).hexdigest()[:16]
                             if backend == 'pure_structure' else
                             self.cache_root/'layout_retrieval'/profile.map_id/profile.difficulty_id/backend)
                self.retrievers[key] = create_layout_retriever(
                    backend, profile.clean_root, floors=profile.floors, cache_root=directory)
            return self.retrievers[key]

    def is_ready(self, profile_key, backend):
        with self._lock:
            return not self._closed.is_set() and (profile_key, backend) in self.fingerprints

    def prepare(self, profile, backend, *, cancelled=None, progress=None):
        def obsolete():
            return self._closed.is_set() or (cancelled is not None and cancelled())
        check_cancelled(obsolete)
        retriever = self.get(profile, backend)
        key = (profile.key, backend)
        with self._lock:
            if key in self.fingerprints:
                return self.fingerprints[key]
        kwargs = dict(cancelled=obsolete)
        if backend == 'pure_structure':
            kwargs['progress'] = progress
        fingerprint = retriever.prepare(**kwargs)
        check_cancelled(obsolete)
        with self._lock:
            check_cancelled(obsolete)
            self.fingerprints[key] = fingerprint
        return fingerprint

    def close(self):
        with self._lock:
            self._closed.set()
            retrievers = list(self.retrievers.values())
            self.retrievers.clear()
            self.fingerprints.clear()
        for retriever in retrievers:
            if hasattr(retriever, 'close'):
                retriever.close()
