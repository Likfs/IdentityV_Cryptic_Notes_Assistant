"""Coalesced background preparation, independent of Qt and projection results."""
from threading import Event, Lock, Thread
from . import floors


class FloorPrewarmer:
    def __init__(self):
        self._lock = Lock()
        self._thread = None
        self._pending = None
        self._cancel = Event()
        self._desired = None
        self._closed = False
        self.ready_key = None
        self.last_error = None

    def request(self, key):
        """key is (session, profile, layout, variant), or None to cancel."""
        with self._lock:
            if self._closed or key == self._desired:
                return
            self._cancel.set()
            self._cancel = Event()
            self._desired = key
            self.ready_key = self.last_error = None
            self._pending = (key, self._cancel) if key is not None else None
            if self._pending is not None and self._thread is None:
                self._thread = Thread(target=self._run, name='template-prewarm', daemon=True)
                self._thread.start()

    def close(self):
        with self._lock:
            self._closed = True
            self._cancel.set()
            self._pending = self._desired = self.ready_key = None

    def _run(self):
        while True:
            with self._lock:
                if self._pending is None:
                    self._thread = None
                    return
                key, cancel = self._pending
                self._pending = None
            ready, error = False, None
            try:
                while not cancel.is_set():
                    ready = floors.prewarm(*key[-3:], cancelled=cancel.is_set)
                    if ready or cancel.wait(.05):
                        break
            except Exception as exc:
                # A real projection reports its own resource errors; warming never shows UI.
                error = str(exc)
            with self._lock:
                if not cancel.is_set() and key == self._desired and not self._closed:
                    self.ready_key = key if ready else None
                    self.last_error = error
