"""One serial background writer, with at most one pending latest diagnostic."""
from threading import Condition, Thread
from queue import SimpleQueue, Empty


class LatestDiagnosticWriter:
    def __init__(self, save):
        self._save=save
        self._condition=Condition()
        self._pending=None
        self._closed=False
        self._thread=None
        self._errors=SimpleQueue()

    def submit(self, image, **details):
        with self._condition:
            if self._closed:return
            self._pending=(image,details)
            if self._thread is None:
                self._thread=Thread(target=self._run,name='map-probe-diagnostics',daemon=True)
                self._thread.start()
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._pending is not None or self._closed)
                if self._pending is None:return
                image,details=self._pending
                self._pending=None
            try:self._save(image,**details)
            except Exception as exc:self._errors.put(str(exc))

    def pop_error(self):
        try:return self._errors.get_nowait()
        except Empty:return None

    def close(self):
        # Flush the latest pending record without blocking the GUI on disk I/O.
        with self._condition:
            self._closed=True
            self._condition.notify()
