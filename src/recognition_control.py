"""Cooperative recognition cancellation; no changes to scores or candidate ordering."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock


@dataclass(eq=False)
class RecognitionRequest:
    """A screenshot request survives later map toggles, never an early toggle.

    Raw Input timestamps may reach the UI after the worker finishes capturing.
    Compare occurrence times under a lock instead of Qt signal delivery order.
    """

    work_id: str
    generation: int
    session: str
    context: tuple
    cancel: Event
    window: list | None = None
    _captured_at: float | None = None
    _first_map_change: float | None = None
    _capture_invalid: bool = False
    _lock: object = field(default_factory=Lock, repr=False)

    def map_changed(self, occurred_at):
        with self._lock:
            if self._first_map_change is None or occurred_at < self._first_map_change:
                self._first_map_change = occurred_at
            self._check_capture()

    def captured(self, completed_at):
        with self._lock:
            self._captured_at = completed_at
            self._check_capture()
            return not self.cancel.is_set()

    def _check_capture(self):
        if (self._captured_at is not None and self._first_map_change is not None
                and self._first_map_change <= self._captured_at):
            self._capture_invalid = True
            self.cancel.set()

    @property
    def snapshot_ready(self):
        with self._lock:
            return self._captured_at is not None and not self.cancel.is_set()

    @property
    def capture_invalid(self):
        with self._lock:
            return self._capture_invalid


class RecognitionCancelled(Exception):
    """Obsolete request, deliberately not a recognition failure."""


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RecognitionCancelled()


@contextmanager
def recognition_lock(lock, cancelled):
    check_cancelled(cancelled)
    while not lock.acquire(timeout=.02):
        check_cancelled(cancelled)
    try:
        check_cancelled(cancelled)
        yield
    finally:
        lock.release()
