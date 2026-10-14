"""Cooperative recognition cancellation; no changes to scores or candidate ordering."""
from contextlib import contextmanager


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
