"""Persist unhandled Qt callback tracebacks before preserving Qt's fatal exit."""
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import traceback

from PyQt6.QtCore import qFatal


@contextmanager
def log_unhandled_qt_exceptions(log_path):
    log_path = Path(log_path)
    previous_hook = sys.excepthook

    def unhandled_exception(exc_type, exc_value, exc_traceback):
        text = (
            f'\n[{datetime.now().astimezone().isoformat(timespec="milliseconds")}] '
            f'Unhandled Qt/Python callback exception; pid={os.getpid()}; '
            f'thread={threading.current_thread().name}; '
            f'native_thread_id={threading.get_native_id()}\n'
            + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        )
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8', errors='backslashreplace') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            stream = sys.stderr if sys.stderr is not None else sys.__stderr__
            if stream is not None:
                try:
                    stream.write(f'Cannot write Qt exception log {log_path}: {error}\n{text}')
                    stream.flush()
                except (OSError, ValueError):
                    pass  # A missing/broken console must not prevent the fatal exit.
        finally:
            # A custom excepthook makes PyQt return instead of aborting. Keep the
            # existing failure behavior; recording an error is not recovery.
            qFatal('Unhandled Python exception')

    sys.excepthook = unhandled_exception
    try:
        yield
    finally:
        sys.excepthook = previous_hook
