import io
import sys

import pytest

from src import qt_exception_logging as logging


class FatalExit(BaseException):
    pass


def test_appends_full_tracebacks_and_restores_hook(tmp_path, monkeypatch):
    log = tmp_path/'logs/qt_exception.log'
    previous = sys.excepthook
    observed = []

    def fatal(message):
        observed.append(log.read_text(encoding='utf-8'))
        raise FatalExit(message)

    monkeypatch.setattr(logging, 'qFatal', fatal)
    with logging.log_unhandled_qt_exceptions(log):
        for message in ('first failure', '第二次异常'):
            try:
                raise ValueError(message)
            except ValueError:
                with pytest.raises(FatalExit):
                    sys.excepthook(*sys.exc_info())
    assert sys.excepthook is previous
    assert len(observed) == 2
    assert 'ValueError: first failure' in observed[0]
    assert 'ValueError: 第二次异常' not in observed[0]
    assert 'ValueError: first failure' in observed[1]
    assert 'ValueError: 第二次异常' in observed[1]
    assert __file__ in observed[1]
    assert 'in test_appends_full_tracebacks_and_restores_hook' in observed[1]


@pytest.mark.parametrize('has_console', [True, False])
def test_unwritable_log_still_calls_fatal(tmp_path, monkeypatch, has_console):
    parent = tmp_path/'not_a_directory'
    parent.write_text('occupied', encoding='utf-8')
    console = io.StringIO() if has_console else None
    monkeypatch.setattr(sys, 'stderr', console)
    monkeypatch.setattr(sys, '__stderr__', console)

    def fatal(message):
        raise FatalExit(message)

    monkeypatch.setattr(logging, 'qFatal', fatal)
    with logging.log_unhandled_qt_exceptions(parent/'qt_exception.log'):
        try:
            raise RuntimeError('callback failure')
        except RuntimeError:
            with pytest.raises(FatalExit):
                sys.excepthook(*sys.exc_info())
    if has_console:
        assert 'Cannot write Qt exception log' in console.getvalue()
        assert 'RuntimeError: callback failure' in console.getvalue()


def test_normal_exit_and_startup_error_restore_hook_without_creating_log(tmp_path):
    log = tmp_path/'qt_exception.log'
    previous = sys.excepthook
    with logging.log_unhandled_qt_exceptions(log):
        assert sys.excepthook is not previous
    assert sys.excepthook is previous
    with pytest.raises(ValueError, match='startup'):
        with logging.log_unhandled_qt_exceptions(log):
            raise ValueError('startup')
    assert sys.excepthook is previous
    assert not log.exists()
