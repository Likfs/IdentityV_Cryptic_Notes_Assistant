"""Real Qt callback failures in an isolated process, without game input or windows."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('mode', ['bootstrap', 'after', 'timer', 'single_shot'])
def test_timer_exception_is_logged_before_fatal_exit(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent('''\
        import os
        from pathlib import Path
        import sys
        from types import SimpleNamespace
        from PyQt6 import QtCore

        root, output, mode = sys.argv[1:]
        sys.path.insert(0, root)
        # Replace only the final native abort to avoid Windows crash dumps in tests.
        QtCore.qFatal = lambda message: os._exit(86)
        sys.excepthook = lambda *args: os._exit(86)
        sys.stdout = sys.stderr = None  # Match a pythonw launch.

        def explode_timer():
            raise ValueError('qt timer traceback probe')

        if mode == 'bootstrap':
            from src import __main__ as entry, runtime_resources, route_projection_app, asset_sync
            entry.LOG_ROOT = Path(output)
            asset_sync.synchronize = lambda root, **kwargs: []
            runtime_resources.read_bundle = lambda: {}
            entry.MapCatalog = lambda: SimpleNamespace(profiles={('test', 'test'): SimpleNamespace(key='test/test')})
            entry.load_settings = lambda catalog: dict(map_id='test', difficulty_id='test')
            class Window(QtCore.QObject):
                def __init__(self):
                    super().__init__()
                    self.timer = QtCore.QTimer(self)
                    self.timer.setSingleShot(True)
                    self.timer.timeout.connect(explode_timer)
                    self.timer.start(0)
                def close(self):
                    QtCore.QCoreApplication.instance().quit()
            route_projection_app.RouteProjectionApp = Window
            sys.argv = ['src', '--smoke']
            entry.main()
        else:
            from src.qt_exception_logging import log_unhandled_qt_exceptions
            app = QtCore.QCoreApplication([])
            with log_unhandled_qt_exceptions(Path(output)/'qt_exception.log'):
                if mode == 'after':
                    from src.route_projection_app import RouteProjectionApp
                    owner = QtCore.QObject()
                    timer = RouteProjectionApp.after(owner, 0, explode_timer)
                elif mode == 'timer':
                    timer = QtCore.QTimer()
                    timer.timeout.connect(explode_timer)
                    timer.start(0)
                else:
                    QtCore.QTimer.singleShot(0, explode_timer)
                QtCore.QTimer.singleShot(2000, app.quit)
                app.exec()
    ''')
    probe = tmp_path/'timer_probe.py'
    probe.write_text(script, encoding='utf-8')
    output = tmp_path/'logs'
    completed = subprocess.run(
        [sys.executable, '-B', str(probe), str(root), str(output), mode],
        cwd=root, env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
        capture_output=True, timeout=20,
    )
    assert completed.returncode == 86, completed.stderr.decode(errors='replace')
    log = output/'qt_exception.log'
    assert log.is_file(), 'The Qt callback exited without saving its traceback'
    text = log.read_text(encoding='utf-8')
    assert str(probe) in text
    line = script[:script.index("    raise ValueError(")].count('\n') + 1
    assert f'line {line}, in explode_timer' in text
    assert 'ValueError: qt timer traceback probe' in text
    assert 'pid=' in text and 'thread=' in text and 'native_thread_id=' in text
