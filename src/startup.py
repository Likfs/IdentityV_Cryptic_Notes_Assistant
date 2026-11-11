"""Responsive startup resource synchronization before the main window is built."""
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2

from src.app_catalog import MapCatalog, load_settings
from src.recognition_control import RecognitionCancelled, check_cancelled
from src.recognition_resources import RecognitionResources

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from src.capture.region import enable_physical_pixel_coordinates
from src.version import APP_NAME, APP_VERSION

# Keep Qt alive between the startup dialog and the main window's event loop.
_application = None


class StartupCancelled(Exception):
    pass


@dataclass
class StartupState:
    messages: list
    catalog: MapCatalog
    settings: dict
    settings_path: Path
    recognition: RecognitionResources
    preparation_error: str | None = None

    def close(self):
        self.recognition.close()


def prepare_startup_recognition(root, messages, *, progress, cancelled, map_id=None, difficulty_id=None):
    check_cancelled(cancelled)
    from src import paths
    runtime_root = paths.RUNTIME_ROOT if root.resolve() == paths.PROJECT_ROOT else root/'runtime'
    catalog = MapCatalog(root/'configs/map_catalog.json')
    settings_path = runtime_root/'settings.json'
    settings = load_settings(catalog, settings_path)
    if map_id or difficulty_id:
        profile = catalog.profile(map_id or settings['map_id'], difficulty_id or settings['difficulty_id'])
        settings['map_id'], settings['difficulty_id'] = profile.map_id, profile.difficulty_id
    resources = RecognitionResources(runtime_root/'cache')
    state = StartupState(messages, catalog, settings, settings_path, resources)
    try:
        profile = catalog.profiles.get((settings['map_id'], settings['difficulty_id']))
        if profile is not None and profile.asset_status != 'pending':
            cv2.setNumThreads(4)
            stage = f'准备识别资源 · {profile.map_name}／{profile.difficulty_name}'
            progress(stage, 0, 0)
            resources.prepare(profile, settings['layout_recognition_backend'], cancelled=cancelled,
                              progress=lambda done, total: progress(stage, done, total))
        check_cancelled(cancelled)
        return state
    except (StartupCancelled, RecognitionCancelled):
        state.close()
        raise
    except Exception as exc:
        # Valid materials still allow changing profile/settings and retrying in the UI.
        state.preparation_error = str(exc)
        return state
    except BaseException:
        state.close()
        raise


class EmptyAssetsWindow(QWidget):
    """An empty catalog has no profile and never starts capture or recognition."""

    def __init__(self, root):
        super().__init__()
        self.setWindowTitle(f'{APP_NAME} {APP_VERSION}')
        self.resize(640, 360)
        self.setStyleSheet('QWidget { background: #202a35; color: #e7edf4; } '
                           'QPushButton { padding: 9px 20px; background: #354453; } '
                           'QPushButton:disabled { color: #82909e; }')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)
        self.title_label = QLabel('暂无地图素材')
        self.title_label.setStyleSheet('font-size: 22px; font-weight: 600;')
        self.note_label = QLabel('请将底图和路线图按目录结构放入素材目录，完成后重新启动程序。')
        self.note_label.setWordWrap(True)
        self.path_label = QLabel(str(root/'assets/maps'))
        self.path_label.setWordWrap(True)
        self.recognize_button = QPushButton('识别布局')
        self.recognize_button.setEnabled(False)
        self.project_button = QPushButton('截图配准')
        self.project_button.setEnabled(False)
        close_button = QPushButton('关闭程序')
        close_button.clicked.connect(self.close)
        for widget in (self.title_label, self.note_label, self.path_label,
                       self.recognize_button, self.project_button, close_button):
            layout.addWidget(widget)
        self.show()


class _Synchronizer(QThread):
    progress = pyqtSignal(str, int, int)

    def __init__(self, root, synchronize, parent, *, map_id=None, difficulty_id=None):
        super().__init__(parent)
        self.root = root
        self.synchronize = synchronize
        self.messages = None
        self.error = None
        self.cancelled = False
        self.state = None
        self.selection = dict(map_id=map_id, difficulty_id=difficulty_id)

    def _progress(self, stage, done, total):
        if self.isInterruptionRequested():
            raise StartupCancelled()
        self.progress.emit(stage, done, total)

    def run(self):
        try:
            self.messages = self.synchronize(self.root, progress=self._progress)
            self.state = prepare_startup_recognition(self.root, self.messages, progress=self._progress,
                cancelled=self.isInterruptionRequested, **self.selection)
            self.cancelled = self.isInterruptionRequested()
            if self.cancelled:
                self.state.close()
                self.state = None
        except (StartupCancelled, RecognitionCancelled):
            self.cancelled = True
        except BaseException:
            self.error = sys.exc_info()


class StartupDialog(QDialog):
    def __init__(self, root, synchronize, *, map_id=None, difficulty_id=None):
        super().__init__()
        self.setWindowTitle(f'{APP_NAME} · 启动')
        self.setMinimumWidth(520)
        self.setStyleSheet('QDialog { background: #202a35; color: #e7edf4; } '
                           'QLabel { color: #e7edf4; } QPushButton { padding: 7px 20px; }')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)
        self.title_label = QLabel('正在准备地图资源')
        self.title_label.setStyleSheet('font-size: 18px; font-weight: 600;')
        self.stage_label = QLabel('读取资源配置…')
        self.stage_label.setWordWrap(True)
        self.stage_label.setMinimumHeight(54)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat('%v / %m')
        self.note_label = QLabel('素材审核及当前难度识别资源准备完成后，自动打开主窗口')
        self.note_label.setWordWrap(True)
        self.button = QPushButton('取消启动')
        self.button.clicked.connect(self.close)
        self.retry_button = QPushButton('重试')
        self.retry_button.clicked.connect(self._retry)
        self.retry_button.hide()
        for widget in (self.title_label, self.stage_label, self.progress_bar, self.note_label, self.button):
            layout.addWidget(widget)
        layout.addWidget(self.retry_button)
        self._worker_args = (root, synchronize)
        self._selection = dict(map_id=map_id, difficulty_id=difficulty_id)
        self._create_worker()

    def _create_worker(self):
        self.worker = _Synchronizer(*self._worker_args, self, **self._selection)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._finished)

    def _retry(self):
        if self.worker.isRunning():
            return
        self.worker.deleteLater()
        self._create_worker()
        self.retry_button.hide()
        self.title_label.setText('正在准备地图资源')
        self.progress_bar.show()
        self.stage_label.setText('重新检查资源…')
        self.note_label.setText('素材审核及当前难度识别资源准备完成后，自动打开主窗口')
        self.button.setText('取消启动')
        self.button.setEnabled(True)
        self.worker.start()

    def _update_progress(self, stage, done, total):
        if self.worker.isInterruptionRequested():
            return
        self.stage_label.setText(stage)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _finished(self):
        if self.worker.error is not None:
            self.title_label.setText('启动失败')
            self.stage_label.setText(str(self.worker.error[1]))
            self.progress_bar.hide()
            self.note_label.setText('请检查提示中的素材或配置，可重试或关闭程序。')
            self.button.setText('关闭')
            self.retry_button.show()
        elif self.worker.cancelled:
            self.reject()
        else:
            self.accept()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.stage_label.setText('正在结束启动，请稍候…')
            self.button.setEnabled(False)
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        if self.worker.isRunning():
            self.close()
        else:
            super().reject()


def synchronize_with_progress(root, *, map_id=None, difficulty_id=None):
    from src.asset_sync import synchronize
    from src.paths import LOG_ROOT
    from src.qt_exception_logging import log_unhandled_qt_exceptions

    enable_physical_pixel_coordinates()
    global _application
    _application = QApplication.instance() or QApplication(sys.argv)
    app = _application
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    quit_on_close = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    dialog = StartupDialog(root, synchronize, map_id=map_id, difficulty_id=difficulty_id)
    try:
        with log_unhandled_qt_exceptions(LOG_ROOT/'qt_exception.log'):
            dialog.show()
            dialog.worker.start()
            result = dialog.exec()
            dialog.worker.wait()
        if dialog.worker.error is not None:
            _, error, traceback = dialog.worker.error
            raise error.with_traceback(traceback)
        if dialog.worker.cancelled or result != QDialog.DialogCode.Accepted:
            return None
        return dialog.worker.state
    finally:
        if dialog.worker.isRunning():
            dialog.worker.requestInterruption()
            dialog.worker.wait()
        app.setQuitOnLastWindowClosed(quit_on_close)
        dialog.deleteLater()
