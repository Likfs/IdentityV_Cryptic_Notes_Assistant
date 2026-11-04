"""Responsive startup resource synchronization before the main window is built."""
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from src.capture.region import enable_physical_pixel_coordinates
from src.version import APP_NAME, APP_VERSION

# Keep Qt alive between the startup dialog and the main window's event loop.
_application = None


class StartupCancelled(Exception):
    pass


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

    def __init__(self, root, synchronize, parent):
        super().__init__(parent)
        self.root = root
        self.synchronize = synchronize
        self.messages = None
        self.error = None
        self.cancelled = False

    def _progress(self, stage, done, total):
        if self.isInterruptionRequested():
            raise StartupCancelled()
        self.progress.emit(stage, done, total)

    def run(self):
        try:
            self.messages = self.synchronize(self.root, progress=self._progress)
            self.cancelled = self.isInterruptionRequested()
        except StartupCancelled:
            self.cancelled = True
        except BaseException:
            self.error = sys.exc_info()


class StartupDialog(QDialog):
    def __init__(self, root, synchronize):
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
        self.note_label = QLabel('资源就绪后将自动打开主窗口')
        self.note_label.setWordWrap(True)
        self.button = QPushButton('取消启动')
        self.button.clicked.connect(self.close)
        for widget in (self.title_label, self.stage_label, self.progress_bar, self.note_label, self.button):
            layout.addWidget(widget)
        self.worker = _Synchronizer(root, synchronize, self)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._finished)

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
            self.note_label.setText('请检查提示中的素材或配置后重新启动。')
            self.button.setText('关闭')
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


def synchronize_with_progress(root):
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
    dialog = StartupDialog(root, synchronize)
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
        return dialog.worker.messages
    finally:
        if dialog.worker.isRunning():
            dialog.worker.requestInterruption()
            dialog.worker.wait()
        app.setQuitOnLastWindowClosed(quit_on_close)
        dialog.deleteLater()
