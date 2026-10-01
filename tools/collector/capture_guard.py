"""Cover the collector and its Qt popups when native windows appear/reappear."""
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget

from tools.common.capture_exclusion import exclude_native_window


class CollectorCaptureGuard(QObject):
    changed = pyqtSignal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.applying = False
        QApplication.instance().installEventFilter(self)

    def owns(self, widget):
        ancestor = widget
        while ancestor is not None:
            if ancestor is self.window:
                return True
            ancestor = ancestor.parent()
        # Qt's shared tooltip may have no owner, including inside a modal dialog.
        if widget.windowType() == Qt.WindowType.ToolTip:
            ancestor = QApplication.activeWindow()
            while ancestor is not None:
                if ancestor is self.window:
                    return True
                ancestor = ancestor.parent()
        return False

    def windows(self):
        return [self.window] + [widget for widget in QApplication.topLevelWidgets()
                               if widget is not self.window and widget.isVisible() and self.owns(widget)]

    def protect(self, widget):
        if self.applying:
            return
        self.applying = True
        try:
            widget.setProperty("captureExcluded", exclude_native_window(int(widget.winId())))
        finally:
            self.applying = False
        self.changed.emit()

    def ensure_excluded(self):
        for widget in self.windows():
            self.protect(widget)
        return all(widget.property("captureExcluded") is True for widget in self.windows())

    def eventFilter(self, obj, event):
        if (not self.applying and isinstance(obj, QWidget) and obj.isWindow() and self.owns(obj)
                and (event.type() == QEvent.Type.Show or
                     (event.type() == QEvent.Type.WinIdChange and obj.testAttribute(Qt.WidgetAttribute.WA_WState_Created)))):
            self.protect(obj)
        return False

    def detach(self):
        QApplication.instance().removeEventFilter(self)
