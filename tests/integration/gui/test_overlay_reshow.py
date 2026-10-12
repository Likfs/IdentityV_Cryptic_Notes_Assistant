import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtTest import QTest
from src import route_projection_qt_ui as ui


class PaintHistory(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.events = []
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.Paint, QEvent.Type.Hide):
            pixmap = self.window.label.pixmap()
            self.events.append((event.type(), not pixmap.isNull(), pixmap.cacheKey()))
        return False


def layer(x):
    image = np.zeros((60, 100, 4), dtype=np.uint8)
    image[10:30, x:x + 10] = (0, 0, 255, 255)
    return image


def test_hide_paints_empty_surface_before_hiding_and_reshow_uses_new_position(qt_app):
    window = ui.RouteOverlayWindow()
    history = PaintHistory(window)
    try:
        window.show(layer(10), 20, 30)
        QTest.qWait(30)
        old_key = window.label.pixmap().cacheKey()
        assert any(kind == QEvent.Type.Paint and filled for kind, filled, _ in history.events)

        history.events.clear()
        window.hide()
        # This must happen synchronously, before Hide, without pumping the event loop.
        assert history.events[0][:2] == (QEvent.Type.Paint, False)
        assert history.events[-1][:2] == (QEvent.Type.Hide, False)
        assert not window.visible and not window.isVisible()
        assert window.label.pixmap().isNull()
        assert window.grab().toImage().pixelColor(15, 15).alpha() == 0

        history.events.clear()
        window.show(layer(60), 20, 30)
        QTest.qWait(30)
        new_key = window.label.pixmap().cacheKey()
        assert old_key != new_key
        assert any(kind == QEvent.Type.Paint for kind, _, _ in history.events)
        assert all(filled and key == new_key for _, filled, key in history.events)
        image = window.grab().toImage()
        assert image.pixelColor(15, 15).alpha() == 0
        assert image.pixelColor(65, 15).red() == 255
        assert image.pixelColor(65, 15).alpha() == 255
    finally:
        window.close()


def test_native_preparation_precedes_first_show_and_keeps_input_flags(qt_app, monkeypatch):
    window = ui.RouteOverlayWindow()
    states = []
    monkeypatch.setattr(window, "_prepare_native_window", lambda: states.append(window.isVisible()))
    try:
        window.show(layer(10))
        window.hide()
        window.show(layer(60))
        assert states == [False, False]
        assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        assert window.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert window.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    finally:
        window.close()


def test_close_before_first_paint_does_not_restore_overlay(qt_app):
    window = ui.RouteOverlayWindow()
    try:
        for x in (10, 60, 10, 60):
            window.show(layer(x))
            window.hide()
        QTest.qWait(30)
        assert not window.isVisible() and not window.visible
        assert window.label.pixmap().isNull()
    finally:
        window.close()


def test_reference_map_retains_hidden_image_and_opacity(qt_app):
    window = ui.ReferenceMapWindow()
    try:
        window.show_map(np.zeros((60, 100, 3), dtype=np.uint8), [0, 0, 100, 60], "二楼", 0.5)
        old_key = window.label.pixmap().cacheKey()
        window.hide()
        assert window.label.pixmap().cacheKey() == old_key
        assert window.windowOpacity() == pytest.approx(0.5, abs=0.01)
        assert window.caption.text() == "二楼"
    finally:
        window.close()
