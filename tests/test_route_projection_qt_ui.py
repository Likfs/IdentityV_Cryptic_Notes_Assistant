from __future__ import annotations

from types import SimpleNamespace

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

from src import route_projection_app as app_module
from src.app_catalog import MapCatalog, load_settings
from src.capture.runtime import GlobalActionHotkeys
from src import route_projection_qt_ui as ui
from src.version import APP_TITLE, APP_VERSION


def test_qt_window_shows_catalog_assets_and_saves_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(app_module, "exclude_window_from_capture", lambda _widget: False)

    class NoHotkeys:
        events = None
        error = None

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(app_module, "GlobalActionHotkeys", NoHotkeys)
    qt_app = QApplication.instance() or QApplication([])
    path = tmp_path / "settings.json"
    # This test exercises the floor-specific route controls in structure mode.
    path.write_text('{"registration_backend": "cpu"}', encoding="utf-8")
    window = app_module.RouteProjectionApp(settings_path=path)
    try:
        assert window.windowTitle() == APP_TITLE
        assert any(label.text() == f"版本 {APP_VERSION}" for label in window.findChildren(QLabel))
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert window.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
        assert window.windowFlags() & Qt.WindowType.WindowSystemMenuHint
        assert not window._wallpaper.isNull()
        assert window.settings['layout_recognition_backend'] == 'icon_structure'
        window.opacity_slider.setValue(81)
        window.reset_button.click()
        assert window.opacity_slider.value() == 34
        assert load_settings(window.catalog, path)['opacity'] == .34
        window.layout_var.set("layout_01")
        window._update_assets()
        assert window.template_panel._pixmap is not None
        assert window.route_panel._pixmap is not None
        window._select_preview_tab(1)
        assert window.overlay_panel._pixmap is not None

        scheduled = []
        actions = []
        monkeypatch.setattr(window, "after", lambda delay, callback: scheduled.append((delay, callback)))
        monkeypatch.setattr(app_module, "game_window_is_foreground", lambda _titles: True)
        monkeypatch.setattr(window, "request_work", lambda mode, *, auto: actions.append((mode, auto)))
        window.start_button.click()
        assert scheduled[-1][0] == 3000
        scheduled[-1][1]()
        assert actions == [("recognize", False)]

        window.route_combo.setCurrentIndex(window.route_combo.findData("route_zhanshi_fast"))
        assert set(window.settings["route_versions"][window.profile.key].values()) == {"route_zhanshi_fast"}
        window.open_settings()
        dialog = window.settings_dialog
        assert dialog is not None
        assert window.pages.currentWidget() is dialog
        assert not dialog.isWindow()
        assert window.hotkeys_suspended

        assert [dialog.registration_backend_combo.itemText(i) for i in range(3)] == [
            "多尺度结构配准（GPU）", "多尺度结构配准（CPU）", "定尺度模板匹配"]
        assert [dialog.registration_backend_combo.itemData(i) for i in range(3)] == ["auto", "cpu", "template"]
        assert not any("三套地图资产" in label.text() for label in dialog.findChildren(QLabel))
        assert dialog.layout_recognition_combo.currentData() == 'icon_structure'
        assert not dialog.skip_layout_confirmation_check.isEnabled()

        recognize = dialog.hotkey_buttons["recognize_hotkey"]
        recognize.click()
        QTest.keyClick(recognize, Qt.Key.Key_R)
        assert dialog.vars["recognize_hotkey"].get() == "R"
        assert recognize.text() == "R"

        map_button = dialog.hotkey_buttons["map_key"]
        map_button.click()
        QTest.keyClick(map_button, Qt.Key.Key_R)
        assert dialog.vars["map_key"].get() == "G"
        QTest.keyClick(map_button, Qt.Key.Key_Tab)
        assert dialog.vars["map_key"].get() == "Tab"
        map_button.click()
        QTest.keyClick(map_button, Qt.Key.Key_H)
        assert dialog.vars["map_key"].get() == "Tab"  # H belongs to the reference map.
        QTest.keyClick(map_button, Qt.Key.Key_J)
        assert dialog.vars["map_key"].get() == "J"

        project = dialog.hotkey_buttons["project_hotkey"]
        project.click()
        QTest.keyClick(project, Qt.Key.Key_Escape)
        assert dialog.vars["project_hotkey"].get() == "F10"
        project.click()
        QTest.keyClick(project, Qt.Key.Key_F9)
        assert dialog.vars["project_hotkey"].get() == "F9"
        assert dialog.map_capture_delay_spin.value() == 150
        dialog.map_capture_delay_spin.setValue(65)
        dialog.vars["opacity"].set(81)
        assert dialog._persist_settings()
        persisted = load_settings(MapCatalog(), path)
        assert persisted["opacity"] == 0.81
        assert persisted["map_capture_delay_ms"] == 65
        assert window.settings["map_capture_delay_ms"] == 65
        assert persisted["map_key"] == "J"
        assert persisted["recognize_hotkey"] == "R"
        assert persisted["project_hotkey"] == "F9"
        assert set(persisted["route_versions"][window.profile.key]) == set(window.catalog.layouts(window.profile))
        assert set(persisted["route_versions"][window.profile.key].values()) == {"route_zhanshi_fast"}
        assert GlobalActionHotkeys.virtual_key("R") == ord("R")
        dialog.close()
        assert window.pages.currentWidget() is window.home_page
        assert window.settings_dialog is None
        assert not window.hotkeys_suspended

        generation = window.map_generation
        window.route_combo.setCurrentIndex(window.route_combo.findData("route_lhp_solo"))
        assert window.map_generation == generation + 1
        persisted = load_settings(MapCatalog(), path)
        assert persisted["route_versions"][window.profile.key]["layout_01"] == "route_lhp_solo"
        assert persisted["route_versions"][window.profile.key]["layout_02"] == "route_lhp_solo"

        window.last_result = SimpleNamespace(timings_ms={
            "capture": 18.2, "floor_detection": 11.8, "classification": 43.3,
            "registration": 72.4, "projection": 5.1,
        })
        window._refresh_status()
        assert window.status_rows["map_presence_ms"].text() == "—"
        window.map_presence_elapsed_ms = 245.2
        window._refresh_status()
        assert window.status_rows["map_presence_ms"].text() == "245 ms"
        assert window.status_rows["capture_ms"].text() == "18 ms"
        assert window.status_rows["floor_ms"].text() == "12 ms"
        assert window.status_rows["registration_ms"].text() == "72 ms"
        assert window.status_rows["projection_ms"].text() == "5 ms"
        assert "fps" not in window.status_rows
    finally:
        window.close()
        qt_app.processEvents()


def test_recognition_toast_fades_and_does_not_take_focus(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(ui, "exclude_window_from_capture", lambda _widget: False)
    monkeypatch.setattr(ui, "physical_to_qt_geometry", lambda x, y, w, h: (x, y, w, h, 1.0))
    qt_app = QApplication.instance() or QApplication([])
    toast = ui.NotificationToast()
    try:
        toast.show_message("布局识别失败", "请保持地图打开后按 F8 重试。",
                           success=False, roi=[10, 20, 100, 100], hold_ms=80, fade_ms=50)
        qt_app.processEvents()
        assert toast.isVisible()
        assert toast.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        assert toast.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
        assert toast.title.text() == "布局识别失败"
        assert "background: transparent" in toast.panel.styleSheet()
        assert "border: none" in toast.panel.styleSheet()
        screen = qt_app.primaryScreen().geometry()
        assert abs(toast.geometry().center().x() - screen.center().x()) <= 1
        assert toast.y() == screen.top() + round(screen.height() * 0.10)
        assert ui.NotificationToast.HOLD_MS >= 3000
        image = toast.grab().toImage()
        visible_pixels = sum(image.pixelColor(x, y).alpha() > 0
                             for y in range(image.height()) for x in range(image.width()))
        assert visible_pixels > 0  # Text is painted before the fade starts.
        QTest.qWait(180)
        assert not toast.isVisible()
    finally:
        toast.close()
        qt_app.processEvents()


def test_recognition_confirmation_is_topmost(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_app = QApplication.instance() or QApplication([])
    observed = {}
    main_window = QWidget()
    main_window.showMinimized()

    def inspect_dialog(dialog):
        observed["topmost"] = bool(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        observed["independent"] = dialog.parent() is None and dialog.windowType() == Qt.WindowType.Window
        observed["modal"] = dialog.windowModality()
        observed["visible"] = dialog.isVisible()
        dialog.close()
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "exec", inspect_dialog)
    assert app_module.messagebox.askyesno("确认锁定布局", "布局识别成功",
                                          parent=main_window, independent=True)
    assert observed == {"topmost": True, "independent": True,
                        "modal": Qt.WindowModality.ApplicationModal,
                        "visible": True}
    main_window.close()
    qt_app.processEvents()


@pytest.mark.parametrize("scores", [(.765, .234), (.05, .05), (0.0, 0.0)])
def test_candidate_dialog_selection_and_dismissal(monkeypatch, scores):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_app = QApplication.instance() or QApplication([])
    profile = MapCatalog().profile("a_bringer_of_doom", "nightmare")
    candidates = [{"layout": "layout_13", "confidence": scores[0]},
                  {"layout": "layout_07", "confidence": scores[1]}]
    dialog = ui.LayoutConfirmationDialog(candidates, profile.entrance_root)
    try:
        dialog.show()
        qt_app.processEvents()
        assert dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        assert dialog.parent() is None
        assert dialog.windowModality() == Qt.WindowModality.ApplicationModal
        assert dialog.selected_layout == "layout_13"
        assert dialog.cards[0].isChecked()
        # Clicking the image area must select the entire card.
        QTest.mouseClick(dialog.cards[1], Qt.MouseButton.LeftButton,
                         pos=dialog.cards[1].rect().center())
        assert dialog.selected_layout == "layout_07"
        assert not dialog.cards[0].isChecked() and dialog.cards[1].isChecked()
        assert not dialog.cards[0].findChild(ui.QRadioButton).isChecked()
        assert dialog.cards[1].findChild(ui.QRadioButton).isChecked()
        dialog.skip_confirmation.setChecked(True)
        dialog.confirm_button.click()
        assert dialog.result() == ui.LayoutConfirmationDialog.DialogCode.Accepted
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        assert dialog.result() == ui.LayoutConfirmationDialog.DialogCode.Rejected
        dialog.show()
        dialog.close_button.click()
        assert dialog.result() == ui.LayoutConfirmationDialog.DialogCode.Rejected
    finally:
        dialog.close()
        qt_app.processEvents()


def test_candidate_dialog_fits_small_screen_with_missing_images(monkeypatch, tmp_path):
    from PyQt6.QtCore import QRect
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_app = QApplication.instance() or QApplication([])
    bounds = QRect(0, 0, 640, 480)
    dialog = ui.LayoutConfirmationDialog(
        [{"layout": "layout_13", "confidence": .765}, {"layout": "layout_07", "confidence": .234}],
        tmp_path, screen=SimpleNamespace(availableGeometry=lambda: bounds))
    try:
        dialog.show()
        qt_app.processEvents()
        assert dialog.width() <= bounds.width() and dialog.height() <= bounds.height()
        assert dialog.findChild(ui.EntrancePreview).source.isNull()
        assert all(card.height() >= 240 for card in dialog.cards)
        assert dialog.rect().contains(dialog.confirm_button.mapTo(dialog, dialog.confirm_button.rect().center()))
    finally:
        dialog.close()
        qt_app.processEvents()


def test_cancel_confirmation_does_not_return_preference(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_app = QApplication.instance() or QApplication([])
    def cancel(dialog):
        dialog.skip_confirmation.setChecked(True)
        dialog.cancel_button.click()
        return dialog.result()
    monkeypatch.setattr(ui.LayoutConfirmationDialog, "exec", cancel)
    assert app_module.messagebox.confirm_layout([{"layout": "layout_13", "confidence": .9}], None) == (None, False)
    qt_app.processEvents()
