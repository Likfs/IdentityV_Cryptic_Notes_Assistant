"""Native workspace contracts: screenshot focus and secondary controls on demand."""
import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton

from src import route_projection_app as app
from src.paths import PROJECT_ROOT
from src.route_projection_qt_ui import ImagePanel
from src import projection_workflow as workflow
from src.capture import visibility
from tests.support.projection_results import result_for


def test_capture_setting_never_creates_diagnostic_window_or_entry(window, monkeypatch):
    monkeypatch.setattr(visibility, 'set_capture_excluded', lambda *_: True)
    assert not window.findChildren(QDialog)
    window.details_section.toggle.click()
    assert not any('诊断图像' in button.text() for button in window.findChildren(QPushButton))
    window.open_settings()
    page = window.settings_page
    for excluded in (False, True, False):
        page.capture_exclusion_check.setChecked(excluded)
        assert page._persist_settings()
        QApplication.processEvents()
        assert window.capture_policy.excluded is excluded
        assert not window.findChildren(QDialog)
        assert not any(widget.windowTitle() == '诊断图像' for widget in QApplication.topLevelWidgets())
        assert not window.overlay.isVisible() and not window.reference_overlay.isVisible()
        assert not window.chest_overlay.isVisible() and not window.toast.isVisible()
    page.close()
    window.start_next_round()
    assert not window.findChildren(QDialog)


@pytest.mark.parametrize('size', [(1280, 800), (1080, 680)])
def test_workspace_renders_one_screenshot_with_secondary_views_folded(window, tmp_path, size):
    image = cv2.imdecode(np.fromfile(PROJECT_ROOT / 'tests/fixtures/registration/layout18_extra_right_room.png',
                                    dtype=np.uint8), cv2.IMREAD_COLOR)
    window.resize(*size)
    window._set_preview(image, '固定测试截图 · 不代表所选布局')
    QApplication.processEvents()
    visible_images = [panel for panel in window.findChildren(ImagePanel) if panel.isVisible()]
    assert visible_images == [window.capture_panel]
    assert not window.display_section.content.isVisible()
    assert not window.details_section.content.isVisible()
    assert not window.log.isVisible()
    assert window.capture_panel.image.width() >= 300
    assert window.capture_panel.image.height() >= 200
    controls = window.floor_selector.parentWidget()
    assert controls.width() <= controls.parentWidget().width(), 'Left controls must not clip horizontally'
    for widget in (window.capture_panel, window.chest_reset_button, window.start_button,
                   window.project_button, window.reset_map_button, window.settings_button):
        point = widget.mapTo(window, QPoint())
        assert window.rect().contains(point)
        assert window.rect().contains(point + QPoint(widget.width() - 1, widget.height() - 1))
    assert window.grab().save(str(tmp_path / 'workspace.png'))


def test_projection_and_opacity_never_replace_or_dim_screenshot_and_round_clears_it(window, direct_inputs):
    screenshot = np.full((60, 80, 3), (41, 87, 129), np.uint8)
    window._set_preview(screenshot, 'current capture')
    before = window.capture_panel._pixmap.toImage()
    window.capture_panel.zoom_in.click()
    assert window.capture_panel.image.zoom == 1.25
    ready = result_for(window, direct_inputs)
    window.last_result = ready
    completed = workflow.complete_projection_preview(ready)
    window._receive_work_event(('preview', window.map_generation, window.session_id, ready, completed))
    assert window.last_result is completed and completed.projection is not None
    window.opacity_slider.setValue(12)
    window.reference_opacity_slider.setValue(75)
    assert window.capture_panel._pixmap.toImage() == before
    assert window.capture_panel.image.zoom == 1.25
    assert window.capture_panel.caption.text() == 'current capture'
    assert window.findChildren(ImagePanel) == [window.capture_panel]
    window.start_next_round()
    assert window.capture_panel._pixmap is None and window.last_result is None
    assert window.capture_panel.image.zoom == 1


def test_segmented_choices_keep_existing_profile_floor_and_button_callbacks(window, monkeypatch):
    scheduled, requests = [], []
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append((delay, callback)))
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: True)
    monkeypatch.setattr(window, 'request_work', lambda mode, *, auto: requests.append((mode, auto)))
    window.project_button.click()
    assert scheduled[-1][0] == 3000
    scheduled[-1][1]()
    assert requests == [('project', False)]
    hard = window.difficulty_combo.findText('困难')
    window.difficulty_selector.buttons[hard].click()
    assert window.profile.difficulty_id == 'hard'
    assert len(window.floor_selector.buttons) == 2
    assert [window.floor_combo.itemData(i) for i in range(2)] == ['2nd', '1st']
    assert window.locked_layout is None and not window.project_button.isEnabled()


def test_settings_tab_switch_ends_recording_and_keeps_shortcuts_on_return(window):
    keys = dict(window.settings)
    window.settings_button.click()
    page = window.settings_page
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == ['识别与显示', '快捷键', '截图设置']
    page.tabs.setCurrentIndex(1)
    button = page.hotkey_buttons['recognize_hotkey']
    button.click()
    assert button.capturing
    page.tabs.setCurrentIndex(2)
    QApplication.processEvents()
    assert not button.capturing
    page.close()
    for field in ('map_key', 'recognize_hotkey', 'project_hotkey', 'reference_hotkey', 'reference_floor_hotkey'):
        assert window.settings[field] == keys[field]


def test_collapsed_logs_catch_up_when_opened(window):
    window._clear_logs()
    window._log('folded message')
    window._flush_logs()
    assert window.log.toPlainText() == '' and window._log_dirty
    window.log_section.toggle.click()
    window._log_last_flush_at = 0
    window._flush_logs()
    assert 'folded message' in window.log.toPlainText()


def test_frontmost_window_warning_is_visible_without_expanding_details(window, monkeypatch):
    scheduled = []
    monkeypatch.setattr(window, 'after', lambda _delay, callback: scheduled.append(callback))
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: False)
    window.start_button.click()
    assert '3 秒' in window.action_hint.text()
    scheduled[-1]()
    assert '游戏窗口未在前台' in window.action_hint.text()
    assert window.action_hint.isVisible() and not window.details_section.content.isVisible()
