from __future__ import annotations

import json
import threading
import time

import pytest
from PIL import Image
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtWidgets import QApplication, QPushButton

from tools.collector import collector_ui as ui
from tools.collector import capture_guard
from tools.common.sample_store import SampleWriter
from tools.common.capture_config import CAPTURE_DEFAULTS_PATH


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(capture_guard, "exclude_native_window", lambda _hwnd: True)
    app = QApplication.instance() or QApplication([])
    config = tmp_path / "capture.json"
    config.write_bytes(CAPTURE_DEFAULTS_PATH.read_bytes())
    window = ui.FixedRoiCollectorApp(config, writer=SampleWriter(tmp_path), services=False)
    monkeypatch.setattr(ui, "grab_region", lambda _region: Image.new("RGB", (64, 48), "#1a9bce"))
    window.show()
    app.processEvents()
    yield window
    drain(window)
    window.close()
    app.processEvents()


def drain(window):
    deadline = time.monotonic() + 5
    while window.busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    QApplication.processEvents()
    assert not window.busy


def test_capture_duplicate_archive_and_delete(collector):
    w = collector
    w.capture_and_save_once(); drain(w)
    assert len(w.pending_ids) == w.recent.count() == 1
    w.capture_and_save_once(); drain(w)
    assert w.skipped_count == 1 and w.saved_count == 1
    w.confirm_pending_layout(); drain(w)
    assert not w.pending_ids
    record = w.recent.item(0).data(Qt.ItemDataRole.UserRole)
    assert record["layout_id"] == "layout_01"
    assert (w.writer.project_root / record["roi_image_path"]).exists()
    assert len(list(w.writer.manifest_root.rglob("*.jsonl"))) == 1
    w.delete_recent(w.recent.item(0)); drain(w)
    assert w.deleted_count == 1 and w.saved_count == 0
    assert w.recent.count() == 0
    assert not list(w.writer.roi_root.rglob("*.png"))
    assert not list(w.writer.manifest_root.rglob("*.jsonl"))


def test_clear_recent_keeps_files_and_pending(collector):
    w = collector
    w.capture_and_save_once(); drain(w)
    w.clear_recent()
    assert w.recent.count() == 0 and w.saved_count == 1
    assert len(w.pending_ids) == 1
    assert len(list(w.writer.roi_root.rglob("*.png"))) == 1
    assert not any("脏样本" in button.text() for button in w.findChildren(QPushButton))
    assert "Ctrl+Alt+D" not in [shortcut.key().toString() for shortcut in w.shortcuts]


def test_manual_deletion_before_archive_does_not_consume_number(collector):
    w = collector
    w.capture_and_save_once(); drain(w)
    next(w.writer.roi_root.rglob("*.png")).unlink()
    w.confirm_pending_layout(); drain(w)
    assert w.saved_count == 0 and w.recent.count() == 0
    assert "layout_01" in w.next_label.text()
    assert not list(w.writer.manifest_root.rglob("*.jsonl"))


def test_supplement_existing_layout_keeps_next_number(collector):
    w = collector
    w.capture_and_save_once(); drain(w)
    w.confirm_pending_layout(); drain(w)
    w.capture_and_save_once(); drain(w)
    w.existing_combo.setCurrentIndex(w.existing_combo.findData("layout_01"))
    w.supplement_existing_layout(); drain(w)
    assert w.writer.layout_sample_count(*[c.id for c in w.context()], "layout_01") == 2
    assert "layout_02" in w.next_label.text()
    assert len(next(w.writer.manifest_root.rglob("*.jsonl")).read_text().splitlines()) == 2


def test_busy_capture_cannot_archive_or_switch_context_and_close_waits(collector, monkeypatch):
    w = collector
    release = threading.Event()
    worker_threads = []
    def capture(_region):
        worker_threads.append(threading.get_ident())
        assert release.wait(3)
        return Image.new("RGB", (24, 24))
    monkeypatch.setattr(ui, "grab_region", capture)
    w.start_sampling()
    assert not w.map_combo.isEnabled() and not w.region_button.isEnabled()
    w.confirm_pending_layout()
    assert w.paused
    w.close()
    assert w.closing and w.isVisible()
    release.set(); drain(w)
    assert not w.isVisible()
    assert worker_threads != [threading.get_ident()]
    assert len(list(w.writer.roi_root.rglob("pending/*.png"))) == 1
    assert not list(w.writer.manifest_root.rglob("*.jsonl"))


def test_pause_resume_stop_and_settings(collector):
    w = collector
    w.start_sampling(); drain(w)
    w.toggle_pause()
    assert w.paused and not w.capture_timer.isActive()
    w.toggle_pause()
    assert not w.paused and w.capture_timer.isActive()
    w.stop_sampling()
    assert not w.running and not w.capture_timer.isActive()
    w.interval.setValue(.35)
    w.minimize_check.setChecked(True)
    w.save_settings()
    config = json.loads(w.config_path.read_text(encoding="utf-8"))
    assert config["default_interval_seconds"] == .35
    assert config["minimize_on_start"] is True
    w.preview.actual_size()
    assert w.preview.transform().m11() == 1
    w.preview.zoom(1.2)
    assert w.preview.transform().m11() == pytest.approx(1.2)
    w.preview.fit()
    assert w.preview.fitting


def test_failed_capture_stops_loop_without_phantom_sample(collector, monkeypatch):
    def fail(_region):
        raise ValueError("模拟截图失败")
    monkeypatch.setattr(ui, "grab_region", fail)
    collector.start_sampling(); drain(collector)
    assert not collector.running and not collector.capture_timer.isActive()
    assert collector.saved_count == 0 and not collector.pending_ids
    assert "模拟截图失败" in collector.statusBar().currentMessage()


def test_roi_selection_translates_negative_monitor_coordinates(collector, monkeypatch):
    class Desktop:
        monitors = [{"left": -800, "top": -100, "width": 1600, "height": 900}]
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def grab(self, _region):
            from types import SimpleNamespace
            image = Image.new("RGB", (1600, 900))
            return SimpleNamespace(size=image.size, rgb=image.tobytes())
    monkeypatch.setattr(ui.mss, "mss", Desktop)
    dialog = ui.RegionDialog(ui.CaptureRegion(-700, 0, 200, 300), collector)
    dialog.set_selection(QRect(30, 40, 100, 200))
    assert dialog.selected_region().as_list() == [-770, -60, 100, 200]
    dialog.fields[0].setValue(-1000)
    dialog.validate()
    assert "超出虚拟桌面" in dialog.error.text()
    dialog.fields[0].setValue(-770)
    dialog.validate()
    assert dialog.result() == ui.QDialog.DialogCode.Accepted


def test_local_shortcuts_only_fallback_when_global_registration_fails(collector):
    calls = []
    collector.services = True
    collector.actions["preview"] = lambda: calls.append("preview")
    collector.hotkeys.ready = True
    collector._local_action("preview", "F5")
    assert calls == []
    collector.hotkeys.errors = ["F5 注册失败，可能被其他程序占用"]
    collector._local_action("preview", "F5")
    assert calls == ["preview"]
    collector.services = False


def test_capture_exclusion_covers_popups_and_reapplies_on_show(collector, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QLabel, QMenu, QWidget
    calls = []
    monkeypatch.setattr(capture_guard, "exclude_native_window", lambda hwnd: calls.append(hwnd) or True)
    collector.hide(); collector.show()
    assert int(collector.winId()) in calls
    dialog = QDialog(collector)
    menu = QMenu(collector)
    menu.addAction("永久删除此样本")
    unrelated = QWidget()
    tooltip = QLabel("区域提示", None, Qt.WindowType.ToolTip)
    try:
        dialog.show(); menu.show(); unrelated.show()
        collector.map_combo.showPopup()
        QApplication.processEvents()
        assert int(dialog.winId()) in calls
        assert int(menu.winId()) in calls
        combo_popup = collector.map_combo.view().window()
        assert int(combo_popup.winId()) in calls
        assert int(unrelated.winId()) not in calls
        QApplication.setActiveWindow(dialog)
        tooltip.show(); QApplication.processEvents()
        assert int(tooltip.winId()) in calls
        assert collector.capture_guard.ensure_excluded()
        assert "已启用" in collector.capture_status_label.text()
    finally:
        collector.map_combo.hidePopup()
        tooltip.close(); dialog.close(); menu.close(); unrelated.close()


def test_exclusion_failure_blocks_capture_and_can_recover(collector, monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "grab_region", lambda region: calls.append(region))
    monkeypatch.setattr(capture_guard, "exclude_native_window", lambda _hwnd: False)
    collector.start_sampling()
    assert not collector.running and not collector.busy
    assert not calls and not collector.pending_ids
    assert "未生效" in collector.capture_status_label.text()
    assert "取消截图" in collector.statusBar().currentMessage()
    monkeypatch.setattr(capture_guard, "exclude_native_window", lambda _hwnd: True)
    assert collector.capture_guard.ensure_excluded()
    assert "已启用" in collector.capture_status_label.text()
