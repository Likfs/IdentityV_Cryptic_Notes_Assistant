from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from src.route_projection_core import (
    AUTO, MapToggleState, capture_bgr, ensure_region_on_screen, escape_key_down,
    floor_control_mouse_down, foreground_game_client, format_projection_metrics,
    game_capture_region, game_window_is_foreground, map_key_down,
    opacity_to_percent, own_window_is_foreground, percent_to_opacity,
)
from src.capture.region import enable_physical_pixel_coordinates
from src.capture.runtime import GlobalActionHotkeys
from PyQt6.QtCore import QPoint, Qt, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPixmap, QTextCursor
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QSlider, QTextEdit, QVBoxLayout,
                             QWidget, QComboBox, QCheckBox, QStackedWidget, QSizeGrip)
from src.route_projection_qt_ui import (STYLE, BG, INK, MUTED, ACCENT, GREEN, AMBER, BAD,
                                        Value, StatusLabel, ImagePanel, SettingsPage, LayoutConfirmationDialog,
                                        ManualLayoutConfirmationDialog,
                                        RouteOverlayWindow, ReferenceMapWindow, NotificationToast, DraggableTitleBar,
                                        exclude_window_from_capture, physical_to_qt_geometry)
from src.projection_workflow import (
    WorkflowResult, complete_projection_preview, process_screenshot, save_workflow_result,
)
from src.version import APP_TITLE, APP_VERSION
import math
from src.paths import CACHE_ROOT, LOG_ROOT, PROJECT_ROOT as ROOT
from src.app_catalog import (
    DEFAULT_SETTINGS_PATH, FLOORS, FLOOR_NAMES, MapCatalog, load_settings, save_settings,
    map_capture_delay_ms, selected_template_variant, TEMPLATE_VERSION_NAMES, load_default_settings,
)
from src.capture.map_keys import GlobalMapKeys
from src.layout_retrieval import create_layout_retriever, BACKEND_LABELS, normalize_recognition_backend
from src.route_projection_pan import MousePanSession
from src.recognition_control import RecognitionCancelled, check_cancelled

LOG_VISIBLE_BLOCK_LIMIT = 2000
LOG_REFRESH_INTERVAL_SECONDS = .1


def recognition_window_identity(titles):
    client = foreground_game_client(titles)
    return [int(ctypes.windll.user32.GetForegroundWindow()), *client.as_list()]




class QtMessageBox:
    @staticmethod
    def confirm_layout(candidates, entrance_root, *, screen_roi=None, is_current=None):
        screen = QGuiApplication.primaryScreen()
        if screen_roi is not None:
            try:
                x, y, width, height, _scale = physical_to_qt_geometry(*map(int, screen_roi))
                screen = QGuiApplication.screenAt(QPoint(x + width // 2, y + height // 2)) or screen
            except (OSError, ValueError):
                pass
        dialog = LayoutConfirmationDialog(candidates, entrance_root, screen=screen)
        if is_current is not None:
            watcher = QTimer(dialog)
            watcher.timeout.connect(lambda: dialog.reject() if not is_current() else None)
            watcher.start(25)
        dialog.show()
        exclude_window_from_capture(dialog)
        bounds = screen.availableGeometry()
        dialog.move(bounds.center() - dialog.rect().center())
        dialog.raise_()
        dialog.activateWindow()
        accepted = dialog.exec() == LayoutConfirmationDialog.DialogCode.Accepted
        choice = dialog.selected_layout if accepted else None
        dialog.deleteLater()
        return choice

    @staticmethod
    def askyesno(title, text, parent=None, *, independent=False, screen_roi=None, entrance_path=None):
        if entrance_path is not None:
            dialog = ManualLayoutConfirmationDialog(title, text, entrance_path, parent)
            dialog.show()
            exclude_window_from_capture(dialog)
            bounds = dialog.screen().availableGeometry()
            dialog.move(bounds.center() - dialog.rect().center())
            dialog.raise_()
            dialog.activateWindow()
            accepted = dialog.exec() == dialog.DialogCode.Accepted
            dialog.deleteLater()
            return accepted
        dialog = QMessageBox(None if independent else parent)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.Yes)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        if independent:
            dialog.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint |
                                  Qt.WindowType.WindowSystemMenuHint)
        else:
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.show()
        exclude_window_from_capture(dialog)
        if independent:
            screen = QGuiApplication.primaryScreen()
            if screen_roi is not None:
                try:
                    x, y, width, height, _scale = physical_to_qt_geometry(*map(int, screen_roi))
                    screen = QGuiApplication.screenAt(QPoint(x + width // 2, y + height // 2)) or screen
                except (OSError, ValueError):
                    pass
            bounds = screen.availableGeometry()
            dialog.move(bounds.center().x() - dialog.width() // 2,
                        bounds.center().y() - dialog.height() // 2)
        dialog.raise_()
        dialog.activateWindow()
        return dialog.exec() == QMessageBox.StandardButton.Yes


messagebox = QtMessageBox


class UiEventBridge(QObject):
    ready = pyqtSignal(object)


class RouteProjectionApp(QWidget):
    def __init__(self, *, settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
        enable_physical_pixel_coordinates()
        super().__init__()
        self.catalog = MapCatalog()
        self.settings_path = Path(settings_path)
        self.settings = load_settings(self.catalog, self.settings_path)
        self.profile = self.catalog.profile(self.settings["map_id"], self.settings["difficulty_id"])
        self.session_id = uuid.uuid4().hex[:12]
        self.results: queue.Queue[tuple] = queue.Queue()
        self._work_bridge = UiEventBridge(self)
        self._work_bridge.ready.connect(self._receive_work_event, Qt.ConnectionType.QueuedConnection)
        self._pending_auto_projection = None
        self.busy = False
        self._active_work = None
        self.closed = False
        self.started_at = time.monotonic()
        self._last_clock_second = -1
        self.map_state = MapToggleState(key_was_down=map_key_down(self.settings["map_key"]))
        self.map_generation = 0
        self.locked_layout: str | None = None
        self.last_decision: dict | None = None
        self.last_result: WorkflowResult | None = None
        self.live_projection = None
        self.reference_context = None
        self.reference_override = None
        self.reference_auto = False
        self.reference_key_was_down = map_key_down(self.settings["reference_hotkey"])
        self.reference_suspended = False
        self._pending_confirmation = None
        self.settings_page: SettingsPage | None = None
        self.preview_images = {}
        self.projection_status_text = "等待地图键"
        self.projection_wait_elapsed_ms: float | None = None
        self.projection_triggered_at: float | None = None
        self.projection_delay_pending = False
        self.hotkeys: GlobalActionHotkeys | None = None
        self.hotkeys_suspended = False
        self.map_var = Value(self.profile.map_name)
        self.difficulty_var = Value(self.profile.difficulty_name)
        self.layout_var = Value(AUTO)
        self.status_var = Value("等待打开游戏地图")
        self.overlay_var = Value("投影已隐藏", self._refresh_status)
        self.hotkey_var = Value("", self._refresh_status)
        self.asset_var = Value("", self._refresh_status)
        self.opacity_var = Value("", self._refresh_status)
        self.metric_var = Value("尚无配准结果", self._refresh_status)
        self.path_var = Value("诊断文件：尚无", self._refresh_status)
        cv2.setNumThreads(4)
        self.setWindowTitle(APP_TITLE)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(1536, 1024)
        self.setMinimumSize(1120, 760)
        self._wallpaper = QPixmap(str(ROOT / 'assets/app_ui/home_wallpaper.jpg'))
        self._wallpaper_scaled = QPixmap()
        self.setStyleSheet(STYLE)
        self._build()
        self.size_grip = QSizeGrip(self)
        self.size_grip.resize(18, 18)
        self.size_grip.raise_()
        self.overlay = RouteOverlayWindow()
        self._motion = MousePanSession(self)
        self.reference_overlay = ReferenceMapWindow()
        self.toast = NotificationToast()
        self._map_key_bridge = UiEventBridge(self)
        self._map_key_bridge.ready.connect(self._receive_map_keys, Qt.ConnectionType.QueuedConnection)
        self.map_keys = GlobalMapKeys(self.settings['map_key'], self.settings['game_window_titles'],
                                     lambda: self._map_key_bridge.ready.emit(None))
        self.apply_settings(self.settings, save=False)
        if self.map_keys is not None:
            self.map_keys.start()
        self.show()
        self.main_capture_excluded = exclude_window_from_capture(self)
        self.after(25, self._tick)
        self.after(400, self._start_layout_retrieval)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_wallpaper") and not self._wallpaper.isNull():
            expanded = self._wallpaper.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                              Qt.TransformationMode.SmoothTransformation)
            x = (expanded.width() - self.width()) // 2
            y = (expanded.height() - self.height()) // 2
            self._wallpaper_scaled = expanded.copy(x, y, self.width(), self.height())
        if hasattr(self, "size_grip"):
            self.size_grip.move(self.width() - 20, self.height() - 20)

    def paintEvent(self, event):
        painter = QPainter(self)
        if hasattr(self, "_wallpaper_scaled") and not self._wallpaper_scaled.isNull():
            painter.drawPixmap(0, 0, self._wallpaper_scaled)
        else:
            painter.fillRect(self.rect(), QColor(BG))
        painter.fillRect(self.rect(), QColor(8, 16, 27, 176))
        painter.end()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_button.setText("□")
            self.size_grip.show()
        else:
            self.showMaximized()
            self.maximize_button.setText("❐")
            self.size_grip.hide()

    def after(self, delay, callback):
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(callback)
        timer.timeout.connect(timer.deleteLater)
        timer.start(delay)
        return timer

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)
        header = DraggableTitleBar(self)
        header.setFixedHeight(64)
        header_layout = QHBoxLayout(header)
        logo = QLabel("◈  地图路线识别 / 投影工具")
        logo.setStyleSheet("font-size: 20px; font-weight: 700; color: #eef5ff; background: transparent;")
        header_layout.addWidget(logo)
        version = QLabel(f"版本 {APP_VERSION}")
        version.setObjectName("muted")
        header_layout.addWidget(version)
        header_layout.addStretch(1)
        self.run_badge = QLabel("●  运行中")
        self.run_badge.setObjectName("good")
        header_layout.addWidget(self.run_badge)
        self.clock_label = QLabel()
        self.clock_label.setObjectName("muted")
        header_layout.addWidget(self.clock_label)
        for symbol, callback in (("—", self.showMinimized), ("□", self.toggle_maximize), ("×", self.close)):
            button = QPushButton(symbol)
            button.setObjectName("windowControl")
            button.setFixedSize(34, 30)
            button.clicked.connect(callback)
            header_layout.addWidget(button)
            if symbol == "□":
                self.maximize_button = button
        root.addWidget(header)

        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.home_page = QWidget()
        self.pages.addWidget(self.home_page)
        home = QVBoxLayout(self.home_page)
        home.setContentsMargins(0, 0, 0, 0)
        home.setSpacing(8)
        body = QHBoxLayout()
        body.setSpacing(8)
        home.addLayout(body, 1)
        left = self._card("地图 / 难度 / 布局设置")
        left.setFixedWidth(315)
        body.addWidget(left)
        self._build_left(left.layout())
        center = self._card("▣  图像预览")
        body.addWidget(center, 1)
        self._build_center(center.layout())
        right = self._card("识别结果 / 实时状态")
        right.setFixedWidth(296)
        body.addWidget(right)
        self._build_right(right.layout())

        logs = self._card("▣  运行日志")
        logs.setFixedHeight(190)
        home.addWidget(logs)
        self._build_logs(logs.layout())

    @staticmethod
    def _card(title):
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(7)
        heading = QLabel(title)
        heading.setObjectName("section")
        layout.addWidget(heading)
        return frame

    @staticmethod
    def _heading(text):
        label = QLabel(text)
        label.setObjectName("subheading")
        return label

    def _build_left(self, layout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        form = QVBoxLayout(content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(7)
        form.addWidget(self._heading("地图"))
        self.map_combo = QComboBox()
        self.map_combo.addItems(self.catalog.map_names.values())
        self.map_combo.setCurrentText(self.profile.map_name)
        self.map_combo.currentTextChanged.connect(lambda text: (self.map_var.set(text), self._map_changed()))
        form.addWidget(self.map_combo)
        form.addWidget(self._heading("难度"))
        self.difficulty_combo = QComboBox()
        self._fill_difficulties()
        self.difficulty_combo.currentTextChanged.connect(lambda text: (self.difficulty_var.set(text), self._profile_changed()))
        form.addWidget(self.difficulty_combo)
        form.addWidget(self._heading("楼层"))
        self.floor_combo = QComboBox()
        for floor in FLOORS:
            self.floor_combo.addItem(FLOOR_NAMES[floor], floor)
        self.floor_combo.setCurrentIndex(FLOORS.index(self.settings["floor"]))
        form.addWidget(self.floor_combo)
        form.addWidget(self._heading("布局识别"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems([AUTO, *self.catalog.layouts(self.profile)])
        self.layout_combo.setToolTip("默认按 F8 自动识别布局。也可直接选择布局，确认后用于后续配准。")
        self.layout_combo.textActivated.connect(self._manual_layout_changed)
        form.addWidget(self.layout_combo)
        form.addWidget(self._heading("地图版本 · 实时切换"))
        self.route_combo = QComboBox()
        self.route_combo.currentTextChanged.connect(self._route_changed)
        form.addWidget(self.route_combo)
        manage = QPushButton("▱  更多设置")
        manage.clicked.connect(self.open_settings)
        form.addWidget(manage)
        form.addWidget(self._heading("投影设置"))
        self.show_overlay_check = QCheckBox("显示投影图")
        self.show_overlay_check.setChecked(True)
        self.show_overlay_check.toggled.connect(self._overlay_enabled_changed)
        form.addWidget(self.show_overlay_check)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("不透明度"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(opacity_to_percent(self.settings["opacity"]))
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        opacity_row.addWidget(self.opacity_slider, 1)
        self.opacity_value = QLabel(f"{self.opacity_slider.value()}%")
        opacity_row.addWidget(self.opacity_value)
        form.addLayout(opacity_row)
        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("左上角路线图"))
        self.reference_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.reference_opacity_slider.setRange(0, 100)
        self.reference_opacity_slider.setValue(opacity_to_percent(self.settings["reference_opacity"]))
        self.reference_opacity_value = QLabel(f"{self.reference_opacity_slider.value()}%")
        self.reference_opacity_slider.valueChanged.connect(self._reference_opacity_changed)
        reference_row.addWidget(self.reference_opacity_slider, 1)
        reference_row.addWidget(self.reference_opacity_value)
        form.addLayout(reference_row)
        self.reset_button = QPushButton("恢复默认投影")
        self.reset_button.clicked.connect(lambda: self.opacity_slider.setValue(
            opacity_to_percent(load_default_settings()['opacity'])))
        form.addWidget(self.reset_button)
        form.addWidget(self._heading("快捷键"))
        self.key_info = QLabel()
        self.key_info.setObjectName("muted")
        self.key_info.setWordWrap(True)
        form.addWidget(self.key_info)
        form.addStretch(1)
        self.start_button = QPushButton("▷  开始识别")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self._schedule_button_action("recognize"))
        layout.addWidget(self.start_button)
        self.recognition_method_label = QLabel('识别方法：图标与结构匹配')
        self.recognition_method_label.setWordWrap(True)
        layout.addWidget(self.recognition_method_label)
        action_row = QHBoxLayout()
        for text, callback in (("手动投影", lambda: self._schedule_button_action("project")),
                               ("隐藏投影", self.hide_overlay), ("复位地图", self.reset_map_state)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            action_row.addWidget(button)
        layout.addLayout(action_row)

    def _build_center(self, layout):
        tabs = QHBoxLayout()
        self.preview_tabs = []
        for index, title in enumerate(("当前截图", "匹配模板", "重叠对比", "最终投影")):
            button = QPushButton(title)
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.clicked.connect(lambda checked, value=index: self._select_preview_tab(value))
            self.preview_tabs.append(button)
            tabs.addWidget(button)
        layout.addLayout(tabs)
        top = QHBoxLayout()
        self.capture_panel = ImagePanel("当前截图区域（游戏画面）", "等待游戏截图")
        self.template_panel = ImagePanel("匹配模板（参考图）", "选择布局后显示底图")
        top.addWidget(self.capture_panel, 1)
        top.addWidget(self.template_panel, 1)
        layout.addLayout(top, 2)
        self.overlay_panel = ImagePanel("重叠效果预览（游戏截图 + 路线投影）", "完成配准后显示投影效果")
        layout.addWidget(self.overlay_panel, 3)
        self._refresh_template()

    def _build_right(self, layout):
        self.current_map_label = StatusLabel()
        self.current_difficulty_label = StatusLabel()
        self.current_layout_label = StatusLabel()
        self.match_label = StatusLabel()
        for label in (self.current_map_label, self.current_difficulty_label, self.current_layout_label, self.match_label):
            layout.addWidget(label)
        layout.addWidget(self._heading("状态信息"))
        self.state_label = StatusLabel("等待打开游戏地图")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.status_rows = {}
        for key, title in (("capture", "截图状态"), ("projection_status", "投影调度"),
                           ("recognition", "识别状态"), ("projection", "投影状态"),
                           ("route", "当前路线"), ("projection_wait_ms", "截图等待耗时"),
                           ("capture_ms", "截图耗时"),
                           ("recognition_ms", "布局识别耗时"), ("registration_ms", "配准耗时"),
                           ("projection_ms", "投影耗时"), ("runtime", "运行时间")):
            row = QHBoxLayout()
            name = QLabel("○  " + title)
            name.setObjectName("muted")
            value = QLabel("—")
            value.setWordWrap(True)
            row.addWidget(name)
            row.addWidget(value, 1)
            layout.addLayout(row)
            self.status_rows[key] = value
        self.decision_label = StatusLabel("尚未识别布局")
        self.decision_label.setWordWrap(True)
        layout.addWidget(self.decision_label)
        self.rank_widgets = []
        self.ranking_heading = self._heading("布局置信度排名")
        layout.addWidget(self.ranking_heading)
        for _ in range(2):
            row = QHBoxLayout()
            name = StatusLabel("—")
            pct = StatusLabel("—")
            row.addWidget(name, 1)
            row.addWidget(pct)
            layout.addLayout(row)
            self.rank_widgets.append((name, pct, None))
        layout.addStretch(1)
        layout.addWidget(self._heading("路线图预览（当前选择）"))
        self.route_panel = ImagePanel("", "尚未选择路线")
        self.route_panel.setFixedHeight(145)
        layout.addWidget(self.route_panel)
        self.metric_label = StatusLabel()
        self.metric_label.setWordWrap(True)
        self.path_label = StatusLabel()
        self.path_label.setWordWrap(True)
        self.roi_label = StatusLabel()
        self.roi_label.setVisible(False)

    def _build_logs(self, layout):
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.log_filter = QComboBox()
        self.log_filter.addItems(["全部日志", "信息", "警告", "错误"])
        self.log_filter.currentTextChanged.connect(self._refresh_logs)
        toolbar.addWidget(self.log_filter)
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self._clear_logs)
        toolbar.addWidget(clear_button)
        export_button = QPushButton("导出")
        export_button.clicked.connect(self._export_logs)
        toolbar.addWidget(export_button)
        layout.addLayout(toolbar)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(LOG_VISIBLE_BLOCK_LIMIT)
        self.log.setToolTip('显示最近 2,000 行；导出包含本次启动后尚未清空的全部日志。')
        self.log.setStyleSheet("background: #121e2a; color: #c6d7e7; border: 1px solid #314a5e; font-family: Consolas;")
        layout.addWidget(self.log, 1)
        self.log_lines = []
        self._log_dirty = False
        self._log_rendered_count = 0
        self._log_rendered_filter = None
        self._log_last_flush_at = 0.
        self._log_total_entries = 0
        self._gui_activity = {}

    def _select_preview_tab(self, index):
        self.selected_preview_tab = index
        self.overlay_panel.heading.setText(("当前截图放大", "匹配模板放大", "重叠对比", "最终投影预览")[index])
        for number, button in enumerate(self.preview_tabs):
            button.setChecked(number == index)
        kind = ("capture", "template", "alignment", "projection")[index]
        value = self.preview_images.get(kind)
        if value is None:
            self.overlay_panel.set_image(None, "该视图暂无图像")
        else:
            self.overlay_panel.set_image(*value)

    def _clear_logs(self):
        self.log_lines.clear()
        self.log.clear()
        self._log_rendered_count = 0
        self._log_dirty = False

    def _refresh_logs(self):
        # Keep log layout off input/result callbacks. The regular GUI tick batches it.
        self._log_dirty = True

    def _flush_logs(self):
        if (not self._log_dirty or not self.log.isVisible() or self.isMinimized()
                or self.busy or self.projection_delay_pending):
            return
        started = time.perf_counter()
        if started - self._log_last_flush_at < LOG_REFRESH_INTERVAL_SECONDS:
            return
        selected = self.log_filter.currentText()
        rebuild = selected != self._log_rendered_filter
        first = 0 if rebuild else self._log_rendered_count
        # Only materialize the visible tail, even after a long hidden/minimized period.
        tail = []
        remaining = LOG_VISIBLE_BLOCK_LIMIT
        for index in range(len(self.log_lines) - 1, first - 1, -1):
            level, line = self.log_lines[index]
            if selected != '全部日志' and selected != level:
                continue
            parts = line.split('\n')
            tail.append('\n'.join(parts[-remaining:]))
            remaining -= len(parts)
            if remaining <= 0:
                break
        text = '\n'.join(reversed(tail))
        if rebuild:
            self.log.setPlainText(text)
        elif tail:
            cursor = QTextCursor(self.log.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(('\n' if not self.log.document().isEmpty() else '') + text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._log_rendered_count = len(self.log_lines)
        self._log_rendered_filter = selected
        self._log_dirty = False
        self._log_last_flush_at = time.perf_counter()
        self._record_gui_duration('log_refresh', started)

    def _record_gui_duration(self, name, started):
        elapsed = (time.perf_counter() - started) * 1000.
        stats = self._gui_activity.setdefault(name, dict(calls=0, total_ms=0., max_ms=0., last_ms=0.))
        stats['calls'] += 1
        stats['total_ms'] += elapsed
        stats['max_ms'] = max(stats['max_ms'], elapsed)
        stats['last_ms'] = elapsed

    def _gui_diagnostic_snapshot(self):
        return dict(log_entries=len(self.log_lines), log_entries_since_start=self._log_total_entries,
                    log_visible_blocks=self.log.document().blockCount(), log_pending=self._log_dirty,
                    activity={name: dict(stats) for name, stats in self._gui_activity.items()})

    def _export_logs(self):
        log_dir = LOG_ROOT
        log_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "导出运行日志", str(log_dir / "route_projection.log"), "文本文件 (*.log *.txt)")
        if path:
            Path(path).write_text("\n".join(line for _level, line in self.log_lines), encoding="utf-8")

    def _fill_difficulties(self):
        choices = [p.difficulty_name for p in self.catalog.difficulties(next(key for key, value in self.catalog.map_names.items() if value == self.map_var.get()))]
        self.difficulty_combo.blockSignals(True)
        self.difficulty_combo.clear()
        self.difficulty_combo.addItems(choices)
        self.difficulty_combo.setCurrentText(self.difficulty_var.get() if self.difficulty_var.get() in choices else choices[0])
        self.difficulty_combo.blockSignals(False)


    def _manual_layout_changed(self, text):
        if (text and text != self.layout_var.get()
                and not getattr(self, '_manual_layout_confirmation', False)):
            self.layout_var.set(text)
            self._layout_changed()

    def _route_changed(self, _label):
        variant_id = self.route_combo.currentData()
        if not variant_id or variant_id == selected_template_variant(self.settings, self.profile.key):
            return
        self.settings.setdefault('template_route_versions', {})[self.profile.key] = variant_id
        save_settings(self.settings, self.settings_path)
        self.map_generation += 1
        self.projection_delay_pending = False
        self._pending_auto_projection = None
        self._pending_template_version_projection = None
        self.reference_context = None
        self.last_result = None
        self.hide_overlay()
        self._clear_alignment_preview()
        self._refresh_reference_map()
        self._update_assets()
        if (self.map_state.is_open and self.locked_layout and not self.closed
                and game_window_is_foreground(self.settings['game_window_titles'])):
            if self.busy:
                self._pending_template_version_projection = (
                    self.map_generation, self.session_id, self.locked_layout, variant_id)
                self._set_status('原图版本已切换，等待旧任务结束后重新截图匹配', ACCENT)
            else:
                self.request_work('project', auto=False)
        else:
            self._set_status('原图版本已切换，下次投影将使用新版本', ACCENT)
        return

    def _opacity_changed(self, value):
        self.opacity_value.setText(f"{value}%")
        self.settings["opacity"] = percent_to_opacity(value)
        save_settings(self.settings, self.settings_path)
        self.opacity_var.set(f"投影不透明度：{value}%")
        self.hide_overlay()

    def _reference_opacity_changed(self, value):
        self.reference_opacity_value.setText(f"{value}%")
        self.settings["reference_opacity"] = percent_to_opacity(value)
        save_settings(self.settings, self.settings_path)
        self.reference_overlay.setWindowOpacity(self.settings["reference_opacity"])

    def _overlay_enabled_changed(self, checked):
        if not checked:
            self.hide_overlay()
        elif self.last_result is not None and self.last_result.accepted and self.last_result.layer is not None:
            self._set_status("显示投影已启用；下次配准后显示路线", ACCENT)

    def _schedule_button_action(self, mode):
        self._set_status("请在 3 秒内切回游戏并打开地图…", AMBER)
        def run():
            if self.closed:
                return
            if game_window_is_foreground(self.settings["game_window_titles"]):
                self.request_work(mode, auto=False)
            else:
                self._set_status("游戏窗口未在前台；请切回游戏后使用快捷键重试", AMBER)
        self.after(3000, run)

    def _refresh_template(self):
        layout = self.layout_var.get()
        if layout == AUTO:
            self.preview_images.pop("template", None)
            self.template_panel.set_image(None, "先识别或手动选择布局")
            return
        try:
            from src.registration.template_matching.direct import load_original
            variant = selected_template_variant(self.settings, self.profile.key)
            image = load_original(self.profile.key, layout, variant)[3]
            title = f"{layout} · {TEMPLATE_VERSION_NAMES[variant]} · 三层完整原图"
            self.template_panel.set_image(image, title)
            self.preview_images['template'] = (image, title)
            return
        except Exception as exc:
            self.preview_images.pop("template", None)
            self.template_panel.set_image(None, str(exc))

    def _refresh_route_preview(self):
        layout = self.layout_var.get()
        if layout == AUTO:
            self.route_panel.set_image(None, "先确认布局")
            return
        try:
            from src.registration.template_matching.direct import load_original
            variant = selected_template_variant(self.settings, self.profile.key)
            self.route_panel.set_image(load_original(self.profile.key, layout, variant)[3],
                                      f"{layout} · {TEMPLATE_VERSION_NAMES[variant]} · 三层完整原图")
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            self.route_panel.set_image(None, f"完整原图不可用：{exc}")
        return

    def _refresh_status(self, *_args):
        if not hasattr(self, "status_rows"):
            return
        started = time.perf_counter()
        self.current_map_label.setText(f"当前地图       {self.profile.map_name}")
        self.current_difficulty_label.setText(f"当前难度       {self.profile.difficulty_name}")
        self.current_layout_label.setText(f"当前布局       {self.locked_layout or '待识别'}")
        score = (self.last_decision or {}).get('score')
        self.match_label.setText(f"匹配分数       {score:.4f}" if score is not None else
                                "匹配分数       —")
        self.status_rows["route"].setText(self.asset_var.get())
        self.status_rows["projection"].setText(self.overlay_var.get())
        self.status_rows["capture"].setText("已截图" if self.last_result is not None else "等待截图")
        self.status_rows["projection_status"].setText(self.projection_status_text)
        self.status_rows["projection_wait_ms"].setText(
            f"{self.projection_wait_elapsed_ms:.0f} ms" if self.projection_wait_elapsed_ms is not None else "—")
        self.status_rows["recognition"].setText("已锁定" if self.locked_layout else "待识别")
        timing = self.last_result.timings_ms if self.last_result is not None else {}
        for key, source in (("capture_ms", "capture"),
                            ("recognition_ms", "layout_retrieval"), ("registration_ms", "registration"),
                            ("projection_ms", "projection")):
            value = timing.get(source)
            self.status_rows[key].setText(f"{value:.0f} ms" if value is not None else "—")
        self.status_rows['projection_wait_ms'].setText(
            f'{self.projection_wait_elapsed_ms:.0f} ms' if self.projection_wait_elapsed_ms is not None else '等待触发')
        self.metric_label.setText(self.metric_var.get())
        self.path_label.setText(self.path_var.get())
        self._record_gui_duration('status_refresh', started)

    def _set_status(self, text: str, color: str = ACCENT):
        self.status_var.set(text)
        self.state_label.configure(text=text, fg=color)
        self._refresh_status()

    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = "错误" if "失败" in text or "错误" in text else "警告" if "未通过" in text or "忽略" in text else "信息"
        self.log_lines.append((level, f"{stamp}   {level:<2}   {text}"))
        self._log_total_entries += 1
        if getattr(self, '_draining_map_keys', False):
            self._map_key_log_dirty = True
        else:
            self._refresh_logs()

    def _map_changed(self, _event=None):
        self._fill_difficulties()
        self.difficulty_var.set(self.difficulty_combo.currentText())
        self._profile_changed()

    def _profile_changed(self, _event=None):
        self.hide_overlay()
        self.map_generation += 1
        self.clear_layout_lock()
        self.profile = next(p for p in self.catalog.profiles.values()
                            if p.map_name == self.map_var.get() and p.difficulty_name == self.difficulty_var.get())
        self.layout_var.set(AUTO)
        self.layout_combo.blockSignals(True)
        self.layout_combo.clear()
        self.layout_combo.addItems([AUTO, *self.catalog.layouts(self.profile)])
        self.layout_combo.blockSignals(False)
        self.settings["map_id"], self.settings["difficulty_id"] = self.profile.map_id, self.profile.difficulty_id
        save_settings(self.settings, self.settings_path)
        self._update_assets()

        self._start_layout_retrieval()


    def _layout_changed(self, _event=None):
        self.map_generation += 1
        self._pending_confirmation = None
        self.session_id = uuid.uuid4().hex[:12]
        self.projection_delay_pending = False
        self.hide_overlay()
        selected = self.layout_var.get()
        if selected == AUTO:
            self.clear_layout_lock()
            self._set_status("已恢复自动识别；请按 F8 识别并确认布局", MUTED)
        else:
            generation, session, profile = self.map_generation, self.session_id, self.profile.key
            self._manual_layout_confirmation = True
            try:
                confirmed = messagebox.askyesno(
                    "确认手动选择布局",
                    f"是否使用手动选择的布局 {selected}？\n\n"
                    "确认后将以此布局进行后续配准，请先确认它与游戏中的实际布局一致。\n"
                    "如不确定，请取消并按 F8 自动识别。",
                    parent=self,
                    entrance_path=(self.profile.entrance_root / f'{selected}.png'
                                   if self.profile.entrance_root else None))
            finally:
                self._manual_layout_confirmation = False
            if self.closed:
                return
            valid = (generation == self.map_generation and session == self.session_id
                     and profile == self.profile.key)
            if confirmed and valid:
                self.last_decision = None
                self.locked_layout = selected
                self.decision_label.configure(text=f"已手动确认 {selected}", fg=ACCENT)
                self._set_status(f"布局已确认：{selected}；打开地图后配准", ACCENT)
            else:
                self.layout_var.set(self.locked_layout or AUTO)
                self._set_status("已取消手动选择，保留原布局" if valid else "地图状态已变化，请重新选择布局", MUTED)
        self._update_assets()

    def _update_assets(self):
        layouts = self.catalog.layouts(self.profile)
        layout = self.layout_var.get()
        self.asset_var.set(f"已收录 {len(layouts)} 个布局")
        if hasattr(self, "route_combo"):
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentText(layout)
            self.layout_combo.blockSignals(False)
            choice_layout = layout if layout != AUTO else (layouts[0] if layouts else "")
            self.route_combo.blockSignals(True)
            from src.registration.template_matching.direct import available_variants
            self.route_combo.clear()
            try:
                choices = available_variants(self.profile.key, choice_layout)
            except (OSError, ValueError, KeyError):
                choices = []
            preferred = selected_template_variant(self.settings, self.profile.key)
            for variant_id in dict.fromkeys([*choices, preferred]):
                self.route_combo.addItem(TEMPLATE_VERSION_NAMES[variant_id], variant_id)
            self.route_combo.setCurrentIndex(self.route_combo.findData(preferred))
            self.asset_var.set(f'{layout} / {TEMPLATE_VERSION_NAMES[preferred]} · 三层完整原图 · 不分楼层')
            self.route_combo.setEnabled(True)
            self.floor_combo.setEnabled(False)
            for index, floor in enumerate(FLOORS):
                self.floor_combo.setItemText(index, '不分楼层')
            self.route_combo.blockSignals(False)
            self._refresh_template()
            self._refresh_route_preview()
        self._refresh_status()


    def _layout_retriever(self):
        cache = getattr(self, 'layout_retrievers', None)
        if cache is None:
            cache = self.layout_retrievers = {}
        backend = self.settings.get('layout_recognition_backend', 'pure_structure')
        key = (self.profile.key, backend)
        if key not in cache:
            cache[key] = create_layout_retriever(backend, self.profile.clean_root,
                cache_root=CACHE_ROOT / 'layout_retrieval' / self.profile.map_id / self.profile.difficulty_id / backend)
        return cache[key]

    def _start_layout_retrieval(self):
        profile_key, retriever = self.profile.key, self._layout_retriever()
        backend = self.settings['layout_recognition_backend']
        def still_current():
            return (not self.closed and self.profile.key == profile_key and
                    self.settings.get('layout_recognition_backend') == backend and
                    getattr(self, 'layout_retrievers', {}).get((profile_key, backend)) is retriever)
        def worker():
            try:
                if not still_current(): return
                fingerprint = retriever.prepare()
                self.results.put(('retrieval_index', profile_key, fingerprint, None, backend, retriever))
            except Exception as exc:
                self.results.put(('retrieval_index', profile_key, None, str(exc), backend, retriever))
            finally:
                if not still_current() and hasattr(retriever, 'close'): retriever.close()
        threading.Thread(target=worker, name='prepare-layout-retrieval', daemon=True).start()


    def apply_settings(self, settings: dict, *, save: bool = True):
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        recognition_backend = normalize_recognition_backend(settings.get('layout_recognition_backend'))
        recognition_changed = self.settings.get('layout_recognition_backend') != recognition_backend
        previous_keys = (self.settings.get("recognize_hotkey"), self.settings.get("project_hotkey"))
        if (self.settings.get("capture_roi") != settings.get("capture_roi")
                or self.settings.get("capture_reference_size") != settings.get("capture_reference_size")):
            self.session_id = uuid.uuid4().hex[:12]
        if save or recognition_changed:
            self.map_generation += 1
            self.projection_delay_pending = False
        settings = {**settings, "registration_backend": "template", "layout_recognition_backend": recognition_backend,
                    "auto_floor": False, "skip_layout_confirmation": False}
        self.settings = settings
        if recognition_changed:
            retired = list(getattr(self, 'layout_retrievers', {}).values())
            self.layout_retrievers = {}
            for retriever in retired:
                if hasattr(retriever, 'close'):
                    threading.Thread(target=retriever.close, name='retire-layout-index', daemon=True).start()
            self._pending_confirmation = None
            self._pending_auto_projection = None
            self.clear_layout_lock()
            self.last_result = None
            self.last_decision = None
            self.reference_context = None
            self.preview_images.clear()
            self.capture_panel.set_image(None, '请重新识别布局')
            self.hide_overlay()
            self._clear_alignment_preview()
            for name, value, _bar in self.rank_widgets:
                name.configure(text='—')
                value.configure(text='—')
            self._start_layout_retrieval()
        self.recognition_method_label.setText('识别方法：' + BACKEND_LABELS[recognition_backend])
        self.ranking_heading.setText('布局匹配分数排名')
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(settings['map_key'], settings['game_window_titles'],
                                    suspended=self.settings_page is not None, discard=save)
        self.map_state.key_was_down = map_key_down(settings["map_key"])
        self.map_state.escape_was_down = escape_key_down()
        self.floor_combo.blockSignals(True)
        self.floor_combo.setCurrentIndex(FLOORS.index(settings["floor"]))
        self.floor_combo.blockSignals(False)
        self._refresh_status()
        self.roi_label.configure(text=f"游戏地图区域  {', '.join(map(str, settings['capture_roi']))}")
        self.hotkey_var.set(f"地图键 {settings['map_key']} · 识别 {settings['recognize_hotkey']} · 投影 {settings['project_hotkey']}")
        self.key_info.setText(self.hotkey_var.get() + f"\n{settings['reference_hotkey']} 显示 / 隐藏左上角路线图\n地图键或 ESC 关闭投影")
        self.reference_key_was_down = map_key_down(settings["reference_hotkey"])
        self.start_button.setText(f"▷  开始识别 ({settings['recognize_hotkey']})")
        percent = opacity_to_percent(float(settings["opacity"]))
        self.opacity_var.set(f"投影不透明度：{percent}%")
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(percent)
        self.opacity_slider.blockSignals(False)
        self.opacity_value.setText(f"{percent}%")
        reference_percent = opacity_to_percent(settings["reference_opacity"])
        self.reference_opacity_slider.blockSignals(True)
        self.reference_opacity_slider.setValue(reference_percent)
        self.reference_opacity_slider.blockSignals(False)
        self.reference_opacity_value.setText(f"{reference_percent}%")
        self.reference_overlay.setWindowOpacity(settings["reference_opacity"])
        self._update_assets()
        if not self.hotkeys_suspended and (self.hotkeys is None or previous_keys != (settings["recognize_hotkey"], settings["project_hotkey"])):
            if self.hotkeys:
                self.hotkeys.stop()
            self.hotkeys = GlobalActionHotkeys(settings["recognize_hotkey"], settings["project_hotkey"])
            self.hotkeys.start()
        if save:
            save_settings(settings, self.settings_path)
            self.hide_overlay()
            self._log("设置已保存并应用")

    def open_settings(self):
        if self.settings_page is not None:
            self.pages.setCurrentWidget(self.settings_page)
            return
        self._handle_map_key()
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'], suspended=True)
        self._pending_confirmation = None
        self.map_generation += 1
        self.projection_delay_pending = False
        self.projection_wait_elapsed_ms = None
        self.projection_triggered_at = None
        self.hide_overlay()
        self.settings["opacity"] = load_settings(self.catalog, self.settings_path)["opacity"]
        self.hotkeys_suspended = True
        if self.hotkeys:
            self.hotkeys.stop()
            self.hotkeys = None
        self.settings_page = SettingsPage(self)
        self.pages.addWidget(self.settings_page)
        self.pages.setCurrentWidget(self.settings_page)

    def _leave_settings(self, page):
        if self.settings_page is not page:
            return
        self.pages.setCurrentWidget(self.home_page)
        self.pages.removeWidget(page)
        self.settings_page = None
        page.deleteLater()
        if not self.closed:
            self.hotkeys_suspended = False
            self.apply_settings(self.settings, save=False)

    def reset_map_state(self):
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'],
                                    suspended=self.settings_page is not None, discard=True)
        self._pending_auto_projection = None
        self.map_state.reset(map_key_down(self.settings["map_key"]))
        self.map_generation += 1
        self.projection_delay_pending = False
        self.clear_layout_lock()
        self.hide_overlay()
        self._set_status("地图状态已复位为关闭", MUTED)
        self._log("地图状态复位为关闭；请确认游戏地图也已关闭")

    def clear_layout_lock(self):
        self.locked_layout = None
        self.last_decision = None
        self.session_id = uuid.uuid4().hex[:12]
        if hasattr(self, "layout_var"):
            self.layout_var.set(AUTO)
        if hasattr(self, "decision_label"):
            self.decision_label.configure(text="布局尚未锁定；请按 F8 识别并确认", fg=MUTED)
        if hasattr(self, "asset_var"):
            self._update_assets()

    def _clear_alignment_preview(self):
        self.preview_images.pop("alignment", None)
        self.preview_images.pop("projection", None)
        self.overlay_panel.set_image(None, "完成配准后显示")

    def _set_preview(self, bgr: np.ndarray, caption: str):
        self._set_preview_kind("capture", bgr, caption)

    def _set_preview_kind(self, kind: str, bgr: np.ndarray, caption: str):
        started = time.perf_counter()
        self.preview_images[kind] = (bgr, caption)
        if kind == "capture":
            self.capture_panel.set_image(bgr, caption)
        if kind == "capture" and getattr(self, "selected_preview_tab", 0) == 0:
            self.overlay_panel.set_image(bgr, caption)
        elif kind in ("alignment", "projection"):
            self._select_preview_tab(2 if kind == "alignment" else 3)
        self._record_gui_duration('preview_update', started)

    def _update_ranking(self, result: WorkflowResult):
        decision = result.decision or self.last_decision
        if result.decision:
            self.last_decision = result.decision
        ranking = decision["ranking"] if decision else []
        for index, (name, pct, _bar) in enumerate(self.rank_widgets):
            item = ranking[index] if index < len(ranking) else None
            name.configure(text=item["layout"] if item else "—", fg=INK if index == 0 else MUTED)
            pct.configure(text=f"{item['score']:.4f}"
                          if item else "—", fg=GREEN if index == 0 else MUTED)
        if self.locked_layout and result.decision is None:
            self.decision_label.configure(text=f"已锁定 {self.locked_layout} · 本次仅配准", fg=GREEN)
        elif decision:
            description = f"待人工确认：{decision['best_label']}"
            detail = f"匹配分数 {decision['score']:.4f} · 分差 {decision['margin']:.4f} · 请核对入口图"
            self.decision_label.configure(text=f"{description} · {detail}", fg=AMBER)
        else:
            text = '没有有效布局候选'
            self.decision_label.configure(text=text, fg=AMBER)

    def request_work(self, mode: str, *, auto: bool) -> None:
        work_requested_at = time.perf_counter()
        gui_at_request = self._gui_diagnostic_snapshot()
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        previous = getattr(self, '_active_work', None) or {}
        restarting = mode == 'recognize' and self.busy and previous.get('mode') == 'recognize'
        if self.busy and not restarting:
            active = '正在识别布局' if previous.get('mode') == 'recognize' else '正在配准路线'
            requested = '识别' if mode == 'recognize' else '配准'
            self._set_status(f'{active}，本次{requested}未启动，请完成后重试', AMBER)
            return
        fixed_layout = self.locked_layout if mode == "project" else None
        if mode == "project" and fixed_layout is None:
            self._set_status("请先按 F8 识别并确认布局，再打开地图投影", AMBER)
            RouteProjectionApp._map_toast(self, "布局尚未确认", "请按 F8 识别并确认布局，然后自动配准路线。", success=False)
            return
        recognition_backend = self.settings.get('layout_recognition_backend', 'icon_structure')
        if mode == "recognize":
            if previous.get('cancel') is not None:
                previous['cancel'].set()
            self._pending_confirmation = None
            self.hide_overlay()
            self._clear_alignment_preview()
            self.session_id = uuid.uuid4().hex[:12]
            self.last_result = self.last_decision = None
            if hasattr(self, 'rank_widgets'):
                for name, score, _bar in self.rank_widgets:
                    name.configure(text='—', fg=INK)
                    score.configure(text='—', fg=MUTED)
                self._refresh_template()
                self._refresh_route_preview()
            if restarting:
                self._log('重复识别：上一请求已作废，立即重新截图，只采用本次识别结果')
        work_id = uuid.uuid4().hex
        cancel = threading.Event()
        self._active_work = dict(id=work_id, mode=mode, cancel=cancel)
        self.busy = True
        template_direct = self.settings.get('registration_backend') == 'template'
        fast_display = mode == 'project' and template_direct and getattr(self, '_work_bridge', None) is not None
        if mode == "project":
            self.hide_overlay()
            self.reference_context = None
            RouteProjectionApp._map_toast(self, "正在配准路线", f"已锁定 {fixed_layout}，" +
                ("正在匹配三层完整原图…"), success=True)
        generation = self.map_generation
        session_id = self.session_id
        profile = self.profile
        roi = list(self.settings["capture_roi"])
        reference_size = list(self.settings["capture_reference_size"])
        window_titles = list(self.settings["game_window_titles"])
        capture_delay_configured = map_capture_delay_ms(self.settings)
        template_variant_id = selected_template_variant(self.settings, profile.key)
        opacity = float(self.settings["opacity"])
        try:
            retriever = self._layout_retriever() if mode == 'recognize' and recognition_backend in BACKEND_LABELS else None
        except Exception as exc:
            self.busy = False
            self._active_work = None
            self._set_status(f'布局识别未启动：{exc}', BAD)
            self._log(f'布局识别初始化失败：{exc}')
            return
        key_started_at = getattr(self, 'projection_triggered_at', None) if auto else None
        self._set_status(('已作废上次识别，正在重新截图…' if restarting else '正在为布局识别截图…')
                         if mode == 'recognize' else '正在为路线配准截图…')
        def worker() -> None:
            delivered = False
            try:
                check_cancelled(cancel.is_set)
                started = time.perf_counter()
                client = foreground_game_client(window_titles)
                capture_window_id = (int(ctypes.windll.user32.GetForegroundWindow())
                                     if template_direct and sys.platform == 'win32' else None)
                recognition_window = recognition_window_identity(window_titles) if retriever is not None else None
                screen_region = game_capture_region(client, roi, reference_size)
                ensure_region_on_screen(screen_region)
                screen_roi = screen_region.as_list()
                screenshot = capture_bgr(screen_roi)
                check_cancelled(cancel.is_set)
                if capture_window_id is not None and int(ctypes.windll.user32.GetForegroundWindow()) != capture_window_id:
                    raise RuntimeError('截图期间游戏窗口发生变化，请重新配准')
                if recognition_window is not None and recognition_window_identity(window_titles) != recognition_window:
                    raise RuntimeError('截图期间游戏窗口发生变化，请重新识别')
                capture_ms = (time.perf_counter() - started) * 1000.0
                bridge = getattr(self, '_work_bridge', None)
                if bridge is not None:
                    bridge.ready.emit(('work_progress', work_id, session_id, generation,
                        '正在识别布局…' if mode == 'recognize' else '正在配准路线，匹配三层完整原图…'))
                result = process_screenshot(
                    screenshot, catalog=self.catalog, profile=profile,
                    layout_override=fixed_layout, opacity=opacity,
                    recognize_only=mode == "recognize",
                    layout_recognition_backend=recognition_backend,
                    template_variant_id=template_variant_id,
                    **({'layout_retriever': retriever,
                        'recognition_reference_size': tuple(roi[2:])} if retriever is not None else {}),
                    **({'recognition_cancelled': cancel.is_set} if mode == 'recognize' else {}),
                    defer_projection_preview=fast_display)
                check_cancelled(cancel.is_set)
                result.timings_ms["capture"] = capture_ms
                result.capture_window_id = capture_window_id
                result.timings_ms['request_setup'] = (thread_dispatched_at-work_requested_at)*1000
                result.timings_ms['worker_start_delay'] = (started-thread_dispatched_at)*1000
                if key_started_at is not None:
                    result.timings_ms['key_to_request'] = (work_requested_at-key_started_at)*1000
                result.gui_diagnostics = {'at_request': gui_at_request,
                                          'capture_delay_configured_ms': capture_delay_configured}
                result.session_id = session_id
                result.recognition_window = recognition_window
                event = ("work", mode, auto, generation, screen_roi, result, None)
                if fast_display:
                    meta = dict(posted_at=time.perf_counter(), requested_at=work_requested_at,
                                key_started_at=key_started_at, session=session_id,
                                ack=threading.Event())
                    self._work_bridge.ready.emit(('ready', event, meta))
                    delivered = True
                    # Give the GUI first use of the ready layer before memory-heavy preview work.
                    # Bounded wait allows shutdown or an unresponsive GUI to finish diagnostics.
                    acknowledged = meta['ack'].wait(1.)
                    ready_result = result
                    result = complete_projection_preview(result)
                    result.timings_ms.update(meta.get('timings', {}))
                    result.gui_diagnostics = {**result.gui_diagnostics, **meta.get('gui_diagnostics', {})}
                    result.timings_ms['preview_finished_from_request'] = (time.perf_counter()-work_requested_at)*1000
                    result.timings_ms['display_acknowledged'] = acknowledged
                    if not self.closed:
                        self._work_bridge.ready.emit(('preview', generation, session_id, ready_result, result))
                else:
                    self.results.put(('recognition_work', work_id, event) if mode == 'recognize' else event)
                if mode == "project":
                    try:
                        saved = save_workflow_result(result)
                        self.results.put(("saved", result, str(saved), None))
                    except Exception as exc:
                        self.results.put(("saved", result, None, str(exc)))
            except RecognitionCancelled:
                return
            except Exception as exc:
                if cancel.is_set():
                    return
                if delivered:
                    if not self.closed:
                        self._work_bridge.ready.emit(('preview_error', generation, session_id, result, str(exc)))
                    return
                event = ("work", mode, auto, generation, roi, None, str(exc))
                if fast_display and not self.closed:
                    self._work_bridge.ready.emit(('ready', event,
                        dict(posted_at=time.perf_counter(), requested_at=work_requested_at,
                             key_started_at=key_started_at, session=session_id, ack=threading.Event())))
                elif not fast_display:
                    self.results.put(('recognition_work', work_id, event) if mode == 'recognize' else event)
            finally:
                if (retriever is not None and hasattr(retriever, 'close') and
                        (self.closed or getattr(self, 'layout_retrievers', {}).get((profile.key, recognition_backend)) is not retriever)):
                    retriever.close()
        thread_dispatched_at = time.perf_counter()
        try:
            threading.Thread(target=worker, name=f"route-{mode}", daemon=True).start()
        except Exception as exc:
            cancel.set()
            self.busy = False
            self._active_work = None
            self._set_status(f'{"识别" if mode == "recognize" else "配准"}任务未启动：{exc}', BAD)

    @pyqtSlot(object)
    def _receive_work_event(self, payload):
        entered_at = time.perf_counter()
        if payload[0] == 'work_progress':
            _, work_id, session, generation, message = payload
            active = getattr(self, '_active_work', None) or {}
            if (not self.closed and self.busy and active.get('id') == work_id
                    and self.session_id == session and self.map_generation == generation):
                self._set_status(message)
            return
        if not self.closed:
            self._handle_map_key()
        if payload[0] == 'preview_error':
            _, generation, session, result, error = payload
            if (not self.closed and generation == self.map_generation and session == self.session_id
                    and self.last_result is result):
                self._log(f'后台预览失败（不影响已显示路线）：{error}')
            return
        if payload[0] == 'preview':
            _, generation, session, ready, completed = payload
            if (self.closed or generation != self.map_generation or session != self.session_id
                    or not self.map_state.is_open or self.last_result is not ready
                    or self.settings.get('registration_backend') != 'template'):
                return
            self.last_result = completed
            if completed.run_directory:
                self.path_var.set(f'诊断文件：{completed.run_directory}')
            if self.live_projection is not None and self.live_projection[0] is ready:
                self.live_projection = (completed, self.live_projection[1], generation)
            if completed.projection is not None:
                self._set_preview_kind('projection', completed.projection,
                                       f'投影预览 · {completed.selected_layout} / 三层完整原图')
            if completed.preview_error:
                self._log(f'后台预览失败（不影响已显示路线）：{completed.preview_error}')
            self.metric_var.set(format_projection_metrics(completed))
            self._log(self.metric_var.get())
            self._refresh_status()
            return
        _, event, meta = payload
        received_at = time.perf_counter()
        presented = False
        try:
            if self.closed:
                return
            RouteProjectionApp._handle_template_floor_click(self)
            _, mode, auto, generation, roi, result, error = event
            valid = (generation == self.map_generation and meta['session'] == self.session_id
                     and self.map_state.is_open and self.settings.get('registration_backend') == 'template'
                     and game_window_is_foreground(self.settings['game_window_titles'])
                     and (result is None or (result.profile_key == self.profile.key
                          and result.selected_layout == self.locked_layout)))
            if not valid:
                self.busy = False
                if getattr(self, 'map_keys', None) is not None:
                    self._log(f'丢弃迟到投影结果：任务代次 {generation}，当前 {self.map_generation}，'
                              f'地图 {"打开" if self.map_state.is_open else "关闭"}，会话 {meta["session"]}')
                return
            # The route layer is independent of screenshot/preview widgets and log rendering.
            feedback = {'result_delivery':(received_at-meta['posted_at'])*1000,
                        'result_queue':(entered_at-meta['posted_at'])*1000,
                        'result_input_handling':(received_at-entered_at)*1000}
            if (result is not None and result.accepted and result.layer is not None
                    and self.show_overlay_check.isChecked()
                    and game_window_is_foreground(self.settings['game_window_titles'])):
                submit_started = time.perf_counter()
                self.overlay.show(result.layer, roi[0], roi[1])
                submitted = time.perf_counter()
                presented = True
                self.live_projection = (result, list(roi), generation)
                if hasattr(self, '_motion'):
                    self._motion.start(result, roi)
                feedback['overlay_submit'] = (submitted-submit_started)*1000
                feedback['request_to_overlay_submit'] = (submitted-meta['requested_at'])*1000
                if meta.get('key_started_at') is not None:
                    feedback['key_to_overlay_submit'] = (submitted-meta['key_started_at'])*1000
            # Publish the feedback atomically, then unblock background preview computation.
            meta['timings'] = feedback
            meta['gui_diagnostics'] = {'at_delivery': self._gui_diagnostic_snapshot()}
            if result is not None:
                # Publish before releasing the worker so the ready UI has full latency metrics.
                result.timings_ms.update(feedback)
                result.gui_diagnostics = {**result.gui_diagnostics, **meta['gui_diagnostics']}
            meta['ack'].set()
            self._apply_result(*event[1:], overlay_presented=presented)
        except Exception as exc:
            self.busy = False
            self.hide_overlay()
            self._set_status(f'投影显示失败：{exc}', BAD)
            self._log(f'投影显示失败：{exc}')
        finally:
            meta['ack'].set()
            RouteProjectionApp._drain_pending_projection(self)

    def _drain_pending_projection(self):
        version_pending = getattr(self, '_pending_template_version_projection', None)
        if version_pending is not None and not self.busy:
            self._pending_template_version_projection = None
            generation, session, layout, variant = version_pending
            if (not self.closed and generation == self.map_generation and session == self.session_id
                    and layout == self.locked_layout and self.settings.get('registration_backend') == 'template'
                    and variant == selected_template_variant(self.settings, self.profile.key)
                    and self.map_state.is_open and game_window_is_foreground(self.settings['game_window_titles'])):
                self.request_work('project', auto=False)
                return
        pending = getattr(self, '_pending_auto_projection', None)
        if pending is None or self.busy:
            return
        self._pending_auto_projection = None
        generation, session, backend, layout = pending
        if (not self.closed and generation == self.map_generation and session == self.session_id
                and backend == self.settings.get('registration_backend', 'template')
                and layout == self.locked_layout):
            RouteProjectionApp._auto_if_open(self, generation)

    def _apply_result(self, mode: str, auto: bool, generation: int, roi: list[int],
                      result: WorkflowResult | None, error: str | None, *, overlay_presented=False) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        self.busy = False
        self._active_work = None
        if mode == 'recognize' and result is not None:
            if getattr(result, 'layout_recognition_backend', 'icon_structure') != getattr(self, 'settings', {}).get('layout_recognition_backend', 'icon_structure'):
                return
            if getattr(result, 'layout_recognition_backend', 'icon_structure') in BACKEND_LABELS:
                if (getattr(self, 'closed', False) or generation != self.map_generation or
                        result.session_id != self.session_id or result.profile_key != self.profile.key):
                    return
            if getattr(result, 'recognition_window', None) is not None:
                try:
                    if recognition_window_identity(self.settings['game_window_titles']) != result.recognition_window:
                        self._set_status('游戏窗口已变化，请重新识别', AMBER)
                        return
                except (OSError, RuntimeError):
                    return
        if mode == 'recognize' and error and generation != self.map_generation:
            return
        if generation != self.map_generation:
            # A worker may have populated tentative hypotheses after a reset.
            # They must not leak into the next map session.
            if mode != "recognize":
                if getattr(self, 'map_keys', None) is not None:
                    self._log(f'丢弃迟到配准结果：任务代次 {generation}，当前 {self.map_generation}')
                return
        if error:
            self._set_status(f"处理失败：{error}", BAD)
            self._log(f"{mode} 失败：{error}")
            if mode == "recognize":
                RouteProjectionApp._show_recognition_toast(
                    self, "布局识别失败", "请保持游戏地图打开并确认游戏窗口在前台，然后按 F8 重试。",
                    success=False, roi=roi)
            else:
                RouteProjectionApp._map_toast(self, "地图处理失败", str(error), success=False, roi=roi)
        elif result:
            self.last_result = result
            self.roi_label.configure(text=f"本次游戏截图  {roi[0]}, {roi[1]} · {roi[2]}×{roi[3]}")
            self._set_preview(result.screenshot, f"本次截图 · {datetime.now():%H:%M:%S} · {result.screenshot.shape[1]}×{result.screenshot.shape[0]}")
            self._update_ranking(result)
            self.path_var.set(f"诊断文件：{result.run_directory or '识别未写入文件'}")
            if mode == "recognize":
                session_id = getattr(result, "session_id", getattr(self, "session_id", None))
                profile_key = getattr(result, "profile_key", getattr(getattr(self, "profile", None), "key", None))
                if not RouteProjectionApp._recognition_can_confirm(self, generation, session_id, profile_key):
                    self._set_status("识别期间地图状态已变化，请按 F8 重新识别", AMBER)
                    RouteProjectionApp._show_recognition_toast(
                        self, "识别结果已失效", "地图画面已变化，请重新打开地图后按 F8 识别。",
                        success=False, roi=roi)
                elif result.accepted and result.selected_layout and result.decision:
                    RouteProjectionApp._show_recognition_toast(
                        self, "布局识别成功", f"首选候选为 {result.selected_layout}，请对照入口图选择布局。",
                        success=True, roi=roi)
                    self._set_status(f"布局识别成功：{result.selected_layout}；等待确认锁定", GREEN)
                    self._pending_confirmation = (generation, result, session_id, profile_key)
                    self.after(0,
                               lambda: RouteProjectionApp._confirm_recognition(self, generation, result, roi))
                else:
                    self._set_status("当前画面暂无法可靠判断布局；请按 F8 重试", AMBER)
                    reason = result.reasons[0] if getattr(result, "reasons", None) else "当前地图画面信息不足"
                    RouteProjectionApp._show_recognition_toast(
                        self, "布局识别失败", f"{reason}。请多探索地图、保持画面稳定，然后按 F8 重试。",
                        success=False, roi=roi)
            else:
                timing = result.timings_ms
                if result.accepted and result.projection is not None:
                    self._set_preview_kind("projection", result.projection,
                                           f"投影预览 · {result.selected_layout} / {FLOOR_NAMES.get(result.floor, '三层完整原图')}")
                self.metric_var.set(format_projection_metrics(result))
                self._log(self.metric_var.get())
                execution = result.registration.validation.get('registration_backend', {}) if result.registration else {}
                if execution:
                    self._log("配准方式：定尺度模板匹配")
                if getattr(self, 'map_keys', None) is not None:
                    self._handle_map_key()
                stale = generation != self.map_generation or not self.map_state.is_open
                if result.accepted and result.layer is not None and not stale:
                    if self.show_overlay_check.isChecked():
                        if not overlay_presented:
                            self.overlay.show(result.layer, roi[0], roi[1])
                        self.live_projection = (result, list(roi), generation)
                        if hasattr(self, '_motion') and not overlay_presented:
                            self._motion.start(result, roi)
                    percent = opacity_to_percent(float(self.settings["opacity"]))
                    self.overlay_var.set(("投影显示中" if self.show_overlay_check.isChecked() else "投影预览完成，显示已关闭") +
                                         f" · {result.selected_layout} / {FLOOR_NAMES.get(result.floor, '三层完整原图')} / "
                                         f"{result.variant_id} · 不透明度 {percent}%")
                    self._set_status("模板匹配候选已投影，请核对位置", AMBER)
                elif stale:
                    self._set_status("地图已关闭，本次结果仅保留诊断", MUTED)
                else:
                    self._set_status("投影未通过校验：" + "；".join(result.reasons[:2]), AMBER)
                if not stale:
                    RouteProjectionApp._finish_map_feedback(self, generation, roi, result)
                self._log(f"{result.selected_layout or '未知布局'} / {result.floor} · "
                          f"{'通过' if result.accepted else '未通过'} · 处理耗时 {timing.get('total', 0):.0f} ms（不含截图与界面等待）")

    def _show_recognition_toast(self, title: str, detail: str, *, success: bool,
                                roi: list[int]) -> None:
        toast = getattr(self, "toast", None)
        if toast is not None:
            toast.show_message(title, detail, success=success, roi=roi)

    def _recognition_can_confirm(self, generation: int, session_id: str | None,
                                 profile_key: str | None) -> bool:
        if getattr(self, "closed", False):
            return False
        if getattr(self, "session_id", None) != session_id:
            return False
        if getattr(getattr(self, "profile", None), "key", None) != profile_key:
            return False
        current = self.map_generation
        return (current == generation or
                (current == generation + 1 and not getattr(getattr(self, "map_state", None), "is_open", True)))

    def _confirm_recognition(self, generation: int, result: WorkflowResult,
                             roi: list[int] | None = None) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        pending = getattr(self, "_pending_confirmation", None)
        if pending is None or pending[0] != generation or pending[1] is not result:
            return
        self._pending_confirmation = None
        if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3]):
            self._set_status("地图状态已变化；请按 F8 重新识别", AMBER)
            return
        decision = result.decision
        candidates = sorted(decision.get("ranking") or [
            {"layout": result.selected_layout, "score": decision["score"]}
        ], key=lambda item: item['score'], reverse=True)[:2]
        selected_layout = messagebox.confirm_layout(
            candidates,
            self.profile.entrance_root if getattr(self, "profile", None) else None,
            screen_roi=roi,
            is_current=lambda: RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3]),
        )
        if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3]):
            if self.session_id == pending[2]:
                self._set_status("地图状态已变化；请按 F8 重新识别", AMBER)
            return
        if selected_layout:
            self.locked_layout = selected_layout
            self.layout_var.set(selected_layout)
            self._update_assets()
            self.decision_label.configure(text=f"已确认锁定 {selected_layout}", fg=ACCENT)
            self._set_status(f"布局已锁定：{selected_layout}；探索门口后按地图键投影", ACCENT)
            self._log(f"F8 识别并确认锁定：{selected_layout}")
        else:
            self.layout_var.set(self.locked_layout or AUTO)
            self._update_assets()
            self.decision_label.configure(text=(f"仍锁定 {self.locked_layout}" if self.locked_layout else
                                                "识别结果未确认，布局尚未锁定"), fg=AMBER)
            self._set_status("已取消锁定；可再次按 F8 识别", MUTED)

    def hide_overlay(self) -> None:
        if hasattr(self, '_motion'):
            self._motion.stop()
        self.live_projection = None
        if hasattr(self, "reference_overlay"):
            self.reference_overlay.hide()
        if hasattr(self, "overlay") and self.overlay.visible:
            self.overlay.hide()
            self.overlay_var.set("投影已隐藏")

    def _map_toast(self, title: str, detail: str, *, success: bool, roi=None) -> None:
        if (getattr(self, "closed", False) or not getattr(getattr(self, "map_state", None), "is_open", False)
                or getattr(self, "settings_page", None) is not None):
            return
        settings = getattr(self, "settings", {})
        if not game_window_is_foreground(settings.get("game_window_titles", [])):
            return
        RouteProjectionApp._show_recognition_toast(self, title, detail, success=success,
                                                   roi=roi or settings.get("capture_roi"))

    def _apply_reference_ready(self, generation, roi, floor, profile_key, layout):
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if (self.closed or generation != self.map_generation or not self.map_state.is_open
                or profile_key != self.profile.key or layout != self.locked_layout
                or (floor not in FLOORS and not (floor == 'all' and self.settings.get('registration_backend') == 'template'))):
            return
        self.reference_context = (generation, list(roi), floor, profile_key, layout)
        self.reference_auto = True
        self._refresh_reference_map()

    def _refresh_reference_map(self):
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        context = self.reference_context
        wanted = self.reference_auto if self.reference_override is None else self.reference_override
        if wanted and context is not None and not game_window_is_foreground(self.settings["game_window_titles"]):
            self.reference_suspended = True
        if (not wanted or context is None or self.closed or not self.map_state.is_open
                or context[0] != self.map_generation or context[3] != self.profile.key
                or context[4] != self.locked_layout or self.settings_page is not None
                or not game_window_is_foreground(self.settings["game_window_titles"])):
            self.reference_overlay.hide()
            return
        _, roi, floor, _, layout = context
        try:
            from src.registration.template_matching.direct import load_original
            variant = selected_template_variant(self.settings, self.profile.key)
            image = load_original(self.profile.key, layout, variant)[3]
            if getattr(self, 'map_keys', None) is not None:
                self._handle_map_key()
                if self.reference_context is not context or context[0] != self.map_generation or not self.map_state.is_open:
                    return
            self.reference_overlay.show_map(image, roi,
                f"{layout} · {TEMPLATE_VERSION_NAMES[variant]} · 三层完整原图 · {self.settings['reference_hotkey']} 隐藏", self.settings['reference_opacity'])
            return
        except (ValueError, OSError, cv2.error, AttributeError) as exc:
            self.reference_overlay.hide()
            self._log(f"左上角路线图不可用：{exc}")

    def _finish_map_feedback(self, generation, roi, result):
        if generation != self.map_generation or not self.map_state.is_open:
            return
        if hasattr(self, "reference_overlay"):
            if result.floor in FLOORS or (result.floor == 'all' and self.settings.get('registration_backend') == 'template'):
                self.reference_context = (generation, list(roi), result.floor, result.profile_key, result.selected_layout)
            else:
                self.reference_context = None
            self.reference_auto = not result.accepted
            self._refresh_reference_map()
        label = f"{result.selected_layout or '布局待确认'} · {FLOOR_NAMES.get(result.floor, '三层完整原图' if result.floor == 'all' else '楼层待确认')}"
        if result.accepted:
            detail = label + ("，路线已投影。" if self.show_overlay_check.isChecked() else "，投影显示已关闭。")
            RouteProjectionApp._map_toast(self, "模板匹配候选已投影", detail, success=True, roi=roi)
        else:
            reason = result.reasons[0] if result.reasons else "可见地图信息不足"
            shown = getattr(getattr(self, "reference_overlay", None), "visible", False)
            detail = label + "：" + reason + (f"。左上角已显示参考路线，{self.settings['reference_hotkey']} 可隐藏。" if shown else "。请继续探索后重试。")
            RouteProjectionApp._map_toast(self, "地图暂未配准", detail, success=False, roi=roi)

    def _handle_reference_key(self):
        down = map_key_down(self.settings["reference_hotkey"])
        pressed = down and not self.reference_key_was_down
        self.reference_key_was_down = down
        if (not self.map_state.is_open or self.settings_page is not None or own_window_is_foreground()
                or not game_window_is_foreground(self.settings["game_window_titles"])):
            if self.reference_overlay.visible:
                self.reference_suspended = True
            self.reference_overlay.hide()
            return
        if self.reference_suspended:
            self.reference_suspended = False
            self._refresh_reference_map()
        if pressed:
            if self.reference_context is not None and self.reference_context[0] != self.map_generation:
                self.reference_context = None
            wanted = self.reference_auto if self.reference_override is None else self.reference_override
            self.reference_override = not wanted
            self._refresh_reference_map()
            if self.reference_override and self.reference_context is None:
                self._map_toast("参考路线待显示", "正在准备完整原图。", success=True)
                if not self.busy:
                    self.request_work("project", auto=False)

    @pyqtSlot(object)
    def _receive_map_keys(self, _wake):
        if not self.closed:
            self._handle_map_key()

    def _handle_map_key(self) -> None:
        listener = getattr(self, 'map_keys', None)
        if listener is not None:
            if getattr(self, '_draining_map_keys', False):
                return
            self._draining_map_keys = True
            try:
                if listener.error:
                    if getattr(self, '_map_key_error', None) != listener.error:
                        self._map_key_error = listener.error
                        self.map_state.is_open = False
                        RouteProjectionApp._apply_map_key_transition(self, 'closed', False, time.perf_counter())
                        self._set_status(listener.error + '；地图键自动投影已停止，请重启程序', BAD)
                        self._log(listener.error)
                    return
                if listener.ready and not getattr(self, '_map_keys_ready_logged', False):
                    self._map_keys_ready_logged = True
                    self._log('地图按键事件监听就绪：Windows Raw Input，G / Esc 独立于界面轮询')
                while True:
                    try:
                        key = listener.events.get_nowait()
                    except queue.Empty:
                        break
                    if key.revision != listener.revision:
                        continue
                    enabled = key.enabled and self.settings_page is None
                    if key.action == 'map':
                        if key.down:
                            self._map_open_press_context = ((key.revision, key.foreground_hwnd)
                                                            if enabled else None)
                        else:
                            # A release in another game window or after reconfiguration
                            # cannot complete the first window's opening gesture.
                            enabled = enabled and getattr(self, '_map_open_press_context', None) == (
                                key.revision, key.foreground_hwnd)
                            self._map_open_press_context = None
                        event = self.map_state.observe(key.down, enabled=enabled)
                        closed_by_escape = False
                    else:
                        pending_escape = (key.down and not self.map_state.escape_was_down
                                          and enabled and self.projection_delay_pending)
                        event = None
                        closed_by_escape = self.map_state.observe_escape(key.down, enabled=enabled) or pending_escape
                        if closed_by_escape:
                            self.map_state.is_open = False
                    if key.down or key.action == 'map':
                        self._log(f"地图按键事件 #{key.sequence}：{key.action} · "
                                  f"{'按下' if key.down else '松开'} · "
                                  f"{'允许' if enabled else '忽略'} · 前台 {key.foreground_title or '无标题'} · "
                                  f"处理等待 {(time.perf_counter()-key.detected_at)*1000:.0f} ms")
                    if event is not None or closed_by_escape:
                        RouteProjectionApp._apply_map_key_transition(self, event, closed_by_escape, key.detected_at)
            finally:
                self._draining_map_keys = False
                if getattr(self, '_map_key_log_dirty', False):
                    self._map_key_log_dirty = False
                    self._refresh_logs()
            return


    def _apply_map_key_transition(self, event, closed_by_escape, key_detected_at):
        self._pending_auto_projection = None
        self.map_generation += 1
        if getattr(self, 'map_keys', None) is not None:
            self._log(f'地图状态转换：{"关闭" if closed_by_escape or event == "closed" else "打开"} · '
                      f'代次 {self.map_generation} · 会话 {getattr(self, "session_id", "无")}')
        self.reference_context = None
        self.reference_override = None
        self.reference_auto = False
        if getattr(self, "toast", None) is not None:
            self.toast.hide()
        self.projection_delay_pending = False
        self.projection_wait_elapsed_ms = None
        self.projection_triggered_at = None
        self.hide_overlay()
        if closed_by_escape or event == "closed":
            self.projection_status_text = "ESC 已关闭" if closed_by_escape else "地图键已关闭"
            self._set_status("地图已关闭，投影已隐藏", MUTED)
            self._log("ESC：地图关闭，投影已隐藏" if closed_by_escape else "地图键：地图关闭，投影已隐藏")
            return
        self.projection_status_text = "等待截图…"
        self.projection_triggered_at = key_detected_at
        self.projection_delay_pending = True
        generation = self.map_generation
        expected_session = getattr(self, "session_id", None)
        capture_delay = map_capture_delay_ms(self.settings)
        delay = max(0, math.ceil(capture_delay - (time.perf_counter()-key_detected_at)*1000))
        self.after(delay, lambda: RouteProjectionApp._project_after_delay(
            self, generation, expected_session))
        action = '匹配三层完整原图'
        self._set_status(f"地图键已松开，等待 {capture_delay} ms 后{action}…")
        self._log(f"地图键 {self.settings['map_key']} 松开：等待 {capture_delay} ms 后{action}")

    def _handle_template_floor_click(self):
        """Invalidate on a floor-control click; rematch after release without naming the floor."""
        # Crossing a button while dragging the map is not a new button press.
        if getattr(getattr(self, '_motion', None), 'map_drag_active', False):
            return
        if (self.settings.get('registration_backend') != 'template' or not self.map_state.is_open
                or self.settings_page is not None or self.closed
                or not game_window_is_foreground(self.settings['game_window_titles'])):
            self._template_floor_click_active = False
            return
        down = floor_control_mouse_down(self.settings['game_window_titles'], self.settings['capture_reference_size'])
        previous = getattr(self, '_template_floor_click_active', False)
        self._template_floor_click_active = down
        if down and not previous:
            self.map_generation += 1
            self.projection_delay_pending = False
            self.reference_context = None
            self.hide_overlay()
        elif previous and not down:
            generation, session = self.map_generation, self.session_id
            self.projection_triggered_at = time.perf_counter()
            self.projection_delay_pending = True
            self._log('定尺度模板匹配：检测到楼层按钮操作，重新匹配完整原图')
            self.after(map_capture_delay_ms(self.settings),
                       lambda: RouteProjectionApp._project_after_delay(self, generation, session))

    def _project_after_delay(self, generation: int, expected_session: str | None = None) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if (generation != self.map_generation or self.closed
                or (expected_session is not None and expected_session != getattr(self, "session_id", None))):
            return
        key_started = getattr(self, 'projection_triggered_at', None)
        capture_delay = map_capture_delay_ms(self.settings)
        if key_started is not None:
            remaining_ms = capture_delay - (time.perf_counter()-key_started)*1000
            if remaining_ms > .001:
                # A coarse Qt timer may fire early; never capture before the deadline.
                self.after(max(1, math.ceil(remaining_ms)),
                           lambda: RouteProjectionApp._project_after_delay(
                               self, generation, expected_session))
                return
        if not self.map_state.is_open or not game_window_is_foreground(self.settings['game_window_titles']):
            self.projection_delay_pending = False
            return
        self.projection_delay_pending = False
        self.projection_status_text = '定尺度模板匹配：匹配三层完整原图'
        self.projection_wait_elapsed_ms = (time.perf_counter()-key_started)*1000 if key_started is not None else None
        self._refresh_status()
        self._log(f'定尺度模板匹配开图：设定等待 {capture_delay} ms，直接截图匹配原图')
        RouteProjectionApp._auto_if_open(self, generation)
        if not self.settings['auto_project']:
            self._set_status('定尺度模板匹配已就绪；自动投影已关闭，可按投影快捷键匹配', MUTED)
        return


    def _auto_if_open(self, generation: int) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if self.map_state.is_open and generation == self.map_generation and not self.closed and self.settings["auto_project"]:
            if game_window_is_foreground(self.settings["game_window_titles"]):
                if self.busy:
                    self._pending_auto_projection = (generation, getattr(self, 'session_id', None),
                        self.settings.get('registration_backend', 'template'), getattr(self, 'locked_layout', None))
                else:
                    self.request_work("project", auto=True)
            else:
                self._set_status("游戏窗口不在前台，本次未截图", AMBER)


    def _tick(self) -> None:
        if self.closed:
            return
        second = int(time.monotonic() - self.started_at)
        if second != self._last_clock_second:
            self._last_clock_second = second
            self.clock_label.setText(datetime.now().strftime("当前时间  %Y-%m-%d  %H:%M:%S"))
            self.status_rows["runtime"].setText(f"{second // 3600:02d}:{(second // 60) % 60:02d}:{second % 60:02d}")
        self._handle_map_key()
        RouteProjectionApp._handle_template_floor_click(self)
        self._handle_reference_key()
        if self.hotkeys:
            while True:
                try:
                    action = self.hotkeys.events.get_nowait()
                except queue.Empty:
                    break
                if self.settings_page is None:
                    if game_window_is_foreground(self.settings["game_window_titles"]):
                        self.request_work("recognize" if action == "recognize" else "project", auto=False)
                    else:
                        self._set_status("快捷键未执行：请切回游戏窗口", AMBER)
            if self.hotkeys.error and "失败" not in self.hotkey_var.get():
                self.hotkey_var.set(self.hotkey_var.get() + f" · {self.hotkeys.error}")
        while True:
            try:
                event = self.results.get_nowait()
            except queue.Empty:
                break
            if event[0] == "saved":
                _, saved_result, directory, error = event
                if saved_result is self.last_result:
                    self.path_var.set(f"诊断文件：{directory}" if directory else f"诊断保存失败：{error}")
                    self._log(self.path_var.get())
            elif event[0] == "reference_ready":
                self._apply_reference_ready(*event[1:])
            elif event[0] == 'retrieval_index':
                if (event[1] == self.profile.key and
                        self.settings.get('layout_recognition_backend', 'icon_structure') == (event[4] if len(event) > 4 else 'icon_structure') and
                        (len(event) < 6 or getattr(self, 'layout_retrievers', {}).get((event[1], event[4])) is event[5])):
                    self._log(f'识别资源准备失败：{event[3]}' if event[3] else f'布局识别资源就绪：{event[2][:12]}')
            elif event[0] == 'recognition_work':
                _, work_id, work = event
                active = getattr(self, '_active_work', None) or {}
                if active.get('id') == work_id and not active['cancel'].is_set():
                    self._apply_result(*work[1:])
                    RouteProjectionApp._drain_pending_projection(self)
            elif event[0] == "work":
                self._apply_result(*event[1:])
                RouteProjectionApp._drain_pending_projection(self)
        self._flush_logs()
        self.after(25, self._tick)

    def _shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        active = getattr(self, '_active_work', None) or {}
        if active.get('cancel') is not None:
            active['cancel'].set()
        for retriever in getattr(self, 'layout_retrievers', {}).values():
            if hasattr(retriever, 'close'):
                threading.Thread(target=retriever.close, name='close-layout-index', daemon=True).start()
        self.layout_retrievers = {}
        self.hide_overlay()
        if hasattr(self._motion, 'shutdown'):
            self._motion.shutdown()
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.stop()
        if self.hotkeys:
            self.hotkeys.stop()
        self.overlay.destroy()
        self.reference_overlay.destroy()
        self.toast.close()

    def closeEvent(self, event):
        page = self.settings_page
        if page is not None and (page._settings_dirty or page._settings_save_job is not None):
            if not page._persist_settings(show_error=True):
                event.ignore()
                return
        self._shutdown()
        event.accept()
        QApplication.instance().quit()
