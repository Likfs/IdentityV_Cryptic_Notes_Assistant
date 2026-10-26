from __future__ import annotations

import ctypes
from ctypes import wintypes
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
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QTextCursor
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QSlider, QTextEdit, QVBoxLayout,
                             QWidget, QComboBox, QCheckBox, QStackedWidget, QSizeGrip,
                             QSizePolicy)
from src.route_projection_qt_ui import (STYLE, BG, INK, MUTED, ACCENT, GREEN, AMBER, BAD,
                                        Value, StatusLabel, ScreenshotPanel, SettingsPage, LayoutConfirmationDialog,
                                        CollapsibleSection, SegmentedSelector,
                                        ManualLayoutConfirmationDialog,
                                        RouteOverlayWindow, ReferenceMapWindow, ChestCounterWindow, NotificationToast, DraggableTitleBar,
                                        exclude_window_from_capture, physical_to_qt_geometry)
from src.projection_workflow import (
    WorkflowResult, complete_projection_preview, process_screenshot, save_workflow_result,
)
from src.version import APP_TITLE, APP_VERSION
import math
from src.paths import CACHE_ROOT, LOG_ROOT
from src.app_catalog import (
    DEFAULT_SETTINGS_PATH, FLOOR_NAMES, MapCatalog, load_settings, save_settings,
    map_capture_delay_ms, selected_template_variant, TEMPLATE_VERSION_NAMES, load_default_settings,
)
from src.capture.map_keys import GlobalMapKeys
from src.layout_retrieval import create_layout_retriever, BACKEND_LABELS, normalize_recognition_backend
from src.route_projection_pan import MousePanSession
from src.recognition_control import RecognitionCancelled, RecognitionRequest, check_cancelled
from src.chest_counter import ChestCounter, CHEST_ACTIONS
from src.capture.visibility import CaptureVisibilityPolicy, flush_desktop

LOG_VISIBLE_BLOCK_LIMIT = 2000
LOG_REFRESH_INTERVAL_SECONDS = .1


def recognition_window_identity(titles):
    client = foreground_game_client(titles)
    return [int(ctypes.windll.user32.GetForegroundWindow()), *client.as_list()]


def recognition_window_matches(identity, titles, *, allow_app=False):
    """Validate the captured game even when the user clicks our confirm button."""
    try:
        if game_window_is_foreground(titles):
            return recognition_window_identity(titles) == identity
        if not allow_app or not own_window_is_foreground() or sys.platform != 'win32':
            return False
        if len(identity) != 5:
            return False
        user32 = ctypes.windll.user32
        hwnd = wintypes.HWND(identity[0])
        if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
            return False
        title = ctypes.create_unicode_buffer(1024)
        user32.GetWindowTextW(hwnd, title, len(title))
        if not any(token.casefold() in title.value.casefold() for token in titles):
            return False
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return False
        return list(identity[1:]) == [origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top]
    except (OSError, RuntimeError, TypeError, ValueError):
        return False




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
        watcher = None
        if is_current is not None:
            watcher = QTimer(dialog)
            def reject_stale_confirmation():
                if not is_current():
                    dialog.reject()
            watcher.timeout.connect(reject_stale_confirmation)
            watcher.start(25)
        dialog.show()
        exclude_window_from_capture(dialog)
        bounds = screen.availableGeometry()
        dialog.move(bounds.center() - dialog.rect().center())
        dialog.raise_()
        dialog.activateWindow()
        try:
            accepted = dialog.exec() == LayoutConfirmationDialog.DialogCode.Accepted
            return dialog.selected_layout if accepted else None
        finally:
            if watcher is not None:
                watcher.stop()
                watcher.timeout.disconnect()
            dialog.deleteLater()

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
        self.chest_counter = ChestCounter()
        self.capture_policy = CaptureVisibilityPolicy(self)
        self.profile = self.catalog.profile(self.settings["map_id"], self.settings["difficulty_id"])
        self.session_id = uuid.uuid4().hex[:12]
        self.results: queue.Queue[tuple] = queue.Queue()
        self._work_bridge = UiEventBridge(self)
        self._work_bridge.ready.connect(self._receive_work_event, Qt.ConnectionType.QueuedConnection)
        self._pending_auto_projection = None
        self.busy = False
        self._active_work = None
        self.closed = False
        from src.registration.template_matching.prewarm import FloorPrewarmer
        self._template_prewarmer = FloorPrewarmer()
        self.started_at = time.monotonic()
        self._last_clock_second = -1
        self.map_state = MapToggleState(key_was_down=map_key_down(self.settings["map_key"]))
        self.map_generation = 0
        self.locked_layout: str | None = None
        self.last_decision: dict | None = None
        self.last_result: WorkflowResult | None = None
        self.live_projection = None
        self.reference_context = None
        self.reference_visible = False
        self.reference_floor_source = '手动'
        self.reference_revision = 0
        self._reference_identity = None
        self._pending_confirmation = None
        self._pending_confirmation_roi = None
        self._recognition_request = None
        self._confirmation_in_progress = None
        self._confirmation_scheduled = None
        self.settings_page: SettingsPage | None = None
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
        bounds = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1340, bounds.width() - 40), min(860, bounds.height() - 40))
        self.setMinimumSize(1080, 680)
        self.setStyleSheet(STYLE)
        self._build()
        self.capture_policy.status_changed.connect(self._capture_status_changed)
        self.size_grip = QSizeGrip(self)
        self.size_grip.resize(18, 18)
        self.size_grip.raise_()
        self.overlay = RouteOverlayWindow()
        self._motion = MousePanSession(self)
        self.reference_overlay = ReferenceMapWindow()
        self.chest_overlay = ChestCounterWindow()
        self.toast = NotificationToast()
        for widget in (self, self.overlay, self.reference_overlay, self.chest_overlay, self.toast):
            self.capture_policy.register(widget)
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
        if hasattr(self, "size_grip"):
            self.size_grip.move(self.width() - 20, self.height() - 20)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
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
        root.setContentsMargins(20, 8, 20, 12)
        root.setSpacing(14)
        header = DraggableTitleBar(self)
        header.setFixedHeight(62)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)
        brand = QVBoxLayout()
        brand.setSpacing(3)
        logo = QLabel("地图路线助手")
        logo.setStyleSheet("font-size: 19px; font-weight: 600;")
        brand.addWidget(logo)
        self.brand_subtitle = QLabel(self.profile.map_name)
        self.brand_subtitle.setObjectName('muted')
        brand.addWidget(self.brand_subtitle)
        header_layout.addLayout(brand)
        version = QLabel(f"版本 {APP_VERSION}")
        version.setObjectName('muted')
        header_layout.addWidget(version)
        header_layout.addStretch(1)
        self.run_badge = QLabel('●  就绪')
        self.run_badge.setObjectName('muted')
        header_layout.addWidget(self.run_badge)
        self.settings_button = QPushButton('设置')
        self.settings_button.setAccessibleName('打开设置')
        self.settings_button.clicked.connect(self.open_settings)
        header_layout.addWidget(self.settings_button)
        for symbol, name, callback in (("—", '最小化', self.showMinimized),
                                       ("□", '最大化或还原', self.toggle_maximize),
                                       ("×", '关闭程序', self.close)):
            button = QPushButton(symbol)
            button.setObjectName('windowControl')
            button.setAccessibleName(name)
            button.setFixedSize(32, 30)
            button.clicked.connect(callback)
            header_layout.addWidget(button)
            if symbol == '□':
                self.maximize_button = button
        root.addWidget(header)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.home_page = QWidget()
        self.pages.addWidget(self.home_page)
        home = QVBoxLayout(self.home_page)
        home.setContentsMargins(0, 0, 0, 0)
        home.setSpacing(12)
        body = QHBoxLayout()
        body.setSpacing(16)
        home.addLayout(body, 1)
        left = self._card('本局地图')
        left.setFixedWidth(224)
        body.addWidget(left)
        self._build_left(left.layout())
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(14)
        body.addWidget(center, 1)
        self._build_center(center_layout)
        right = self._card('本局状态')
        right.setFixedWidth(222)
        body.addWidget(right)
        self._build_right(right.layout())
        self.log_section = CollapsibleSection('运行记录')
        home.addWidget(self.log_section)
        self._build_logs(self.log_section.body)
        footer = QHBoxLayout()
        self.key_info = QLabel()
        self.key_info.setObjectName('muted')
        footer.addWidget(self.key_info, 1)
        self.capture_status_label = QLabel()
        self.capture_status_label.setObjectName('muted')
        footer.addWidget(self.capture_status_label)
        home.addLayout(footer)

    @staticmethod
    def _card(title):
        frame = QFrame()
        frame.setObjectName('card')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        heading = QLabel(title)
        heading.setObjectName('section')
        layout.addWidget(heading)
        return frame

    @staticmethod
    def _heading(text):
        label = QLabel(text)
        label.setObjectName('muted')
        return label

    @staticmethod
    def _separator(layout):
        line = QFrame()
        line.setObjectName('separator')
        layout.addWidget(line)

    def _build_left(self, layout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        form = QVBoxLayout(content)
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(10)
        self.map_combo = QComboBox(content)
        self.map_combo.addItems(self.catalog.map_names.values())
        self.map_combo.setCurrentText(self.profile.map_name)
        self.map_combo.currentTextChanged.connect(lambda text: (self.map_var.set(text), self._map_changed()))
        # A single map is already named in the header. Keep the selector for additional maps.
        if self.map_combo.count() > 1:
            form.addWidget(self._heading('地图'))
            form.addWidget(self.map_combo)
        else:
            self.map_combo.hide()
        form.addWidget(self._heading('难度'))
        self.difficulty_combo = QComboBox(content)
        self._fill_difficulties()
        self.difficulty_combo.currentTextChanged.connect(lambda text: (self.difficulty_var.set(text), self._profile_changed()))
        self.difficulty_combo.hide()
        self.difficulty_selector = SegmentedSelector(self.difficulty_combo)
        form.addWidget(self.difficulty_selector)
        form.addWidget(self._heading('路线版本'))
        self.route_combo = QComboBox()
        self.route_combo.setAccessibleName('路线版本')
        self.route_combo.currentTextChanged.connect(self._route_changed)
        form.addWidget(self.route_combo)
        form.addWidget(self._heading('布局选择'))
        self.layout_combo = QComboBox()
        self.layout_combo.setAccessibleName('布局选择')
        self.layout_combo.addItems([AUTO, *self.catalog.layouts(self.profile)])
        self.layout_combo.setToolTip('默认识别后人工确认，也可手动指定布局；自动选择可在设置中开启。')
        self.layout_combo.textActivated.connect(self._manual_layout_changed)
        form.addWidget(self.layout_combo)
        for combo in (self.map_combo, self.route_combo, self.layout_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(1)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._separator(form)
        form.addWidget(self._heading('参考图'))
        self.floor_combo = QComboBox(content)
        for floor in self.profile.floors:
            self.floor_combo.addItem(FLOOR_NAMES[floor], floor)
        self.floor_combo.setCurrentIndex(self.floor_combo.findData(self.settings['reference_floor']))
        self.floor_combo.currentIndexChanged.connect(self._reference_floor_changed)
        self.floor_combo.hide()
        self.floor_selector = SegmentedSelector(self.floor_combo)
        self.floor_selector.setToolTip('仅选择参考图楼层，投影自动比较有效楼层。')
        form.addWidget(self.floor_selector)
        self.reference_toggle_button = QPushButton('显示参考图')
        self.reference_toggle_button.clicked.connect(self._toggle_reference_map)
        form.addWidget(self.reference_toggle_button)
        self._separator(form)
        self.display_section = CollapsibleSection('显示设置')
        form.addWidget(self.display_section)
        display = self.display_section.body
        self.show_overlay_check = QCheckBox('显示路线投影')
        self.show_overlay_check.setChecked(True)
        self.show_overlay_check.toggled.connect(self._overlay_enabled_changed)
        display.addWidget(self.show_overlay_check)
        for prefix, title, setting, callback in (
                ('opacity', '投影不透明度', 'opacity', self._opacity_changed),
                ('reference_opacity', '参考图不透明度', 'reference_opacity', self._reference_opacity_changed)):
            row = QHBoxLayout()
            row.addWidget(self._heading(title), 1)
            value = QLabel()
            value.setObjectName('muted')
            row.addWidget(value)
            display.addLayout(row)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(opacity_to_percent(self.settings[setting]))
            slider.setAccessibleName(title)
            slider.valueChanged.connect(callback)
            display.addWidget(slider)
            setattr(self, prefix + '_slider', slider)
            setattr(self, prefix + '_value', value)
        self.reset_button = QPushButton('恢复默认投影透明度')
        self.reset_button.clicked.connect(lambda: self.opacity_slider.setValue(
            opacity_to_percent(load_default_settings()['opacity'])))
        display.addWidget(self.reset_button)
        self.hide_overlay_button = QPushButton('隐藏投影')
        self.hide_overlay_button.clicked.connect(self.hide_overlay)
        display.addWidget(self.hide_overlay_button)
        self.reset_map_button = QPushButton('复位地图状态')
        self.reset_map_button.setObjectName('quiet')
        self.reset_map_button.clicked.connect(self.reset_map_state)
        form.addWidget(self.reset_map_button)
        form.addStretch(1)

    def _build_chest_controls(self, layout):
        panel = QFrame()
        panel.setObjectName('card')
        row = QHBoxLayout(panel)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)
        self.chest_count_label = QLabel('本局已开启')
        self.chest_count_label.setObjectName('muted')
        row.addWidget(self.chest_count_label)
        self.chest_action_buttons = {}
        for kind, title in (('gold', '金箱'), ('purple', '紫箱')):
            row.addSpacing(5)
            kind_label = QLabel(title)
            kind_label.setStyleSheet('font-size: 13px; color: ' + ('#e4c48a;' if kind == 'gold' else '#c3acdb;'))
            row.addWidget(kind_label)
            for suffix, symbol in (('subtract', '−'), ('add', '+')):
                action = f'chest_{kind}_{suffix}'
                button = QPushButton(symbol)
                button.setStyleSheet('padding: 0; min-height: 0;')
                button.setFixedSize(27, 27)
                button.setAccessibleName(CHEST_ACTIONS[action][0])
                button.clicked.connect(lambda _checked=False, value=action: self._change_chest_count(value))
                self.chest_action_buttons[action] = button
                row.addWidget(button)
                if suffix == 'subtract':
                    count = QLabel('0')
                    count.setObjectName('counterGold' if kind == 'gold' else 'counterPurple')
                    count.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    count.setMinimumWidth(20)
                    row.addWidget(count)
                    setattr(self, kind + '_count_label', count)
        row.addStretch(1)
        self.chest_undo_button = QPushButton('撤销')
        self.chest_undo_button.setObjectName('quiet')
        self.chest_undo_button.setEnabled(False)
        self.chest_undo_button.clicked.connect(lambda: self._change_chest_count('undo'))
        self.chest_reset_button = QPushButton('下一局')
        self.chest_reset_button.setToolTip('计数清零并重置布局、参考图和投影；撤销仅恢复计数。请先关闭游戏地图。')
        self.chest_reset_button.clicked.connect(self.start_next_round)
        row.addWidget(self.chest_undo_button)
        row.addWidget(self.chest_reset_button)
        layout.addWidget(panel)

    def _build_center(self, layout):
        self._build_chest_controls(layout)
        self.capture_panel = ScreenshotPanel()
        self.capture_panel.setObjectName('subcard')
        self.capture_panel.setMinimumHeight(200)
        layout.addWidget(self.capture_panel, 1)
        actions = QHBoxLayout()
        copy = QVBoxLayout()
        copy.setSpacing(4)
        self.action_title = QLabel('从识别本局布局开始')
        self.action_title.setWordWrap(True)
        self.action_hint = QLabel('打开游戏地图，再使用识别快捷键')
        self.action_hint.setObjectName('muted')
        self.action_hint.setWordWrap(True)
        copy.addWidget(self.action_title)
        copy.addWidget(self.action_hint)
        actions.addLayout(copy, 1)
        self.project_button = QPushButton('重新投影')
        self.project_button.clicked.connect(lambda: self._schedule_button_action('project'))
        actions.addWidget(self.project_button)
        self.start_button = QPushButton('识别布局')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(lambda: self._schedule_button_action('recognize'))
        actions.addWidget(self.start_button)
        layout.addLayout(actions)

    def _build_right(self, layout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        side = QVBoxLayout(content)
        side.setContentsMargins(0, 4, 0, 0)
        side.setSpacing(14)
        self.session_label = StatusLabel('等待识别')
        self.session_label.setObjectName('muted')
        side.addWidget(self.session_label)
        side.addWidget(self._heading('当前布局'))
        self.current_layout_label = StatusLabel('—')
        self.current_layout_label.setObjectName('layoutNumber')
        side.addWidget(self.current_layout_label)
        self.layout_description = QLabel('识别后确认本局布局')
        self.layout_description.setObjectName('muted')
        self.layout_description.setWordWrap(True)
        side.addWidget(self.layout_description)
        self._separator(side)
        self.status_rows = {}
        def status_row(target, key, title):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(self._heading(title))
            value = QLabel('—')
            value.setObjectName('muted')
            value.setWordWrap(True)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(value, 1)
            target.addLayout(row)
            self.status_rows[key] = value
        for key, title in (('projection', '路线投影'), ('reference', '参考图'), ('selection', '布局选择')):
            status_row(side, key, title)
        self._separator(side)
        self.ranking_heading = self._heading('匹配候选')
        side.addWidget(self.ranking_heading)
        self.rank_widgets = []
        for _ in range(2):
            row = QHBoxLayout()
            name, score = StatusLabel('—'), StatusLabel('—')
            name.setStyleSheet('font-size: 13px;')
            score.setStyleSheet('font-size: 13px;')
            row.addWidget(name, 1)
            row.addWidget(score)
            side.addLayout(row)
            self.rank_widgets.append((name, score, None))
        note = self._heading('匹配分数不代表正确概率。')
        note.setWordWrap(True)
        side.addWidget(note)
        self.confirm_layout_button = QPushButton('确认布局')
        self.confirm_layout_button.setObjectName('primary')
        self.confirm_layout_button.setToolTip('主动核对本次识别候选；地图关闭时也可确认。')
        self.confirm_layout_button.clicked.connect(self._confirm_pending_layout)
        self.confirm_layout_button.hide()
        side.addWidget(self.confirm_layout_button)
        self.details_section = CollapsibleSection('匹配详情')
        side.addWidget(self.details_section)
        details = self.details_section.body
        self.current_map_label = StatusLabel()
        self.current_difficulty_label = StatusLabel()
        self.match_label = StatusLabel()
        self.recognition_method_label = QLabel()
        self.recognition_method_label.setWordWrap(True)
        self.decision_label = StatusLabel('尚未识别布局')
        self.decision_label.setWordWrap(True)
        self.state_label = StatusLabel('等待打开游戏地图')
        self.state_label.setWordWrap(True)
        for label in (self.current_map_label, self.current_difficulty_label, self.match_label,
                      self.recognition_method_label, self.decision_label, self.state_label):
            label.setObjectName('muted')
            details.addWidget(label)
        for key, title in (('capture', '截图状态'), ('projection_status', '投影调度'),
                           ('recognition', '识别状态'), ('route', '当前路线'),
                           ('projection_wait_ms', '截图等待'), ('capture_ms', '截图耗时'),
                           ('recognition_ms', '识别耗时'), ('registration_ms', '配准耗时'),
                           ('projection_ms', '投影耗时'), ('runtime', '运行时间')):
            status_row(details, key, title)
        self.clock_label = self._heading('')
        self.metric_label, self.path_label, self.roi_label = StatusLabel(), StatusLabel(), StatusLabel()
        for label in (self.clock_label, self.metric_label, self.path_label, self.roi_label):
            label.setObjectName('muted')
            label.setWordWrap(True)
            details.addWidget(label)
        side.addStretch(1)

    def _build_logs(self, layout):
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.log_filter = QComboBox()
        self.log_filter.addItems(['全部日志', '信息', '警告', '错误'])
        self.log_filter.currentTextChanged.connect(self._refresh_logs)
        toolbar.addWidget(self.log_filter)
        clear_button = QPushButton('清空')
        clear_button.clicked.connect(self._clear_logs)
        toolbar.addWidget(clear_button)
        export_button = QPushButton('导出')
        export_button.clicked.connect(self._export_logs)
        toolbar.addWidget(export_button)
        layout.addLayout(toolbar)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(130)
        self.log.document().setMaximumBlockCount(LOG_VISIBLE_BLOCK_LIMIT)
        self.log.setToolTip('最近 2,000 行；导出包含本次启动后尚未清空的全部日志。')
        self.log.setStyleSheet('background: #1b222a; color: #c6d2dd; border: 1px solid #38424e; border-radius: 6px; font-size: 12px;')
        layout.addWidget(self.log)
        self.log_lines = []
        self._log_dirty = False
        self._log_rendered_count = 0
        self._log_rendered_filter = None
        self._log_last_flush_at = 0.
        self._log_total_entries = 0
        self._gui_activity = {}

    def _capture_status_changed(self, status):
        short = ('防捕获设置失败' if '失败' in status else
                 '允许截图和录屏' if status.startswith('已允许') else status)
        self.capture_status_label.setText(short)
        self.capture_status_label.setToolTip(status)
        self.capture_status_label.setStyleSheet(f'color: {AMBER if "失败" in status else MUTED}; font-size: 12px;')
        page = self.settings_page
        if page is not None:
            page.capture_status_label.setText(status)

    def _change_chest_count(self, action):
        changed = (self.chest_counter.undo() if action == 'undo' else self.chest_counter.reset()
                   if action == 'reset' else self.chest_counter.apply(action))
        counter = self.chest_counter
        self.gold_count_label.setText(str(counter.gold))
        self.purple_count_label.setText(str(counter.purple))
        self.chest_undo_button.setEnabled(counter.can_undo)
        if changed:
            self._log(f'手动宝箱计数：{action} · 金箱 {counter.gold} · 紫箱 {counter.purple}')
        self._refresh_chest_overlay()

    def _refresh_chest_overlay(self):
        if (self.closed or not self.settings.get('chest_counter_visible', True)
                or self.settings_page is not None or QApplication.activeModalWidget() is not None
                or own_window_is_foreground()
                or not game_window_is_foreground(self.settings['game_window_titles'])):
            self.chest_overlay.hide()
            return
        try:
            client = foreground_game_client(self.settings['game_window_titles'])
        except (OSError, ValueError, RuntimeError):
            self.chest_overlay.hide()
            return
        self.chest_overlay.show_counts(self.chest_counter.gold, self.chest_counter.purple, client)

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
        self._invalidate_recognition()
        self.settings.setdefault('template_route_versions', {})[self.profile.key] = variant_id
        save_settings(self.settings, self.settings_path)
        self.map_generation += 1
        self.projection_delay_pending = False
        self._pending_auto_projection = None
        self._pending_template_version_projection = None
        self.last_result = None
        self.hide_overlay()
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
        session = self.session_id
        self._set_status("请在 3 秒内切回游戏并打开地图…", AMBER)
        def run():
            if self.closed or session != self.session_id:
                return
            if game_window_is_foreground(self.settings["game_window_titles"]):
                self.request_work(mode, auto=False)
            else:
                self._set_status("游戏窗口未在前台；请切回游戏后使用快捷键重试", AMBER)
        self.after(3000, run)

    def _refresh_status(self, *_args):
        if not hasattr(self, "status_rows"):
            return
        started = time.perf_counter()
        self.current_map_label.setText(f"当前地图       {self.profile.map_name}")
        self.current_difficulty_label.setText(f"当前难度       {self.profile.difficulty_name}")
        self.current_layout_label.setText(self.locked_layout.removeprefix('layout_') if self.locked_layout else '—')
        self.layout_description.setText(f'{self.locked_layout} · 本局已选定' if self.locked_layout else '识别后确认本局布局')
        self.brand_subtitle.setText(self.profile.map_name)
        self.difficulty_selector.refresh()
        self.floor_selector.refresh()
        self.status_rows['reference'].setText(
            FLOOR_NAMES[self.settings['reference_floor']] + ' · 开启' if self.reference_visible else '未显示')
        self.status_rows['selection'].setText('自动选择' if self.settings.get('auto_select_layout') else '人工确认')
        self.run_badge.setText('●  处理中' if self.busy else '●  就绪')
        active_mode = (self._active_work or {}).get('mode')
        self.session_label.setText(
            ('正在配准' if active_mode == 'project' else '正在识别') if self.busy else
            '本局布局已选定' if self.locked_layout else
            '等待确认布局' if self._pending_confirmation else '等待识别')
        self.session_label.setToolTip(self.status_var.get())
        self.start_button.setEnabled(active_mode != 'project')
        self.project_button.setEnabled(bool(self.locked_layout) and not self.busy)
        pending = self._pending_confirmation is not None
        self.confirm_layout_button.setVisible(pending)
        self.confirm_layout_button.setEnabled(not self.busy and self._confirmation_in_progress is None)
        if pending or self._confirmation_in_progress is not None:
            self.project_button.setEnabled(False)
        self.action_title.setText('处理中…' if self.busy else '布局已选定，可以重新投影' if self.locked_layout else '从识别本局布局开始')
        attention = getattr(self, '_status_color', MUTED) in (AMBER, BAD)
        self.action_hint.setText(self.status_var.get() if attention or self.busy or '3 秒' in self.status_var.get() else
                                 '地图缩到最小后重新截图匹配' if self.locked_layout else
                                 f'打开游戏地图，再按 {self.settings["recognize_hotkey"]}')
        self.action_hint.setStyleSheet(f'color: {AMBER if attention else MUTED}; font-size: 12px;')
        self.action_hint.setToolTip(self.status_var.get())
        if pending:
            self.action_title.setText('布局候选已就绪，等待确认')
            self.action_hint.setText('再次打开地图，或点击右侧“确认布局”')
        score = (self.last_decision or {}).get('score')
        self.match_label.setText(f"匹配分数       {score:.4f}" if score is not None else
                                "匹配分数       —")
        self.status_rows["route"].setText(self.asset_var.get())
        self.status_rows['projection'].setText('已显示' if getattr(getattr(self, 'overlay', None), 'visible', False) else '未显示')
        self.status_rows['projection'].setToolTip(self.overlay_var.get())
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
        self._status_color = color
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
        if self.settings['floor'] not in self.profile.floors:
            self.settings['floor'] = self.profile.floors[0]
        if self.settings.get('reference_floor') not in self.profile.floors:
            self.settings['reference_floor'] = self.profile.floors[0]
        save_settings(self.settings, self.settings_path)
        self._update_assets()

        self._start_layout_retrieval()


    def _layout_changed(self, _event=None):
        self._invalidate_recognition()
        self._template_prewarmer.request(None)
        self.map_generation += 1
        self._pending_confirmation = None
        self.session_id = uuid.uuid4().hex[:12]
        self.projection_delay_pending = False
        self.hide_overlay()
        selected = self.layout_var.get()
        if selected == AUTO:
            self.clear_layout_lock()
            self._set_status("已恢复自动识别；请按 F8 识别并选定布局", MUTED)
        else:
            generation, session, profile = self.map_generation, self.session_id, self.profile.key
            self._manual_layout_confirmation = True
            try:
                confirmed = messagebox.askyesno(
                    "确认手动选择布局",
                    f"是否使用手动选择的布局 {selected}？\n\n"
                    "确认后将以此布局进行后续配准，请先确认它与游戏中的实际布局一致。\n"
                    "如不确定，请取消并按 F8 自动识别。" +
                    ("\n入口图待补，当前展示二楼底图。" if not self.profile.entrance_root else ""),
                    parent=self,
                    entrance_path=(self.profile.entrance_root / f'{selected}.png'
                                   if self.profile.entrance_root else
                                   self.catalog.clean_template(self.profile, selected, self.profile.floors[0])))
            finally:
                self._manual_layout_confirmation = False
            if self.closed or session != self.session_id:
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
            choices = [v for v in choices if v in self.profile.template_versions]
            if preferred not in self.profile.template_versions or (choices and preferred not in choices):
                preferred = next((v for v in self.profile.template_versions if v in choices), self.profile.template_versions[0])
                self.settings.setdefault('template_route_versions', {})[self.profile.key] = preferred
            for variant_id in dict.fromkeys([*choices, preferred]):
                self.route_combo.addItem(TEMPLATE_VERSION_NAMES[variant_id], variant_id)
            self.route_combo.setCurrentIndex(self.route_combo.findData(preferred))
            self.asset_var.set(f'{layout} / {TEMPLATE_VERSION_NAMES[preferred]} · 单层图片 · 自动匹配楼层')
            self.route_combo.setEnabled(True)
            self.floor_combo.setEnabled(True)
            self.floor_combo.blockSignals(True)
            self.floor_combo.clear()
            for floor in self.profile.floors:
                self.floor_combo.addItem(FLOOR_NAMES[floor], floor)
            if self.settings.get('reference_floor') not in self.profile.floors:
                self.settings['reference_floor'] = self.profile.floors[0]
            self.floor_combo.setCurrentIndex(self.floor_combo.findData(self.settings['reference_floor']))
            self.floor_combo.blockSignals(False)
            self.route_combo.blockSignals(False)
        self._refresh_status()
        self._prepare_template_sources()
        if hasattr(self, 'reference_overlay'):
            self._refresh_reference_map()


    def _prepare_template_sources(self):
        prewarmer = getattr(self, '_template_prewarmer', None)
        if prewarmer is None:
            return
        layout = getattr(self, 'locked_layout', None)
        key = ((self.session_id, self.profile.key, layout,
                selected_template_variant(self.settings, self.profile.key))
               if layout and not self.closed else None)
        prewarmer.request(key)

    def _layout_retriever(self):
        cache = getattr(self, 'layout_retrievers', None)
        if cache is None:
            cache = self.layout_retrievers = {}
        backend = self.settings.get('layout_recognition_backend', 'pure_structure')
        key = (self.profile.key, backend)
        if key not in cache:
            cache[key] = create_layout_retriever(backend, self.profile.clean_root,
                floors=self.profile.floors,
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
        if save or recognition_changed or any(self.settings.get(field) != settings.get(field) for field in (
                'capture_roi', 'capture_reference_size', 'game_window_titles', 'template_route_versions')):
            self._invalidate_recognition()
        previous_keys = (self.settings.get("recognize_hotkey"), self.settings.get("project_hotkey"))
        if (self.settings.get("capture_roi") != settings.get("capture_roi")
                or self.settings.get("capture_reference_size") != settings.get("capture_reference_size")):
            self.session_id = uuid.uuid4().hex[:12]
        if save or recognition_changed:
            self.map_generation += 1
            self.projection_delay_pending = False
        settings = {**settings, "registration_backend": "template", "layout_recognition_backend": recognition_backend,
                    "auto_floor": False, "skip_layout_confirmation": False,
                    "auto_select_layout": settings.get('auto_select_layout') is True}
        self.settings = settings
        self.capture_policy.set_excluded(settings.get('capture_exclusion_enabled', True))
        self._capture_status_changed(self.capture_policy.status)
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
            self.capture_panel.set_image(None, '请重新识别布局')
            self.hide_overlay()
            for name, value, _bar in self.rank_widgets:
                name.configure(text='—')
                value.configure(text='—')
            self._start_layout_retrieval()
        self.recognition_method_label.setText('识别方法：' + BACKEND_LABELS[recognition_backend])
        self.ranking_heading.setText('匹配候选')
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(settings['map_key'], settings['game_window_titles'],
                                    suspended=self.settings_page is not None, discard=save,
                                    action_keys={**{action: settings[f'{action}_hotkey'] for action in CHEST_ACTIONS},
                                                 'reference_toggle': settings['reference_hotkey'],
                                                 'reference_next_floor': settings['reference_floor_hotkey'],
                                                 'next_round': settings['next_round_hotkey'],
                                                 'recognize': settings['recognize_hotkey'],
                                                 'project': settings['project_hotkey']})
        self.map_state.key_was_down = map_key_down(settings["map_key"])
        self.map_state.escape_was_down = escape_key_down()
        self.floor_combo.blockSignals(True)
        self.floor_combo.setCurrentIndex(self.floor_combo.findData(settings['reference_floor']))
        self.floor_combo.blockSignals(False)
        self._refresh_status()
        self.roi_label.configure(text=f"游戏地图区域  {', '.join(map(str, settings['capture_roi']))}")
        self.hotkey_var.set(f"地图键 {settings['map_key']} · 识别 {settings['recognize_hotkey']} · 投影 {settings['project_hotkey']}")
        self.key_info.setText(f'{settings["map_key"]} 地图开关    {settings["reference_hotkey"]} 参考图    {settings["reference_floor_hotkey"]} 下一层    {settings["next_round_hotkey"]} 下一局    Esc 隐藏投影')
        self.key_info.setToolTip(self.hotkey_var.get() + '\n' + ' · '.join(
            f'{settings[f"{action}_hotkey"]} {label}' for action, (label, _, _, _) in CHEST_ACTIONS.items()))
        for action, (label, _, _, _) in CHEST_ACTIONS.items():
            self.chest_action_buttons[action].setToolTip(f'{label} · {settings[f"{action}_hotkey"]}（游戏前台时生效）')
        self.start_button.setText(f"识别布局  {settings['recognize_hotkey']}")
        self.chest_reset_button.setToolTip(
            f'下一局 · {settings["next_round_hotkey"]}（游戏前台时生效）\n'
            '计数清零并重置布局、参考图和投影；撤销仅恢复计数。请先关闭游戏地图。')
        self.project_button.setText(f"重新投影  {settings['project_hotkey']}")
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
        self._refresh_chest_overlay()
        self._update_assets()
        if self.map_keys is not None:
            if self.hotkeys:
                self.hotkeys.stop()
                self.hotkeys = None
        elif not self.hotkeys_suspended and (self.hotkeys is None or previous_keys != (settings["recognize_hotkey"], settings["project_hotkey"])):
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
        self._invalidate_recognition()
        self._handle_map_key()
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'], suspended=True)
        self._pending_confirmation = None
        self.map_generation += 1
        self.projection_delay_pending = False
        self.projection_wait_elapsed_ms = None
        self.projection_triggered_at = None
        self.hide_overlay()
        self.reference_overlay.hide()
        self.chest_overlay.hide()
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

    def start_next_round(self):
        # Reset even when the counter is already zero. Undo affects counts only.
        self._change_chest_count('reset')
        self.reset_map_state()
        self._set_status('已进入下一局：计数已清零，布局待识别；打开游戏地图后按识别快捷键', MUTED)
        self._log('下一局：宝箱计数清零，布局识别及投影状态已重置')

    def reset_map_state(self):
        self._invalidate_recognition()
        active = self._active_work or {}
        if active.get('cancel') is not None:
            active['cancel'].set()
        self._active_work = None
        self.busy = False
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'],
                                    suspended=self.settings_page is not None, discard=True)
        self._pending_auto_projection = None
        self._pending_template_version_projection = None
        self._pending_confirmation = None
        self.map_state.reset(map_key_down(self.settings["map_key"]))
        self.map_generation += 1
        self.projection_delay_pending = False
        self.projection_triggered_at = None
        self.projection_wait_elapsed_ms = None
        self.projection_status_text = '等待地图键'
        self._template_floor_click_active = False
        self.last_result = None
        self.hide_overlay()
        self.toast.hide()
        if self.hotkeys is not None:
            while True:
                try:
                    self.hotkeys.events.get_nowait()
                except queue.Empty:
                    break
        self.clear_layout_lock()
        self.capture_panel.set_image(None, '等待新一局截图')
        for name, score, _bar in self.rank_widgets:
            name.configure(text='—', fg=INK)
            score.configure(text='—', fg=MUTED)
        self.metric_var.set('尚无配准结果')
        self.path_var.set('诊断文件：尚无')
        self.roi_label.configure(text=f"游戏地图区域  {', '.join(map(str, self.settings['capture_roi']))}")
        self._set_status("地图状态已复位为关闭", MUTED)
        self._log("地图状态复位为关闭；请确认游戏地图也已关闭")

    def clear_layout_lock(self):
        self._invalidate_recognition()
        if hasattr(self, 'reference_overlay'):
            self._reset_reference_map()
        self.locked_layout = None
        self.last_decision = None
        self.session_id = uuid.uuid4().hex[:12]
        if hasattr(self, "layout_var"):
            self.layout_var.set(AUTO)
        if hasattr(self, "decision_label"):
            self.decision_label.configure(text="布局尚未锁定；请按 F8 识别并选定", fg=MUTED)
        if hasattr(self, "asset_var"):
            self._update_assets()

    def _set_preview(self, bgr: np.ndarray, caption: str):
        started = time.perf_counter()
        self.capture_panel.set_image(bgr, caption)
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
            description = f"识别候选：{decision['best_label']}"
            detail = f"匹配分数 {decision['score']:.4f} · 分差 {decision['margin']:.4f} · 请核对入口图"
            self.decision_label.configure(text=f"{description} · {detail}", fg=AMBER)
        else:
            text = '没有有效布局候选'
            self.decision_label.configure(text=text, fg=AMBER)

    def _recognition_context(self):
        settings = self.settings
        return (self.profile.key, settings['layout_recognition_backend'],
                tuple(settings['capture_roi']), tuple(settings['capture_reference_size']),
                tuple(settings['game_window_titles']), selected_template_variant(settings, self.profile.key))

    def _invalidate_recognition(self):
        request = getattr(self, '_recognition_request', None)
        if request is not None:
            request.cancel.set()
            if (getattr(self, '_active_work', None) or {}).get('id') == request.work_id:
                self._active_work = None
                self.busy = False
        self._recognition_request = None
        self._pending_confirmation = None
        self._pending_confirmation_roi = None
        self._confirmation_scheduled = None

    def _recognition_is_current(self, request, *, require_capture=True):
        return (request is not None and request is getattr(self, '_recognition_request', None)
                and not self.closed and request.session == self.session_id
                and not request.cancel.is_set() and request.context == self._recognition_context()
                and (not require_capture or request.snapshot_ready))

    def _recognition_capture_invalid(self, request):
        if (request is not getattr(self, '_recognition_request', None)
                or (self._active_work or {}).get('id') != request.work_id
                or request.session != self.session_id or self.closed):
            return
        self._invalidate_recognition()
        self._set_status('截图完成前地图已变化，请重新打开地图后识别', AMBER)
        self._log('布局识别取消：截图完成前发生地图开关，未采用本次画面')

    def _confirm_pending_layout(self):
        pending = self._pending_confirmation
        if pending is not None:
            self._confirm_recognition(pending[0], pending[1], self._pending_confirmation_roi, proactive=True)

    def _maybe_schedule_confirmation(self):
        pending = getattr(self, '_pending_confirmation', None)
        if (pending is None or getattr(self, '_confirmation_in_progress', None) is not None
                or getattr(self, '_confirmation_scheduled', None) is pending
                or not getattr(getattr(self, 'map_state', None), 'is_open', False)
                or getattr(self, 'settings_page', None) is not None or self.closed
                or not game_window_is_foreground(self.settings['game_window_titles'])):
            return
        if not RouteProjectionApp._recognition_can_confirm(self, pending[0], pending[2], pending[3],
                getattr(pending[1], 'recognition_request_id', None)):
            self._invalidate_recognition()
            self._set_status('布局候选已失效，请重新打开地图识别', AMBER)
            return
        self._confirmation_scheduled = pending
        roi = self._pending_confirmation_roi
        def confirm():
            if self._confirmation_scheduled is not pending:
                return
            self._confirmation_scheduled = None
            if self._pending_confirmation is pending:
                self._confirm_recognition(pending[0], pending[1], roi)
        self.after(0, confirm)

    def request_work(self, mode: str, *, auto: bool) -> None:
        work_requested_at = time.perf_counter()
        gui_at_request = self._gui_diagnostic_snapshot()
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if mode == 'project' and (self._pending_confirmation is not None or self._confirmation_in_progress is not None):
            self._set_status('请先确认本次布局候选，再重新截图配准', AMBER)
            return
        previous = getattr(self, '_active_work', None) or {}
        restarting = mode == 'recognize' and self.busy and previous.get('mode') == 'recognize'
        if self.busy and not restarting:
            active = '正在识别布局' if previous.get('mode') == 'recognize' else '正在配准路线'
            requested = '识别' if mode == 'recognize' else '配准'
            self._set_status(f'{active}，本次{requested}未启动，请完成后重试', AMBER)
            return
        fixed_layout = self.locked_layout if mode == "project" else None
        if mode == "project" and fixed_layout is None:
            self._set_status("请先按 F8 识别并选定布局，再打开地图投影", AMBER)
            RouteProjectionApp._map_toast(self, "布局尚未选定", "请按 F8 识别并选定布局，然后自动配准路线。", success=False)
            return
        recognition_backend = self.settings.get('layout_recognition_backend', 'icon_structure')
        if mode == "recognize":
            self._invalidate_recognition()
            if previous.get('cancel') is not None:
                previous['cancel'].set()
            self._pending_confirmation = None
            self.hide_overlay()
            self.session_id = uuid.uuid4().hex[:12]
            self.last_result = self.last_decision = None
            if hasattr(self, 'rank_widgets'):
                for name, score, _bar in self.rank_widgets:
                    name.configure(text='—', fg=INK)
                    score.configure(text='—', fg=MUTED)
            if restarting:
                self._log('重复识别：上一请求已作废，立即重新截图，只采用本次识别结果')
        work_id = uuid.uuid4().hex
        cancel = threading.Event()
        self._active_work = dict(id=work_id, mode=mode, cancel=cancel,
                                 generation=self.map_generation, session=self.session_id)
        self.busy = True
        template_direct = self.settings.get('registration_backend') == 'template'
        fast_display = mode == 'project' and template_direct and getattr(self, '_work_bridge', None) is not None
        if mode == "project":
            self.hide_overlay()
            RouteProjectionApp._map_toast(self, "正在配准路线", f"已锁定 {fixed_layout}，" +
                ("正在分别匹配各层图片…"), success=True)
        generation = self.map_generation
        session_id = self.session_id
        profile = self.profile
        roi = list(self.settings["capture_roi"])
        reference_size = list(self.settings["capture_reference_size"])
        window_titles = list(self.settings["game_window_titles"])
        capture_delay_configured = map_capture_delay_ms(self.settings)
        clean_capture = not self.settings.get('capture_exclusion_enabled', True)
        template_variant_id = selected_template_variant(self.settings, profile.key)
        opacity = float(self.settings["opacity"])
        recognition = None
        if mode == 'recognize':
            recognition = RecognitionRequest(work_id, generation, session_id, self._recognition_context(), cancel)
            self._recognition_request = recognition
        try:
            retriever = self._layout_retriever() if mode == 'recognize' and recognition_backend in BACKEND_LABELS else None
        except Exception as exc:
            self.busy = False
            self._active_work = None
            if recognition is not None:
                self._invalidate_recognition()
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
                if recognition is not None:
                    recognition.window = recognition_window
                screen_region = game_capture_region(client, roi, reference_size)
                ensure_region_on_screen(screen_region)
                screen_roi = screen_region.as_list()
                try:
                    if clean_capture:
                        flush_desktop()
                    screenshot = capture_bgr(screen_roi)
                    if recognition is not None and not recognition.captured(time.perf_counter()):
                        raise RecognitionCancelled()
                finally:
                    if getattr(self, '_work_bridge', None) is not None:
                        self._work_bridge.ready.emit(('capture_finished', work_id))
                check_cancelled(cancel.is_set)
                if capture_window_id is not None and int(ctypes.windll.user32.GetForegroundWindow()) != capture_window_id:
                    raise RuntimeError('截图期间游戏窗口发生变化，请重新配准')
                if recognition_window is not None and recognition_window_identity(window_titles) != recognition_window:
                    raise RuntimeError('截图期间游戏窗口发生变化，请重新识别')
                capture_ms = (time.perf_counter() - started) * 1000.0
                bridge = getattr(self, '_work_bridge', None)
                if bridge is not None:
                    bridge.ready.emit(('work_progress', work_id, session_id, generation,
                        '截图已完成，可关闭地图；正在识别布局…' if mode == 'recognize' else '正在配准路线，自动比较各层图片…'))
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
                if recognition is not None:
                    result.recognition_request_id = work_id
                event = ("work", mode, auto, generation, screen_roi, result, None)
                if fast_display:
                    meta = dict(work_id=work_id, posted_at=time.perf_counter(), requested_at=work_requested_at,
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
                    self.results.put(('recognition_work' if mode == 'recognize' else 'work_result', work_id, event))
                if mode in ("project", "recognize"):
                    check_cancelled(cancel.is_set)
                    try:
                        saved = save_workflow_result(result)
                        self.results.put(("saved", result, str(saved), None))
                    except Exception as exc:
                        self.results.put(("saved", result, None, str(exc)))
            except RecognitionCancelled:
                if recognition is not None and recognition.capture_invalid:
                    try:
                        self._work_bridge.ready.emit(('recognition_capture_invalid', recognition))
                    except RuntimeError:
                        pass
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
                        dict(work_id=work_id, posted_at=time.perf_counter(), requested_at=work_requested_at,
                             key_started_at=key_started_at, session=session_id, ack=threading.Event())))
                elif not fast_display:
                    self.results.put(('recognition_work' if mode == 'recognize' else 'work_result', work_id, event))
            finally:
                try:
                    if getattr(self, '_work_bridge', None) is not None:
                        self._work_bridge.ready.emit(('capture_finished', work_id))
                except RuntimeError:
                    pass
                if (retriever is not None and hasattr(retriever, 'close') and
                        (self.closed or getattr(self, 'layout_retrievers', {}).get((profile.key, recognition_backend)) is not retriever)):
                    retriever.close()
        thread_dispatched_at = time.perf_counter()
        try:
            self.capture_policy.begin_capture(work_id)
            threading.Thread(target=worker, name=f"route-{mode}", daemon=True).start()
        except Exception as exc:
            self.capture_policy.end_capture(work_id)
            cancel.set()
            self.busy = False
            self._active_work = None
            if recognition is not None:
                self._invalidate_recognition()
            self._set_status(f'{"识别" if mode == "recognize" else "配准"}任务未启动：{exc}', BAD)

    @pyqtSlot(object)
    def _receive_work_event(self, payload):
        entered_at = time.perf_counter()
        if payload[0] == 'capture_finished':
            if not self.closed:
                self.capture_policy.end_capture(payload[1])
            return
        if payload[0] == 'work_progress':
            self._handle_map_key()
            _, work_id, session, generation, message = payload
            active = getattr(self, '_active_work', None) or {}
            if (not self.closed and self.busy and active.get('id') == work_id
                    and self.session_id == session and
                    (self.map_generation == generation or
                     (active.get('mode') == 'recognize' and self._recognition_is_current(self._recognition_request)))):
                self._set_status(message)
            return
        if not self.closed:
            self._handle_map_key()
        if payload[0] == 'recognition_capture_invalid':
            self._recognition_capture_invalid(payload[1])
            return
        if payload[0] == 'floor_reference':
            _, key, image, error = payload
            if self.closed or key != self._reference_image_key():
                return
            self._floor_reference_pending = None
            if error:
                self._reference_load_error = key
                self._log(f'单层参考图不可用：{error}')
                self._set_status(f'参考图加载失败：{error}', AMBER)
                return
            self._floor_reference_image = (key, image)
            self._refresh_reference_map()
            return
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
            if 'work_id' in meta and meta['work_id'] != (self._active_work or {}).get('id'):
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
        if self._pending_confirmation is not None or self._confirmation_in_progress is not None:
            self._pending_auto_projection = None
            self._maybe_schedule_confirmation()
            return
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
        result_session = getattr(result, 'session_id', None)
        if self.closed or (result_session is not None and result_session != self.session_id):
            return
        active = self._active_work or {}
        if active and active.get('generation', self.map_generation) != generation:
            return
        recognition = getattr(self, '_recognition_request', None) if mode == 'recognize' else None
        request_id = getattr(result, 'recognition_request_id', None)
        if request_id is not None:
            if (recognition is None or recognition.work_id != request_id
                    or not self._recognition_is_current(recognition)):
                return
        elif recognition is not None:
            # Errors have no result object; the queued work ID already matched.
            if (generation != recognition.generation or not self._recognition_is_current(
                    recognition, require_capture=error is None)):
                return
        elif mode == 'recognize' and generation != self.map_generation:
            return
        self.busy = False
        self._active_work = None
        if mode == 'recognize' and result is not None:
            if getattr(result, 'layout_recognition_backend', 'icon_structure') != getattr(self, 'settings', {}).get('layout_recognition_backend', 'icon_structure'):
                return
            if getattr(result, 'layout_recognition_backend', 'icon_structure') in BACKEND_LABELS:
                if (getattr(self, 'closed', False) or (recognition is None and generation != self.map_generation) or
                        result.session_id != self.session_id or result.profile_key != self.profile.key):
                    return
            if getattr(result, 'recognition_window', None) is not None:
                try:
                    if not recognition_window_matches(result.recognition_window, self.settings['game_window_titles'],
                                                      allow_app=recognition is not None):
                        self._invalidate_recognition()
                        self._set_status('游戏窗口已变化，请重新识别', AMBER)
                        return
                except (OSError, RuntimeError):
                    return
        if mode == 'recognize' and error and recognition is None and generation != self.map_generation:
            return
        if generation != self.map_generation:
            # A worker may have populated tentative hypotheses after a reset.
            # They must not leak into the next map session.
            if mode != "recognize":
                if getattr(self, 'map_keys', None) is not None:
                    self._log(f'丢弃迟到配准结果：任务代次 {generation}，当前 {self.map_generation}')
                return
        if error:
            if mode == 'recognize':
                self._recognition_request = None
            self._set_status(f"处理失败：{error}", BAD)
            self._log(f"{mode} 失败：{error}")
            if mode == "recognize":
                self._pending_auto_projection = None
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
            self.path_var.set(f"诊断文件：{result.run_directory or '后台保存中…'}")
            if mode == "recognize":
                self._pending_auto_projection = None
                session_id = getattr(result, "session_id", getattr(self, "session_id", None))
                profile_key = getattr(result, "profile_key", getattr(getattr(self, "profile", None), "key", None))
                if not RouteProjectionApp._recognition_can_confirm(self, generation, session_id, profile_key,
                        getattr(result, 'recognition_request_id', None)):
                    self._set_status("识别期间地图状态已变化，请按 F8 重新识别", AMBER)
                    RouteProjectionApp._show_recognition_toast(
                        self, "识别结果已失效", "地图画面已变化，请重新打开地图后按 F8 识别。",
                        success=False, roi=roi)
                elif (result.accepted and result.selected_layout in self.catalog.layouts(self.profile)
                      and result.decision):
                    if self.settings.get('auto_select_layout', False):
                        self._lock_recognized_layout(result.selected_layout, automatic=True)
                        RouteProjectionApp._show_recognition_toast(
                            self, '已自动选择布局', f'{result.selected_layout}；按最高匹配分数选择，可能选错。请缩到最小后重新截图配准。',
                            success=True, roi=roi)
                        return
                    RouteProjectionApp._show_recognition_toast(
                        self, "布局识别成功", f"首选候选为 {result.selected_layout}，请对照入口图选择布局。",
                        success=True, roi=roi)
                    self._set_status(f"布局识别成功：{result.selected_layout}；等待确认锁定", GREEN)
                    self._pending_confirmation = (generation, result, session_id, profile_key)
                    self._pending_confirmation_roi = list(roi)
                    self._pending_auto_projection = None
                    self._set_status('布局候选已就绪；再次开图或点击“确认布局”核对', GREEN)
                    self._maybe_schedule_confirmation()
                else:
                    self._set_status("当前画面暂无法可靠判断布局；请按 F8 重试", AMBER)
                    reason = result.reasons[0] if getattr(result, "reasons", None) else "当前地图画面信息不足"
                    RouteProjectionApp._show_recognition_toast(
                        self, "布局识别失败", f"{reason}。请多探索地图、保持画面稳定，然后按 F8 重试。",
                        success=False, roi=roi)
            else:
                timing = result.timings_ms
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
                                         f" · {result.selected_layout} / {FLOOR_NAMES.get(result.floor, '未取得楼层候选')} / "
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
                                 profile_key: str | None, request_id: str | None = None) -> bool:
        if getattr(self, "closed", False):
            return False
        if getattr(self, "session_id", None) != session_id:
            return False
        if getattr(getattr(self, "profile", None), "key", None) != profile_key:
            return False
        request = getattr(self, '_recognition_request', None)
        if request_id is not None and (request is None or request.work_id != request_id):
            return False
        if request is not None:
            return (generation == request.generation and self._recognition_is_current(request)
                    and (request.window is None or recognition_window_matches(
                        request.window, self.settings['game_window_titles'], allow_app=True)))
        current = self.map_generation
        return (current == generation or
                (current == generation + 1 and not getattr(getattr(self, "map_state", None), "is_open", True)))

    def _confirm_recognition(self, generation: int, result: WorkflowResult,
                             roi: list[int] | None = None, *, proactive=False) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        pending = getattr(self, "_pending_confirmation", None)
        if pending is None or pending[0] != generation or pending[1] is not result:
            return
        request_id = getattr(result, 'recognition_request_id', None)
        if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3], request_id):
            self._invalidate_recognition()
            self._set_status("地图状态已变化；请按 F8 重新识别", AMBER)
            return
        if (not proactive and (not self.map_state.is_open or not game_window_is_foreground(
                self.settings['game_window_titles']))) or self._confirmation_in_progress is not None:
            return
        self._pending_confirmation = None
        self._confirmation_in_progress = pending
        self._refresh_status()
        decision = result.decision
        candidates = sorted(decision.get("ranking") or [
            {"layout": result.selected_layout, "score": decision["score"]}
        ], key=lambda item: item['score'], reverse=True)[:2]
        if not self.profile.entrance_root and getattr(self.profile, 'clean_root', None):
            previews = []
            for item in candidates:
                floor = item.get('candidate_floor', '2nd')
                if floor not in self.profile.floors:
                    floor = self.profile.floors[0]
                previews.append(dict(item, preview_floor=floor,
                    preview_path=self.profile.clean_root/item['layout']/f"{item['layout']}_{floor}.png"))
            candidates = previews
        try:
            selected_layout = messagebox.confirm_layout(
                candidates,
                self.profile.entrance_root if getattr(self, "profile", None) else None,
                screen_roi=roi,
                is_current=lambda: RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3], request_id)
                    and (proactive or self.map_state.is_open),
            )
        finally:
            if self._confirmation_in_progress is pending:
                self._confirmation_in_progress = None
                if not self.closed:
                    self._refresh_status()
        self._handle_map_key()
        if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3], request_id):
            if self.session_id == pending[2]:
                self._invalidate_recognition()
                self._set_status("地图状态已变化；请按 F8 重新识别", AMBER)
            return
        if not proactive and not self.map_state.is_open:
            self._pending_confirmation = pending
            self._pending_confirmation_roi = roi
            self._set_status('布局候选已保留；再次开图或点击“确认布局”核对', GREEN)
            return
        if selected_layout:
            self._lock_recognized_layout(selected_layout, automatic=False)
        else:
            self._recognition_request = None
            self.layout_var.set(self.locked_layout or AUTO)
            self._update_assets()
            self.decision_label.configure(text=(f"仍锁定 {self.locked_layout}" if self.locked_layout else
                                                "识别结果未确认，布局尚未锁定"), fg=AMBER)
            self._set_status((f"本次识别未确认，仍锁定 {self.locked_layout}；若已换局请复位地图" if self.locked_layout
                              else "本次识别未确认，布局尚未锁定；可再次按 F8 识别"), MUTED)

    def _lock_recognized_layout(self, layout: str, *, automatic: bool):
        self._recognition_request = None
        self._pending_confirmation = None
        self._pending_confirmation_roi = None
        self.locked_layout = layout
        self.layout_var.set(layout)
        self._update_assets()
        action = '已自动选择' if automatic else '已确认锁定'
        self.decision_label.configure(text=f'{action} {layout}', fg=ACCENT)
        self._set_status(f'{action} {layout}；请将地图缩到最小后按地图键配准', ACCENT)
        self._log(f'布局识别：{action} {layout}')

    def hide_overlay(self) -> None:
        if hasattr(self, '_motion'):
            self._motion.stop()
        self.live_projection = None
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

    def _reset_reference_map(self):
        self.reference_visible = False
        self.reference_floor_source = '手动'
        self.reference_revision += 1
        self._reference_identity = None
        self.reference_context = None
        self._floor_reference_image = None
        self._floor_reference_pending = None
        self._reference_load_error = None
        self._reference_display_key = None
        self.reference_overlay.hide()

    def _sync_reference_context(self):
        floor = self.settings.get('reference_floor')
        if floor not in self.profile.floors:
            floor = self.settings['reference_floor'] = self.profile.floors[0]
        variant = selected_template_variant(self.settings, self.profile.key)
        identity = (self.profile.key, self.locked_layout, variant, floor)
        previous = self._reference_identity
        if identity != previous:
            if previous is not None and identity[:2] != previous[:2]:
                self.reference_visible = False
                self.reference_floor_source = '手动'
            self.reference_revision += 1
            self._reference_identity = identity
            self._floor_reference_image = None
            self._floor_reference_pending = None
            self._reference_load_error = None
            self._reference_display_key = None
            self.reference_overlay.hide()
        self.reference_context = ((self.reference_revision, list(self.settings['capture_roi']), floor,
                                   self.profile.key, self.locked_layout) if self.locked_layout else None)
        self.reference_toggle_button.setText(
            f'{"隐藏" if self.reference_visible else "显示"}参考图 ({self.settings["reference_hotkey"]})')
        self.floor_selector.refresh()
        self.status_rows['reference'].setText(
            FLOOR_NAMES[floor] + ' · 开启' if self.reference_visible and self.locked_layout else '未显示')


    def _reference_image_key(self):
        self._sync_reference_context()
        context = self.reference_context
        if context is None:
            return None
        return (context[0], context[3], context[4], selected_template_variant(self.settings, self.profile.key), context[2])

    def _toggle_reference_map(self, _checked=False):
        self._sync_reference_context()
        if not self.locked_layout:
            self._set_status('请先识别并选定布局，再显示参考图', AMBER)
            if game_window_is_foreground(self.settings['game_window_titles']):
                self._show_recognition_toast('布局尚未选定', '请先按 F8 识别并选定布局。', success=False)
            return
        self.reference_visible = not self.reference_visible
        self._reference_load_error = None
        self._refresh_reference_map()

    def _reference_floor_changed(self, _index=None):
        floor = self.floor_combo.currentData()
        self._set_reference_floor(floor, source='手动', reveal=True)

    def _set_reference_floor(self, floor, *, source, reveal):
        if floor not in self.profile.floors:
            return
        changed = self.settings['reference_floor'] != floor
        self.settings['reference_floor'] = floor
        blocked = self.floor_combo.blockSignals(True)
        try:
            self.floor_combo.setCurrentIndex(self.floor_combo.findData(floor))
        finally:
            self.floor_combo.blockSignals(blocked)
        if changed:
            save_settings(self.settings, self.settings_path)
        self._sync_reference_context()
        self.reference_floor_source = source
        if reveal and self.locked_layout:
            self.reference_visible = True
        self._refresh_reference_map()

    def _next_reference_floor(self):
        floor = self.settings.get('reference_floor')
        floors = self.profile.floors
        index = (floors.index(floor) + 1) % len(floors) if floor in floors else 0
        self.floor_combo.setCurrentIndex(self.floor_combo.findData(floors[index]))

    def _refresh_reference_map(self):
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        self._sync_reference_context()
        context = self.reference_context
        if (not self.reference_visible or context is None or self.closed
                or self.settings_page is not None or own_window_is_foreground()
                or QApplication.activeModalWidget() is not None
                or not game_window_is_foreground(self.settings["game_window_titles"])):
            self.reference_overlay.hide()
            return
        _, roi, floor, _, layout = context
        variant = selected_template_variant(self.settings, self.profile.key)
        key = self._reference_image_key()
        cached = getattr(self, '_floor_reference_image', None)
        image = cached[1] if cached is not None and cached[0] == key else None
        if image is None:
            self.reference_overlay.hide()
            if (getattr(self, '_floor_reference_pending', None) != key
                    and getattr(self, '_reference_load_error', None) != key):
                self._floor_reference_pending = key
                profile_key = self.profile.key
                def worker():
                    try:
                        from src.registration.template_matching.floors import load
                        loaded = load(profile_key, layout, variant, floor)[3]
                        packet = ('floor_reference', key, loaded, None)
                    except Exception as exc:
                        packet = ('floor_reference', key, None, str(exc))
                    try:
                        self._work_bridge.ready.emit(packet)
                    except RuntimeError:
                        pass
                try:
                    threading.Thread(target=worker, name='floor-reference', daemon=True).start()
                except Exception as exc:
                    self._floor_reference_pending = None
                    self._reference_load_error = key
                    self._log(f'参考图加载未启动：{exc}')
            return
        display_key = (key, self.settings['reference_opacity'], self.settings['reference_hotkey'],
                       self.settings['reference_floor_hotkey'], self.reference_floor_source)
        if self.reference_overlay.visible and getattr(self, '_reference_display_key', None) == display_key:
            return
        self.reference_overlay.show_map(image, roi,
            f"当前参考楼层：{FLOOR_NAMES[floor]}（{self.reference_floor_source}） · {self.settings['reference_hotkey']} 显隐 / {self.settings['reference_floor_hotkey']} 切层\n"
            f"{layout} · {TEMPLATE_VERSION_NAMES[variant]}",
            self.settings['reference_opacity'])
        self._reference_display_key = display_key

    def _finish_map_feedback(self, generation, roi, result):
        if self.closed or generation != self.map_generation or not self.map_state.is_open:
            return
        if (result.accepted and getattr(result, 'layer', None) is not None
                and result.profile_key == self.profile.key and result.selected_layout == self.locked_layout
                and getattr(result, 'variant_id', None) == selected_template_variant(self.settings, self.profile.key)
                and result.floor in self.profile.floors):
            self._set_reference_floor(result.floor, source='最近配准', reveal=False)
        label = f"{result.selected_layout or '布局待确认'} · {FLOOR_NAMES.get(result.floor, '未取得楼层候选')}"
        if result.accepted:
            detail = label + ("，路线已投影。" if self.show_overlay_check.isChecked() else "，投影显示已关闭。")
            warnings = getattr(getattr(result, 'registration', None), 'validation', {}).get('warnings', [])
            if warnings:
                detail += '\n' + '；'.join(warnings)
                self._set_status('模板匹配候选可能错位，请核对位置', AMBER)
            RouteProjectionApp._map_toast(self, '模板候选可能错位' if warnings else '模板匹配候选已投影',
                                         detail, success=not warnings, roi=roi)
        else:
            reason = result.reasons[0] if result.reasons else "可见地图信息不足"
            shown = getattr(getattr(self, "reference_overlay", None), "visible", False)
            detail = label + "：" + reason + (f"。左上角已显示参考路线，{self.settings['reference_hotkey']} 可隐藏。" if shown else "。请继续探索后重试。")
            RouteProjectionApp._map_toast(self, "地图暂未配准", detail, success=False, roi=roi)

    def _handle_reference_key(self):
        # Key edges, including modifier state, are delivered by the Raw Input queue.
        self._refresh_reference_map()

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
                        RouteProjectionApp._invalidate_recognition(self)
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
                    if key.action == 'next_round':
                        if key.down and enabled and game_window_is_foreground(self.settings['game_window_titles']):
                            self.start_next_round()
                        continue
                    if key.action in ('recognize', 'project'):
                        if key.down and enabled and game_window_is_foreground(self.settings['game_window_titles']):
                            self.request_work(key.action, auto=False)
                        continue
                    if key.action in CHEST_ACTIONS:
                        if key.down and enabled:
                            self._change_chest_count(key.action)
                        continue
                    if key.action in ('reference_toggle', 'reference_next_floor'):
                        if key.down and enabled:
                            if key.action == 'reference_toggle':
                                self._toggle_reference_map()
                            else:
                                self._next_reference_floor()
                        continue
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
        if getattr(self, "toast", None) is not None:
            self.toast.hide()
        self.projection_delay_pending = False
        self.projection_wait_elapsed_ms = None
        self.projection_triggered_at = None
        self.hide_overlay()
        recognition = getattr(self, '_recognition_request', None)
        if recognition is not None:
            recognition.map_changed(key_detected_at)
            if recognition.capture_invalid:
                self._recognition_capture_invalid(recognition)
        if closed_by_escape or event == "closed":
            self.projection_status_text = "ESC 已关闭" if closed_by_escape else "地图键已关闭"
            if recognition is not None and recognition.capture_invalid:
                self._set_status('地图已关闭；本次截图未完成，请重新开图识别', AMBER)
            elif recognition is not None and recognition.snapshot_ready and self.busy:
                self._set_status('地图已关闭，已完成截图的布局识别继续进行', MUTED)
            else:
                self._set_status("地图已关闭，投影已隐藏", MUTED)
            self._log("ESC：地图关闭，投影已隐藏" if closed_by_escape else "地图键：地图关闭，投影已隐藏")
            return
        if getattr(self, '_pending_confirmation', None) is not None:
            RouteProjectionApp._maybe_schedule_confirmation(self)
            if self._pending_confirmation is not None:
                self._set_status('地图已打开，等待确认本局布局', GREEN)
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
        action = '自动匹配单层图片'
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
            recognition = getattr(self, '_recognition_request', None)
            if recognition is not None:
                recognition.map_changed(time.perf_counter())
                if recognition.capture_invalid:
                    self._recognition_capture_invalid(recognition)
            self.map_generation += 1
            self.projection_delay_pending = False
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
        self.projection_status_text = '定尺度模板匹配：自动匹配单层图片'
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
        if (getattr(self, '_pending_confirmation', None) is not None
                or getattr(self, '_confirmation_in_progress', None) is not None):
            return
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
        self._refresh_chest_overlay()
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
            elif event[0] == 'retrieval_index':
                if (event[1] == self.profile.key and
                        self.settings.get('layout_recognition_backend', 'icon_structure') == (event[4] if len(event) > 4 else 'icon_structure') and
                        (len(event) < 6 or getattr(self, 'layout_retrievers', {}).get((event[1], event[4])) is event[5])):
                    self._log(f'识别资源准备失败：{event[3]}' if event[3] else f'布局识别资源就绪：{event[2][:12]}')
            elif event[0] in ('recognition_work', 'work_result'):
                _, work_id, work = event
                active = getattr(self, '_active_work', None) or {}
                if active.get('id') == work_id and not active['cancel'].is_set():
                    self._apply_result(*work[1:])
                    RouteProjectionApp._drain_pending_projection(self)
            elif event[0] == "work":
                self._apply_result(*event[1:])
                RouteProjectionApp._drain_pending_projection(self)
        self._flush_logs()
        self._maybe_schedule_confirmation()
        self.after(25, self._tick)

    def _shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._invalidate_recognition()
        active = getattr(self, '_active_work', None) or {}
        if active.get('cancel') is not None:
            active['cancel'].set()
        self._template_prewarmer.close()
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
        self.chest_overlay.destroy()
        self.toast.close()
        self.capture_policy.close()

    def closeEvent(self, event):
        page = self.settings_page
        if page is not None and (page._settings_dirty or page._settings_save_job is not None):
            if not page._persist_settings(show_error=True):
                event.ignore()
                return
        self._shutdown()
        event.accept()
        QApplication.instance().quit()
