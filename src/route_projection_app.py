from __future__ import annotations

from src.route_projection_core import *
from PyQt6.QtCore import QPoint, Qt, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPixmap
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QSlider, QTextEdit, QVBoxLayout,
                             QWidget, QComboBox, QCheckBox, QStackedWidget, QSizeGrip)
from src.route_projection_qt_ui import (STYLE, BG, CARD, CARD_2, INK, MUTED, ACCENT, GREEN, AMBER, BAD,
                                        Value, CompatLabel, ImagePanel, SettingsDialog, LayoutConfirmationDialog,
                                        ManualLayoutConfirmationDialog,
                                        RouteOverlayWindow, ReferenceMapWindow, NotificationToast, DraggableTitleBar,
                                        exclude_window_from_capture, physical_to_qt_geometry)
from src.projection.route_overlay import load_route_overlay_asset
from src.projection_workflow import switch_result_version, complete_projection_preview
from src.version import APP_TITLE, APP_VERSION
from src.registration.runtime_registration import SoftPoseHint, prepare_registration_template
from src.registration.multiscale_structure.runtime import initialize as initialize_registration, prewarm_template
from src.map_presence import MapPresence, save_map_presence_diagnostic
import math
from src.floor_indicator import CapturedFloorSample
from src.map_probe_diagnostics import LatestDiagnosticWriter
from src.paths import CACHE_ROOT, LOG_ROOT, DIAGNOSTIC_ROOT
from src.app_catalog import map_capture_delay_ms, selected_template_variant, TEMPLATE_VERSION_NAMES, load_default_settings
from src.capture.map_keys import GlobalMapKeys
from src.layout_retrieval import LayoutRetriever, BACKEND_LABELS

MAP_PROBE_WINDOW_MS = 500


def recognition_window_identity(titles):
    client = foreground_game_client(titles)
    return [int(ctypes.windll.user32.GetForegroundWindow()), *client.as_list()]


def game_window_is_foreground(titles: list[str]) -> bool:
    """Keep foreground title lookup injectable for runtime checks and tests."""
    if not titles:
        return False
    current = foreground_window_title().casefold()
    return any(title.casefold() in current for title in titles)


class QtMessageBox:
    @staticmethod
    def confirm_layout(candidates, entrance_root, *, screen_roi=None):
        screen = QGuiApplication.primaryScreen()
        if screen_roi is not None:
            try:
                x, y, width, height, _scale = physical_to_qt_geometry(*map(int, screen_roi))
                screen = QGuiApplication.screenAt(QPoint(x + width // 2, y + height // 2)) or screen
            except (OSError, ValueError):
                pass
        dialog = LayoutConfirmationDialog(candidates, entrance_root, screen=screen)
        dialog.show()
        exclude_window_from_capture(dialog)
        bounds = screen.availableGeometry()
        dialog.move(bounds.center() - dialog.rect().center())
        dialog.raise_()
        dialog.activateWindow()
        accepted = dialog.exec() == LayoutConfirmationDialog.DialogCode.Accepted
        choice = dialog.selected_layout if accepted else None
        skip = accepted and dialog.skip_confirmation.isChecked()
        dialog.deleteLater()
        return choice, skip

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

    @staticmethod
    def showerror(title, text, parent=None):
        QMessageBox.critical(parent, title, text)


messagebox = QtMessageBox


class MapProbeBridge(QObject):
    ready = pyqtSignal(object)


class RouteProjectionApp(QWidget):
    def __init__(self, *, settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
        enable_physical_pixel_coordinates()
        super().__init__()
        self.catalog = MapCatalog()
        self.settings_path = Path(settings_path)
        self.settings = load_settings(self.catalog, self.settings_path)
        self.profile = self.catalog.profile(self.settings["map_id"], self.settings["difficulty_id"])
        self.model_cache = {}
        self.prepared_cache = {}
        self.template_prepare_lock = threading.Lock()
        self.asset_cache = {}
        self.registration_priors: dict[tuple[str, str, str], np.ndarray] = {}
        self.registration_soft_priors: dict[tuple[str, str, str], SoftPoseHint] = {}
        self.registration_evidence: dict[tuple[str, str, str], TemporalEvidence] = {}
        self.session_id = uuid.uuid4().hex[:12]
        self.results: queue.Queue[tuple] = queue.Queue()
        self._map_probe_bridge = MapProbeBridge(self)
        self._map_probe_bridge.ready.connect(self._receive_map_probe, Qt.ConnectionType.QueuedConnection)
        self._work_bridge = MapProbeBridge(self)
        self._work_bridge.ready.connect(self._receive_work_event, Qt.ConnectionType.QueuedConnection)
        self._pending_auto_projection = None
        self._map_probe_diagnostics = LatestDiagnosticWriter(save_map_presence_diagnostic)
        self._map_probe_trace = None
        self.busy = False
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
        self.reference_cache = {}
        self.reference_key_was_down = map_key_down(self.settings["reference_hotkey"])
        self.reference_suspended = False
        self._pending_confirmation = None
        self.settings_dialog: SettingsDialog | None = None
        self.preview_images = {}
        self.floor_probe_busy = False
        self.floor_probe_next = 0.0
        self.floor_probe_candidate: str | None = None
        self.floor_probe_count = 0
        self.floor_probe_unknown_count = 0
        self.floor_probe_needs_projection = False
        self.floor_monitor_active = False
        self.map_presence_text = "等待地图键"
        self.map_presence_elapsed_ms: float | None = None
        self.map_presence_started_at: float | None = None
        self.map_probe_pending = False
        self.hotkeys: GlobalActionHotkeys | None = None
        self.hotkeys_suspended = False
        self.map_var = Value(self.profile.map_name)
        self.difficulty_var = Value(self.profile.difficulty_name)
        self.floor_var = Value(self.settings["floor"])
        self.layout_var = Value(AUTO)
        self.status_var = Value("等待打开游戏地图")
        self.detect_var = Value("尚未识别布局")
        self.overlay_var = Value("投影已隐藏", self._refresh_status)
        self.floor_detect_var = Value("楼层标识：等待打开地图", self._refresh_status)
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
        self.reference_overlay = ReferenceMapWindow()
        self.toast = NotificationToast()
        self._map_key_bridge = MapProbeBridge(self)
        self._map_key_bridge.ready.connect(self._receive_map_keys, Qt.ConnectionType.QueuedConnection)
        self.map_keys = GlobalMapKeys(self.settings['map_key'], self.settings['game_window_titles'],
                                     lambda: self._map_key_bridge.ready.emit(None))
        self.apply_settings(self.settings, save=False)
        if self.map_keys is not None:
            self.map_keys.start()
        self.show()
        self.main_capture_excluded = exclude_window_from_capture(self)
        self.after(25, self._tick)
        self.after(300, self._start_model_load)
        self.after(350, self._start_registration_backend)
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
        self.floor_combo.currentIndexChanged.connect(self._manual_floor_changed)
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
        self.recognition_method_label = QLabel('识别方法：CNN')
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
        self.preview_labels = {"capture": self.capture_panel, "alignment": self.overlay_panel}
        self.preview_captions = {"capture": self.capture_panel.caption, "alignment": self.overlay_panel.caption}
        self._refresh_template()

    def _build_right(self, layout):
        self.current_map_label = CompatLabel()
        self.current_difficulty_label = CompatLabel()
        self.current_layout_label = CompatLabel()
        self.match_label = CompatLabel()
        for label in (self.current_map_label, self.current_difficulty_label, self.current_layout_label, self.match_label):
            layout.addWidget(label)
        layout.addWidget(self._heading("状态信息"))
        self.state_label = CompatLabel("等待打开游戏地图")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.status_rows = {}
        for key, title in (("capture", "截图状态"), ("map_presence", "地图界面"),
                           ("recognition", "识别状态"), ("projection", "投影状态"),
                           ("route", "当前路线"), ("map_presence_ms", "地图界面校验耗时"),
                           ("capture_ms", "截图耗时"), ("floor_ms", "楼层识别耗时"),
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
        self.decision_label = CompatLabel("尚未识别布局")
        self.decision_label.setWordWrap(True)
        layout.addWidget(self.decision_label)
        self.rank_widgets = []
        self.ranking_heading = self._heading("布局置信度排名")
        layout.addWidget(self.ranking_heading)
        for _ in range(2):
            row = QHBoxLayout()
            name = CompatLabel("—")
            pct = CompatLabel("—")
            row.addWidget(name, 1)
            row.addWidget(pct)
            layout.addLayout(row)
            self.rank_widgets.append((name, pct, None))
        layout.addStretch(1)
        layout.addWidget(self._heading("路线图预览（当前选择）"))
        self.route_panel = ImagePanel("", "尚未选择路线")
        self.route_panel.setFixedHeight(145)
        layout.addWidget(self.route_panel)
        self.metric_label = CompatLabel()
        self.metric_label.setWordWrap(True)
        self.path_label = CompatLabel()
        self.path_label.setWordWrap(True)
        self.roi_label = CompatLabel()
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
        self.log.setStyleSheet("background: #121e2a; color: #c6d7e7; border: 1px solid #314a5e; font-family: Consolas;")
        layout.addWidget(self.log, 1)
        self.log_lines = []

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

    def _refresh_logs(self):
        selected = self.log_filter.currentText()
        self.log.setPlainText("\n".join(line for level, line in self.log_lines if selected == "全部日志" or selected == level))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

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

    def _manual_floor_changed(self):
        floor = self.floor_combo.currentData()
        if floor and floor != self.floor_var.get():
            self.floor_var.set(floor)
            self._floor_changed()

    def _manual_layout_changed(self, text):
        if (text and text != self.layout_var.get()
                and not getattr(self, '_manual_layout_confirmation', False)):
            self.layout_var.set(text)
            self._layout_changed()

    def _route_changed(self, _label):
        if self.settings.get('registration_backend') == 'template':
            variant_id = self.route_combo.currentData()
            if not variant_id or variant_id == selected_template_variant(self.settings, self.profile.key):
                return
            self.settings.setdefault('template_route_versions', {})[self.profile.key] = variant_id
            save_settings(self.settings, self.settings_path)
            self.map_generation += 1
            self.map_probe_pending = False
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
        variant_id = self.route_combo.currentData()
        if not variant_id:
            return
        live = self.live_projection
        reference = self.reference_context
        previous = live[0] if live is not None else self.last_result
        try:
            self.catalog.select_version_for_all(self.profile, variant_id, self.settings["route_versions"])
        except ValueError as exc:
            self._set_status(str(exc), AMBER)
            self._update_assets()
            return
        save_settings(self.settings, self.settings_path)
        self.map_generation += 1
        self.map_probe_pending = False
        self.hide_overlay()
        self._clear_alignment_preview()
        self._update_assets()
        if reference is not None and reference[0] == self.map_generation - 1:
            self.reference_context = (self.map_generation, *reference[1:])
            self._refresh_reference_map()
            self.reference_suspended = True
        if (previous is None or not previous.accepted or previous.profile_key != self.profile.key
                or previous.floor != self.floor_var.get() or previous.selected_layout != self.layout_var.get()):
            self._set_status("地图版本已切换，下一次配准将使用新版本", ACCENT)
            return
        try:
            switched = switch_result_version(previous, catalog=self.catalog, profile=self.profile,
                variant_id=variant_id, opacity=float(self.settings["opacity"]),
                asset_cache=self.asset_cache, prepared_cache=self.prepared_cache)
            if not switched.accepted:
                self._set_status("新版本未通过投影复核：" + "；".join(switched.reasons), AMBER)
                return
            self.last_result = switched
            self._set_preview_kind("projection", switched.projection, f"{self.route_combo.currentText()} · 实时预览")
            if getattr(self, 'map_keys', None) is not None:
                self._handle_map_key()
            if (live is not None and live[0] is previous and live[2] == self.map_generation-1
                    and self.map_state.is_open and self.show_overlay_check.isChecked()):
                roi = live[1]
                self.overlay.show(switched.layer, roi[0], roi[1])
                self.live_projection = (switched, roi, self.map_generation)
                self.overlay_var.set(f"投影显示中 · {self.route_combo.currentText()}")
            self._set_status(f"已切换为{self.route_combo.currentText()} · {switched.timings_ms['version_switch']:.0f} ms", ACCENT)
        except (ValueError, RuntimeError, OSError) as exc:
            self._set_status(f"版本切换失败：{exc}", AMBER)

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
            if self.settings.get('registration_backend') == 'template':
                from src.registration.template_matching.direct import load_original
                variant = selected_template_variant(self.settings, self.profile.key)
                image = load_original(self.profile.key, layout, variant)[3]
                title = f"{layout} · {TEMPLATE_VERSION_NAMES[variant]} · 三层完整原图"
                self.template_panel.set_image(image, title)
                self.preview_images['template'] = (image, title)
                return
            path = self.catalog.clean_template(self.profile, layout, self.floor_var.get())
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.template_panel.set_image(image, f"{layout} · {FLOOR_NAMES[self.floor_var.get()]}")
            self.preview_images["template"] = (image, f"参考底图 · {layout} / {FLOOR_NAMES[self.floor_var.get()]}")
        except Exception as exc:
            self.preview_images.pop("template", None)
            self.template_panel.set_image(None, str(exc))

    def _refresh_route_preview(self):
        layout = self.layout_var.get()
        if layout == AUTO:
            self.route_panel.set_image(None, "先确认布局")
            return
        if self.settings.get('registration_backend') == 'template':
            try:
                from src.registration.template_matching.direct import load_original
                variant = selected_template_variant(self.settings, self.profile.key)
                self.route_panel.set_image(load_original(self.profile.key, layout, variant)[3],
                                          f"{layout} · {TEMPLATE_VERSION_NAMES[variant]} · 三层完整原图")
            except (OSError, ValueError, KeyError, cv2.error) as exc:
                self.route_panel.set_image(None, f"完整原图不可用：{exc}")
            return
        variant = self.catalog.selected_variant(self.profile, layout, self.settings["route_versions"])
        if variant is None:
            self.route_panel.set_image(None, "该布局暂无路线")
            return
        try:
            template_path = self.catalog.clean_template(self.profile, layout, self.floor_var.get())
            base = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            asset = load_route_overlay_asset(variant.calibration_path, self.floor_var.get())
            height, width = base.shape[:2]
            layer = cv2.warpPerspective(asset.rgba, asset.asset_to_clean, (width, height))
            alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
            composite = (base.astype(np.float32) * (1.0 - alpha) +
                         layer[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
            self.route_panel.set_image(composite, f"{layout} / {variant.label}")
        except Exception as exc:
            self.route_panel.set_image(None, f"路线预览不可用：{exc}")

    def _refresh_status(self, *_args):
        if not hasattr(self, "status_rows"):
            return
        self.current_map_label.setText(f"当前地图       {self.profile.map_name}")
        self.current_difficulty_label.setText(f"当前难度       {self.profile.difficulty_name}")
        self.current_layout_label.setText(f"当前布局       {self.locked_layout or '待识别'}")
        confidence = (self.last_decision or {}).get("confidence")
        score = (self.last_decision or {}).get('score')
        self.match_label.setText(f"匹配分数       {score:.4f}" if score is not None else
                                f"匹配度          {confidence:.1%}" if confidence is not None else "匹配度          —")
        self.status_rows["route"].setText(self.asset_var.get())
        self.status_rows["projection"].setText(self.overlay_var.get())
        self.status_rows["capture"].setText("已截图" if self.last_result is not None else "等待截图")
        self.status_rows["map_presence"].setText(self.map_presence_text)
        self.status_rows["map_presence_ms"].setText(
            f"{self.map_presence_elapsed_ms:.0f} ms" if self.map_presence_elapsed_ms is not None else "—")
        self.status_rows["recognition"].setText("已锁定" if self.locked_layout else "待识别")
        timing = self.last_result.timings_ms if self.last_result is not None else {}
        for key, source in (("capture_ms", "capture"), ("floor_ms", "floor_detection"),
                            ("recognition_ms", "classification"), ("registration_ms", "registration"),
                            ("projection_ms", "projection")):
            value = timing.get(source)
            self.status_rows[key].setText(f"{value:.0f} ms" if value is not None else "—")
        if self.settings.get('registration_backend') == 'template':
            self.status_rows['floor_ms'].setText('已跳过')
            self.status_rows['map_presence_ms'].setText(
                f'已跳过；截图等待 {self.map_presence_elapsed_ms:.0f} ms' if self.map_presence_elapsed_ms is not None else '已跳过')
        self.metric_label.setText(self.metric_var.get())
        self.path_label.setText(self.path_var.get())

    def _set_status(self, text: str, color: str = ACCENT):
        self.status_var.set(text)
        self.state_label.configure(text=text, fg=color)
        self._refresh_status()

    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = "错误" if "失败" in text or "错误" in text else "警告" if "未通过" in text or "忽略" in text else "信息"
        self.log_lines.append((level, f"{stamp}   {level:<2}   {text}"))
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
        self._start_model_load()

        self._start_layout_retrieval()

    def _floor_changed(self, _event=None):
        self.map_generation += 1
        self.settings["floor"] = self.floor_var.get()
        self.settings["auto_floor"] = False
        save_settings(self.settings, self.settings_path)
        self.floor_detect_var.set(f"手动楼层：{FLOOR_NAMES[self.floor_var.get()]}")
        self._update_assets()
        self.hide_overlay()

    def _layout_changed(self, _event=None):
        self.map_generation += 1
        self._pending_confirmation = None
        self.session_id = uuid.uuid4().hex[:12]
        self.map_probe_pending = False
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
                self.registration_priors.clear()
                self.registration_soft_priors.clear()
                self.registration_evidence.clear()
                self.last_decision = None
                self.locked_layout = selected
                self.decision_label.configure(text=f"已手动确认 {selected}", fg=ACCENT)
                self._set_status(f"布局已确认：{selected}；打开地图后配准", ACCENT)
                self._preload_locked_layout(selected)
            else:
                self.layout_var.set(self.locked_layout or AUTO)
                self._set_status("已取消手动选择，保留原布局" if valid else "地图状态已变化，请重新选择布局", MUTED)
        self._update_assets()

    def _update_assets(self):
        layouts = self.catalog.layouts(self.profile)
        layout = self.layout_var.get()
        if layout != AUTO:
            variant = self.catalog.selected_variant(self.profile, layout, self.settings["route_versions"])
            if variant:
                state = variant.floor_status.get(self.floor_var.get(), "—")
                self.asset_var.set(f"{layout} / {variant.label} / {FLOOR_NAMES[self.floor_var.get()]} · {state}")
            else:
                self.asset_var.set(f"{layout} 暂无路线版本")
        else:
            self.asset_var.set(f"已收录 {len(layouts)} 个布局")
        if hasattr(self, "route_combo"):
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentText(layout)
            self.layout_combo.blockSignals(False)
            choice_layout = layout if layout != AUTO else (layouts[0] if layouts else "")
            variants = self.catalog.variants(self.profile, choice_layout) if choice_layout else []
            selected = self.catalog.selected_variant(self.profile, choice_layout, self.settings["route_versions"]) if variants else None
            self.route_combo.blockSignals(True)
            self.route_combo.clear()
            for item in variants:
                self.route_combo.addItem(item.label, item.variant_id)
            if selected:
                self.route_combo.setCurrentIndex(self.route_combo.findData(selected.variant_id))
            template = self.settings.get('registration_backend') == 'template'
            if template:
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
            self.floor_combo.setEnabled(not template)
            for index, floor in enumerate(FLOORS):
                self.floor_combo.setItemText(index, '不分楼层' if template else FLOOR_NAMES[floor])
            self.route_combo.blockSignals(False)
            self._refresh_template()
            self._refresh_route_preview()
        self._refresh_status()

    def _start_model_load(self):
        if self.settings.get('layout_recognition_backend', 'cnn') != 'cnn':
            return
        profile = self.profile
        if profile.key in self.model_cache:
            self._log(f"模型就绪：{profile.map_name} / {profile.difficulty_name}")
            return
        def worker():
            try:
                bundle = load_classifier(profile.model_path, device="cpu")
                self.results.put(("model", profile.key, bundle, None))
            except Exception as exc:
                self.results.put(("model", profile.key, None, str(exc)))
        threading.Thread(target=worker, name="load-layout-model", daemon=True).start()

    def _layout_retriever(self):
        cache = getattr(self, 'layout_retrievers', None)
        if cache is None:
            cache = self.layout_retrievers = {}
        if self.profile.key not in cache:
            cache[self.profile.key] = LayoutRetriever(self.profile.clean_root,
                cache_root=CACHE_ROOT / 'layout_retrieval' / self.profile.map_id / self.profile.difficulty_id)
        return cache[self.profile.key]

    def _start_layout_retrieval(self):
        if self.settings.get('layout_recognition_backend', 'cnn') != 'icon_structure':
            return
        profile_key, retriever = self.profile.key, self._layout_retriever()
        def worker():
            try:
                fingerprint = retriever.prepare()
                self.results.put(('retrieval_index', profile_key, fingerprint, None))
            except Exception as exc:
                self.results.put(('retrieval_index', profile_key, None, str(exc)))
        threading.Thread(target=worker, name='prepare-layout-retrieval', daemon=True).start()

    def _start_registration_backend(self):
        backend = self.settings.get("registration_backend", "auto")
        def worker():
            info = initialize_registration(backend)
            if backend == "auto" and not self.closed:
                with self.template_prepare_lock:
                    prepared = list(self.prepared_cache.values())
                for template in prepared:
                    if self.closed or self.settings.get("registration_backend", "auto") != backend:
                        break
                    info = prewarm_template(template, backend)
            self.results.put(("registration_ready", backend, info))
        threading.Thread(target=worker, name="initialize-registration", daemon=True).start()

    def _preload_locked_layout(self, layout: str) -> None:
        """Prepare all three floor templates off the UI thread after layout lock."""
        if self.settings.get('registration_backend') == 'template':
            # Preview/source cache already loads the original; no clean templates are needed.
            return
        profile = self.profile
        cache_root = (CACHE_ROOT / 'route_projection_templates'
                      / profile.map_id / profile.difficulty_id)

        def worker() -> None:
            for floor in FLOORS:
                if self.closed or self.locked_layout != layout or self.profile.key != profile.key:
                    return
                key = (profile.key, layout, floor)
                try:
                    with self.template_prepare_lock:
                        if key in self.prepared_cache:
                            continue
                        template = self.catalog.clean_template(profile, layout, floor)
                        if not template.is_file():
                            continue
                        prepared = prepare_registration_template(
                            layout, floor, template, cache_root)
                        self.prepared_cache[key] = prepared
                    self.results.put(("preloaded_template", key, prepared, None))
                    prewarm_template(prepared, self.settings.get("registration_backend", "auto"))
                except Exception as exc:
                    self.results.put(("preloaded_template", key, None, str(exc)))

        threading.Thread(target=worker, name=f"preload-{layout}", daemon=True).start()

    def apply_settings(self, settings: dict, *, save: bool = True):
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        backend_changed = self.settings.get("registration_backend", "auto") != settings.get("registration_backend", "auto")
        recognition_changed = (self.settings.get('layout_recognition_backend', 'cnn') !=
                               settings.get('layout_recognition_backend', 'cnn'))
        previous_keys = (self.settings.get("recognize_hotkey"), self.settings.get("project_hotkey"))
        if (self.settings.get("capture_roi") != settings.get("capture_roi")
                or self.settings.get("capture_reference_size") != settings.get("capture_reference_size")):
            self.registration_priors.clear()
            self.registration_soft_priors.clear()
            self.registration_evidence.clear()
            self.session_id = uuid.uuid4().hex[:12]
        if save:
            self.map_generation += 1
            self.map_probe_pending = False
        self.settings = settings
        if recognition_changed:
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
            self._start_model_load()
            self._start_layout_retrieval()
        method = settings.get('layout_recognition_backend', 'cnn')
        self.recognition_method_label.setText('识别方法：' + BACKEND_LABELS.get(method, 'CNN'))
        self.ranking_heading.setText('布局匹配分数排名' if method == 'icon_structure' else '布局置信度排名')
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(settings['map_key'], settings['game_window_titles'],
                                    suspended=self.settings_dialog is not None, discard=save)
        if settings.get('registration_backend') == 'template':
            self.floor_monitor_active = False
            self.opening_floor_sample = None
        if backend_changed:
            self._start_registration_backend()
        self.map_state.key_was_down = map_key_down(settings["map_key"])
        self.map_state.escape_was_down = escape_key_down()
        self.floor_var.set(settings["floor"])
        self.floor_combo.blockSignals(True)
        self.floor_combo.setCurrentIndex(FLOORS.index(settings["floor"]))
        self.floor_combo.blockSignals(False)
        if settings.get('registration_backend') == 'template':
            self.floor_detect_var.set('定尺度模板匹配：完整原图匹配，不识别楼层')
        elif not settings["auto_floor"]:
            self.floor_detect_var.set(f"手动楼层：{FLOOR_NAMES[settings['floor']]}")
        elif save:
            self.floor_detect_var.set("楼层标识：等待打开地图")
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
        if self.settings_dialog is not None:
            self.pages.setCurrentWidget(self.settings_dialog)
            return
        self._handle_map_key()
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'], suspended=True)
        self._pending_confirmation = None
        self.map_generation += 1
        self.map_probe_pending = False
        self.map_presence_elapsed_ms = None
        self.map_presence_started_at = None
        self.hide_overlay()
        self.settings["opacity"] = load_settings(self.catalog, self.settings_path)["opacity"]
        self.hotkeys_suspended = True
        if self.hotkeys:
            self.hotkeys.stop()
            self.hotkeys = None
        self.settings_dialog = SettingsDialog(self)
        self.pages.addWidget(self.settings_dialog)
        self.pages.setCurrentWidget(self.settings_dialog)

    def _leave_settings(self, page):
        if self.settings_dialog is not page:
            return
        self.pages.setCurrentWidget(self.home_page)
        self.pages.removeWidget(page)
        self.settings_dialog = None
        page.deleteLater()
        if not self.closed:
            self.hotkeys_suspended = False
            self.apply_settings(self.settings, save=False)

    def reset_map_state(self):
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.configure(self.settings['map_key'], self.settings['game_window_titles'],
                                    suspended=self.settings_dialog is not None, discard=True)
        self._pending_auto_projection = None
        self.map_state.reset(map_key_down(self.settings["map_key"]))
        self.map_generation += 1
        self.map_probe_pending = False
        self.clear_layout_lock()
        self.hide_overlay()
        self._set_status("地图状态已复位为关闭", MUTED)
        self._log("地图状态复位为关闭；请确认游戏地图也已关闭")

    def clear_layout_lock(self):
        self.locked_layout = None
        self.last_decision = None
        self.registration_priors.clear()
        self.registration_soft_priors.clear()
        self.registration_evidence.clear()
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
        self.preview_images[kind] = (bgr, caption)
        if kind == "capture":
            self.capture_panel.set_image(bgr, caption)
        if kind == "capture" and getattr(self, "selected_preview_tab", 0) == 0:
            self.overlay_panel.set_image(bgr, caption)
        elif kind in ("alignment", "projection"):
            self._select_preview_tab(2 if kind == "alignment" else 3)

    def _update_ranking(self, result: WorkflowResult):
        decision = result.decision or self.last_decision
        if result.decision:
            self.last_decision = result.decision
        ranking = decision["ranking"] if decision else []
        for index, (name, pct, _bar) in enumerate(self.rank_widgets):
            item = ranking[index] if index < len(ranking) else None
            name.configure(text=item["layout"] if item else "—", fg=INK if index == 0 else MUTED)
            pct.configure(text=(f"{item['score']:.4f}" if 'score' in item else f"{item['confidence']:.1%}")
                          if item else "—", fg=GREEN if index == 0 else MUTED)
        if self.locked_layout and result.decision is None:
            self.decision_label.configure(text=f"已锁定 {self.locked_layout} · 本次仅配准", fg=GREEN)
        elif decision:
            accepted = decision["accepted"]
            structure_confirmed = bool(result.accepted and result.registration)
            description = (f"结构验证确认：{result.selected_layout}" if structure_confirmed and not accepted else
                           f"可判断：{decision['best_label']}" if accepted else
                           f"仅有候选：{decision['best_label']}")
            detail = (f"匹配分数 {decision['score']:.4f} · 分差 {decision['margin']:.4f} · 请核对入口图"
                      if 'score' in decision else
                      f"置信度 {decision['confidence']:.1%} · 前两名差距 {decision['margin']:.1%}")
            self.decision_label.configure(text=f"{description} · {detail}",
                                          fg=GREEN if accepted or structure_confirmed else AMBER)
        else:
            text = '没有有效布局候选' if getattr(self, 'settings', {}).get('layout_recognition_backend', 'cnn') == 'icon_structure' else '模型尚未就绪'
            self.decision_label.configure(text=text, fg=AMBER)

    def request_work(self, mode: str, *, auto: bool) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if mode == "recognize" and self._pending_confirmation is not None:
            self._set_status("请先确认本次布局识别结果", AMBER)
            return
        if self.busy:
            self._set_status("上一次配准仍在处理，请稍候", AMBER)
            return
        fixed_layout = self.locked_layout if mode == "project" else None
        if mode == "project" and fixed_layout is None:
            self._set_status("请先按 F8 识别并确认布局，再打开地图投影", AMBER)
            RouteProjectionApp._map_toast(self, "布局尚未确认", "请按 F8 识别并确认布局，然后自动配准路线。", success=False)
            return
        recognition_backend = self.settings.get('layout_recognition_backend', 'cnn')
        if mode == "recognize" and recognition_backend == 'cnn' and self.profile.key not in self.model_cache:
            self._set_status("布局模型正在加载，请稍候", AMBER)
            return
        if mode == "recognize":
            self.hide_overlay()
            self._clear_alignment_preview()
            self.floor_monitor_active = False
            self.registration_priors.clear()
            self.registration_soft_priors.clear()
            self.registration_evidence.clear()
            self.session_id = uuid.uuid4().hex[:12]
        self.busy = True
        template_direct = self.settings.get('registration_backend') == 'template'
        fast_display = mode == 'project' and template_direct and getattr(self, '_work_bridge', None) is not None
        if mode == "project":
            self.hide_overlay()
            self.reference_context = None
            RouteProjectionApp._map_toast(self, "正在识别地图", f"已锁定 {fixed_layout}，" +
                ("正在匹配三层完整原图…" if template_direct else "正在识别楼层并配准路线…"), success=True)
            self.floor_monitor_active = self.map_state.is_open and not template_direct
            self.floor_probe_needs_projection = False
            self.floor_probe_next = time.monotonic() + 0.6
        generation = self.map_generation
        session_id = self.session_id
        profile = self.profile
        roi = list(self.settings["capture_roi"])
        reference_size = list(self.settings["capture_reference_size"])
        window_titles = list(self.settings["game_window_titles"])
        fallback_floor = 'all' if template_direct else self.floor_var.get()
        auto_floor = bool(self.settings["auto_floor"]) and not template_direct
        opening_floor = getattr(self, "opening_floor_sample", None)
        self.opening_floor_sample = None  # Single-use, even if the worker is cancelled.
        route_versions = json.loads(json.dumps(self.settings["route_versions"]))
        template_variant_id = selected_template_variant(self.settings, profile.key)
        opacity = float(self.settings["opacity"])
        registration_backend = self.settings.get("registration_backend", "auto")
        retriever = self._layout_retriever() if mode == 'recognize' and recognition_backend == 'icon_structure' else None
        work_requested_at = time.perf_counter()
        key_started_at = getattr(self, 'map_presence_started_at', None) if auto else None
        bundle = self.model_cache.get(profile.key) if fixed_layout is None else None
        self._set_status("正在截图与识别…" if mode == "recognize" else
                         "正在截图与配准已锁定布局…" if fixed_layout else "正在截图、识别并配准…")
        def worker() -> None:
            delivered = False
            try:
                started = time.perf_counter()
                client = foreground_game_client(window_titles)
                recognition_window = recognition_window_identity(window_titles) if retriever is not None else None
                screen_region = game_capture_region(client, roi, reference_size)
                ensure_region_on_screen(screen_region)
                screen_roi = screen_region.as_list()
                floor_detection = None
                floor_image = None
                floor_region = None
                floor_ms = 0.0
                floor_reused = False
                if mode == "project" and auto_floor:
                    floor_started = time.perf_counter()
                    if opening_floor is not None and opening_floor.reusable(
                            client=client.as_list(), reference_size=reference_size,
                            generation=generation, session_id=session_id, now=floor_started):
                        floor_detection, floor_image, floor_region = (
                            opening_floor.detection, opening_floor.image, opening_floor.region)
                        floor_reused = True
                    else:
                        floor_detection, floor_image, floor_region = capture_floor_indicator(client, reference_size)
                    floor_ms = (time.perf_counter() - floor_started) * 1000.0
                screenshot = capture_bgr(screen_roi)
                if recognition_window is not None and recognition_window_identity(window_titles) != recognition_window:
                    raise RuntimeError('截图期间游戏窗口发生变化，请重新识别')
                initial_floor_detection = floor_detection
                if floor_detection is not None:
                    floor_started = time.perf_counter()
                    verified, verified_image, verified_region = capture_floor_indicator(client, reference_size)
                    floor_ms += (time.perf_counter() - floor_started) * 1000.0
                    floor_image, floor_region = verified_image, verified_region
                    if not (floor_detection.accepted and verified.accepted
                            and floor_detection.floor == verified.floor):
                        floor_detection = FloorDetection(
                            None, verified.scores, verified.score_margin, verified.highlight_floor,
                            verified.highlight_margin, "楼层在地图截图期间未保持稳定",
                        )
                    else:
                        floor_detection = verified
                capture_ms = max(0.0, (time.perf_counter() - started) * 1000.0 - floor_ms)
                if floor_detection is not None and not floor_detection.accepted:
                    result = WorkflowResult(
                        screenshot=screenshot, decision=None, selected_layout=fixed_layout,
                        variant_id=None, variant_source=None, registration=None,
                        overlay_validation=None, layer=None, projection=None, accepted=False,
                        reasons=[f"楼层标识无法确认：{floor_detection.reason}"],
                        timings_ms={}, profile_key=profile.key, floor="unknown",
                        template_path=None, asset_path=None, calibration_path=None,
                    )
                else:
                    if mode == "project" and not template_direct:
                        self.results.put(("reference_ready", generation, screen_roi,
                                          floor_detection.floor if floor_detection else fallback_floor,
                                          profile.key, fixed_layout))
                    result = process_screenshot(
                        screenshot, catalog=self.catalog, profile=profile, model_bundle=bundle,
                        floor=floor_detection.floor if floor_detection else fallback_floor,
                        layout_override=fixed_layout,
                        route_versions=route_versions, opacity=opacity,
                        prepared_cache=self.prepared_cache, asset_cache=self.asset_cache,
                        recognize_only=mode == "recognize",
                        registration_priors=self.registration_priors,
                        registration_soft_priors=self.registration_soft_priors,
                        registration_evidence=self.registration_evidence,
                        template_prepare_lock=self.template_prepare_lock,
                        registration_backend=registration_backend,
                        template_variant_id=template_variant_id,
                        layout_recognition_backend=recognition_backend,
                        **({'layout_retriever': retriever,
                            'recognition_reference_size': tuple(roi[2:])} if retriever is not None else {}),
                        **({'defer_projection_preview':True} if fast_display else {}))
                if floor_detection is not None:
                    result.floor_detection = {**floor_detection.diagnostics(), "screen_roi": floor_region}
                    result.floor_detection["before_map_capture"] = initial_floor_detection.diagnostics()
                    result.floor_detection["before_map_capture_reused_from_open_probe"] = floor_reused
                    result.floor_indicator_image = floor_image
                    result.timings_ms["floor_detection"] = floor_ms
                result.timings_ms["capture"] = capture_ms
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
                    result.timings_ms['preview_finished_from_request'] = (time.perf_counter()-work_requested_at)*1000
                    result.timings_ms['display_acknowledged'] = acknowledged
                    if not self.closed:
                        self._work_bridge.ready.emit(('preview', generation, session_id, ready_result, result))
                else:
                    self.results.put(event)
                if mode == "project":
                    try:
                        saved = save_workflow_result(result)
                        self.results.put(("saved", result, str(saved), None))
                    except Exception as exc:
                        self.results.put(("saved", result, None, str(exc)))
            except Exception as exc:
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
                    self.results.put(event)
        threading.Thread(target=worker, name=f"route-{mode}", daemon=True).start()

    @pyqtSlot(object)
    def _receive_work_event(self, payload):
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
            feedback = {'result_delivery':(received_at-meta['posted_at'])*1000}
            if (result is not None and result.accepted and result.layer is not None
                    and self.show_overlay_check.isChecked()
                    and game_window_is_foreground(self.settings['game_window_titles'])):
                submit_started = time.perf_counter()
                self.overlay.show(result.layer, roi[0], roi[1])
                submitted = time.perf_counter()
                presented = True
                self.live_projection = (result, list(roi), generation)
                feedback['overlay_submit'] = (submitted-submit_started)*1000
                feedback['request_to_overlay_submit'] = (submitted-meta['requested_at'])*1000
                if meta.get('key_started_at') is not None:
                    feedback['key_to_overlay_submit'] = (submitted-meta['key_started_at'])*1000
            # Publish the feedback atomically, then unblock background preview computation.
            meta['timings'] = feedback
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
                and backend == self.settings.get('registration_backend', 'auto')
                and layout == self.locked_layout):
            RouteProjectionApp._auto_if_open(self, generation)

    def _apply_result(self, mode: str, auto: bool, generation: int, roi: list[int],
                      result: WorkflowResult | None, error: str | None, *, overlay_presented=False) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        self.busy = False
        if mode == 'recognize' and result is not None:
            if getattr(result, 'layout_recognition_backend', 'cnn') != getattr(self, 'settings', {}).get('layout_recognition_backend', 'cnn'):
                return
            if getattr(result, 'layout_recognition_backend', 'cnn') == 'icon_structure':
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
            self.registration_evidence.clear()
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
            if result.floor_detection:
                detection = result.floor_detection
                scores = detection["scores"]
                best_score = max(scores.values())
                if detection["accepted"] and result.floor in FLOORS:
                    self.floor_detect_var.set(
                        f"楼层标识：{FLOOR_NAMES.get(result.floor, '三层完整原图')} · 相似度 {best_score:.1%} · 差距 {detection['score_margin']:.1%}")
                    if generation == self.map_generation and self.floor_var.get() != result.floor:
                        self.floor_var.set(result.floor)
                        self.floor_combo.blockSignals(True)
                        self.floor_combo.setCurrentIndex(FLOORS.index(result.floor))
                        self.floor_combo.blockSignals(False)
                        self.settings["floor"] = result.floor
                        self._update_assets()
                        self._log(f"自动切换楼层：{FLOOR_NAMES.get(result.floor, '三层完整原图')}")
                else:
                    self.floor_detect_var.set(f"楼层标识：无法确认 · {detection['reason']}")
                self.floor_probe_next = time.monotonic() + 0.25
            if (generation == self.map_generation and result.accepted and result.selected_layout
                    and result.registration and result.registration.screenshot_to_clean is not None
                    and result.registration.observation_bbox is not None
                    and result.registration.observation_mask is not None
                    and result.registration.validation.get('projection_scope') != 'local'
                    and result.registration.validation.get('trusted_history_eligible', True)
                    and result.registration.validation.get("scale_only_relaxed")):
                evidence_key = (result.profile_key, result.selected_layout, result.floor)
                self.registration_soft_priors[evidence_key] = SoftPoseHint(
                    result.registration.screenshot_to_clean.copy(),
                    result.registration.observation_bbox,
                    result.registration.observation_mask.copy(),
                )
            if (generation == self.map_generation and result.accepted and result.selected_layout and result.registration and
                    result.registration.screenshot_to_clean is not None and
                    result.registration.validation.get('trusted_history_eligible', True) and
                    not result.registration.validation.get("route_corroborated") and
                    result.registration.validation.get("projection_scope") != "local" and
                    not result.registration.validation.get("scale_only_relaxed")):
                evidence_key = (result.profile_key, result.selected_layout, result.floor)
                self.registration_soft_priors.pop(evidence_key, None)
                self.registration_priors[evidence_key] = (
                    result.registration.screenshot_to_clean.copy())
                prepared = self.prepared_cache.get(evidence_key)
                if prepared is not None:
                    evidence = self.registration_evidence.get(evidence_key)
                    if evidence is None or not evidence.matches(prepared):
                        evidence = TemporalEvidence.empty(prepared)
                        self.registration_evidence[evidence_key] = evidence
                    if evidence.add_registration(result.registration, prepared,
                                                 source=result.run_directory or
                                                 f"capture:{datetime.now().isoformat(timespec='milliseconds')}"):
                        update = evidence.updates[-1]
                        self._log(f"结构记忆：{result.selected_layout}/{result.floor} 新增 {update.new_structure_pixels} 像素；"
                                  f"累计 {len(evidence.updates)} 次高可信观测")
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
                    automatic = (getattr(self, "settings", {}).get("skip_layout_confirmation", False)
                                 and result.decision.get("accepted", False))
                    RouteProjectionApp._show_recognition_toast(
                        self, "布局识别成功", (f"识别为 {result.selected_layout}，将自动锁定最高置信度布局。" if automatic
                                              else f"首选候选为 {result.selected_layout}，请对照入口图选择布局。"),
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
                elif result.registration and result.registration.alignment_preview is not None:
                    self._set_preview_kind("alignment", result.registration.alignment_preview,
                                           f"{'配准叠图' if result.accepted else '候选未通过'} · {result.selected_layout} / {FLOOR_NAMES.get(result.floor, '三层完整原图')}")
                self.metric_var.set(format_projection_metrics(result))
                self._log(self.metric_var.get())
                execution = result.registration.validation.get('registration_backend', {}) if result.registration else {}
                if execution:
                    name = {'cuda': '多尺度结构配准（GPU）', 'cpu': '多尺度结构配准（CPU）', 'template': '定尺度模板匹配'}.get(execution['effective_backend'], execution['effective_backend'])
                    reason = execution.get('fallback_reason')
                    self._log(f"配准后端：{name}" + (f" · 回退原因：{reason}" if reason else ""))
                if getattr(self, 'map_keys', None) is not None:
                    self._handle_map_key()
                stale = generation != self.map_generation or not self.map_state.is_open
                if result.accepted and result.layer is not None and not stale:
                    audit = result.registration.validation.get('visible_acceptance', {}) if result.registration else {}
                    if audit.get('basis') == 'single_category_geometry_support':
                        value = result.registration.validation['raster']['semantic_precision']
                        self._log(f"联合证据补充接受：整片类别一致率 {value:.2%}，"
                                  "常规门槛 72%，补充下限 68%；其余检查及严格几何证据通过")
                    if self.show_overlay_check.isChecked():
                        if not overlay_presented:
                            self.overlay.show(result.layer, roi[0], roi[1])
                        self.live_projection = (result, list(roi), generation)
                    percent = opacity_to_percent(float(self.settings["opacity"]))
                    self.overlay_var.set(("投影显示中" if self.show_overlay_check.isChecked() else "投影预览完成，显示已关闭") +
                                         f" · {result.selected_layout} / {FLOOR_NAMES.get(result.floor, '三层完整原图')} / "
                                         f"{result.variant_id} · 不透明度 {percent}%")
                    if result.registration and result.registration.validation.get('registration_algorithm')=='fixed_scale_template':
                        self._set_status("模板匹配候选已投影，请核对位置", AMBER)
                    elif result.registration and result.registration.validation.get('projection_scope')=='local':
                        self._set_status("完整路线已显示，继续探索可改善缩放精度", AMBER)
                    elif result.registration and result.registration.validation.get("scale_only_relaxed"):
                        disagreement = (result.overlay_validation or {}).get("route_pose_disagreement") or {}
                        p95 = disagreement.get("p95_px")
                        detail = (f"；远处路线候选偏移 P95 {p95:.1f}px" if p95 is not None
                                  else "；远处路线仍需核对")
                        self._set_status("同位置缩放限制已放宽，路线已投影" + detail, AMBER)
                    else:
                        self._set_status("路线复核通过，已投影" if result.registration and
                                         result.registration.validation.get("route_corroborated") else
                                         "配准通过，路线已投影", ACCENT)
                elif stale:
                    self._set_status("地图已关闭，本次结果仅保留诊断", MUTED)
                else:
                    self._set_status("投影未通过校验：" + "；".join(result.reasons[:2]), AMBER)
                if not stale:
                    RouteProjectionApp._finish_map_feedback(self, generation, roi, result)
                self._log(f"{result.selected_layout or '未知布局'} / {result.floor} · "
                          f"{'通过' if result.accepted else '未通过'} · {timing.get('total', 0):.0f} ms")

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
            {"layout": result.selected_layout, "confidence": decision["confidence"]}
        ], key=lambda item: item.get('score', item.get('confidence', 0.)), reverse=True)[:2]
        settings = getattr(self, "settings", {})
        if settings.get("skip_layout_confirmation", False) and decision.get("accepted", False):
            selected_layout, skip_confirmation = candidates[0]["layout"], False
        else:
            selected_layout, skip_confirmation = messagebox.confirm_layout(
                candidates,
                self.profile.entrance_root if getattr(self, "profile", None) else None,
                screen_roi=roi,
            )
        if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3]):
            self._set_status("地图状态已变化；请按 F8 重新识别", AMBER)
            return
        if selected_layout:
            if skip_confirmation:
                updated = {**settings, "skip_layout_confirmation": True}
                try:
                    save_settings(updated, self.settings_path)
                except OSError as exc:
                    self._log(f"布局确认偏好未保存：{exc}")
                    messagebox.showerror("设置未保存", "无法保存“下次不再提示”，下次仍会显示确认框。", parent=self)
                    # The error dialog runs an event loop too; reject stale results.
                    if not RouteProjectionApp._recognition_can_confirm(self, generation, pending[2], pending[3]):
                        return
                else:
                    self.settings = updated
            if self.locked_layout != selected_layout:
                self.registration_priors.clear()
                self.registration_soft_priors.clear()
                self.registration_evidence.clear()
            self.locked_layout = selected_layout
            self.layout_var.set(selected_layout)
            self._update_assets()
            self.decision_label.configure(text=f"已确认锁定 {selected_layout}", fg=ACCENT)
            self._set_status(f"布局已锁定：{selected_layout}；探索门口后按地图键投影", ACCENT)
            self._log(f"F8 识别并确认锁定：{selected_layout}")
            self._preload_locked_layout(selected_layout)
        else:
            self.layout_var.set(self.locked_layout or AUTO)
            self._update_assets()
            self.decision_label.configure(text=(f"仍锁定 {self.locked_layout}" if self.locked_layout else
                                                "识别结果未确认，布局尚未锁定"), fg=AMBER)
            self._set_status("已取消锁定；可再次按 F8 识别", MUTED)

    def hide_overlay(self) -> None:
        self.live_projection = None
        if hasattr(self, "reference_overlay"):
            self.reference_overlay.hide()
        if hasattr(self, "overlay") and self.overlay.visible:
            self.overlay.hide()
            self.overlay_var.set("投影已隐藏")

    def _map_toast(self, title: str, detail: str, *, success: bool, roi=None) -> None:
        if (getattr(self, "closed", False) or not getattr(getattr(self, "map_state", None), "is_open", False)
                or getattr(self, "settings_dialog", None) is not None):
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
                or context[4] != self.locked_layout or self.settings_dialog is not None
                or not game_window_is_foreground(self.settings["game_window_titles"])):
            self.reference_overlay.hide()
            return
        _, roi, floor, _, layout = context
        try:
            if self.settings.get('registration_backend') == 'template':
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
            variant = self.catalog.selected_variant(self.profile, layout, self.settings["route_versions"])
            if variant is None:
                raise ValueError("当前布局暂无路线资产")
            key = (self.profile.key, layout, floor, variant.variant_id)
            image = self.reference_cache.get(key)
            if image is None:
                path = self.catalog.clean_template(self.profile, layout, floor)
                base = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                asset = load_route_overlay_asset(variant.calibration_path, floor)
                layer = cv2.warpPerspective(asset.rgba, asset.asset_to_clean, (base.shape[1], base.shape[0]))
                alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
                image = (base * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
                self.reference_cache[key] = image
            if getattr(self, 'map_keys', None) is not None:
                self._handle_map_key()
                if self.reference_context is not context or context[0] != self.map_generation or not self.map_state.is_open:
                    return
            self.reference_overlay.show_map(image, roi,
                f"{layout} · {FLOOR_NAMES[floor]} · 参考图 · {self.settings['reference_hotkey']} 隐藏", self.settings["reference_opacity"])
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
            if result.registration and result.registration.validation.get("projection_scope") == "local":
                detail += "继续探索可改善缩放精度。"
            template = bool(result.registration and result.registration.validation.get('registration_algorithm') == 'fixed_scale_template')
            RouteProjectionApp._map_toast(self, "模板匹配候选已投影" if template else "地图配准成功", detail, success=True, roi=roi)
        else:
            reason = result.reasons[0] if result.reasons else "可见地图信息不足"
            shown = getattr(getattr(self, "reference_overlay", None), "visible", False)
            detail = label + "：" + reason + (f"。左上角已显示参考路线，{self.settings['reference_hotkey']} 可隐藏。" if shown else "。请继续探索后重试。")
            RouteProjectionApp._map_toast(self, "地图暂未配准", detail, success=False, roi=roi)

    def _handle_reference_key(self):
        down = map_key_down(self.settings["reference_hotkey"])
        pressed = down and not self.reference_key_was_down
        self.reference_key_was_down = down
        if (not self.map_state.is_open or self.settings_dialog is not None or own_window_is_foreground()
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
                self._map_toast("参考路线待显示", "正在准备完整原图。" if self.settings.get('registration_backend') == 'template'
                                else "正在等待当前楼层确认，请保持地图打开。", success=True)
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
                    enabled = key.enabled and self.settings_dialog is None
                    if key.action == 'map':
                        event = self.map_state.observe(key.down, enabled=enabled)
                        closed_by_escape = False
                    else:
                        pending_escape = (key.down and not self.map_state.escape_was_down
                                          and enabled and self.map_probe_pending)
                        event = None
                        closed_by_escape = self.map_state.observe_escape(key.down, enabled=enabled) or pending_escape
                        if closed_by_escape:
                            self.map_state.is_open = False
                    if key.down:
                        self._log(f"地图按键事件 #{key.sequence}：{key.action} · "
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
        # Legacy direct-state callers/tests. The formal app always creates a listener;
        # a failed listener stops auto projection instead of silently polling again.
        RouteProjectionApp._poll_map_key(self)

    def _poll_map_key(self) -> None:
        down = map_key_down(self.settings["map_key"])
        escape_down = escape_key_down()
        game_focused = game_window_is_foreground(self.settings["game_window_titles"])
        enabled = self.settings_dialog is None and not own_window_is_foreground() and game_focused
        ignored_press = down and not self.map_state.key_was_down and not enabled
        pending_escape = (escape_down and not self.map_state.escape_was_down and enabled
                          and self.map_probe_pending)
        event = self.map_state.observe(down, enabled=enabled)
        closed_by_escape = self.map_state.observe_escape(escape_down, enabled=enabled) or pending_escape
        if closed_by_escape:
            self.map_state.is_open = False
        if ignored_press and self.settings_dialog is None and not own_window_is_foreground():
            self._log(f"忽略地图键：前台窗口不是游戏（{foreground_window_title() or '无标题'}）")
        if event is None and not closed_by_escape:
            return
        RouteProjectionApp._apply_map_key_transition(self, event, closed_by_escape, time.perf_counter())

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
        self.map_probe_pending = False
        self.map_presence_elapsed_ms = None
        self.map_presence_started_at = None
        self.hide_overlay()
        self.floor_monitor_active = False
        self.opening_floor_sample = None
        self.floor_probe_candidate = None
        self.floor_probe_count = 0
        self.floor_probe_unknown_count = 0
        self.floor_probe_needs_projection = False
        if closed_by_escape or event == "closed":
            self.map_presence_text = "ESC 已关闭" if closed_by_escape else "地图键已关闭"
            self._set_status("地图已关闭，投影已隐藏", MUTED)
            self._log("ESC：地图关闭，投影已隐藏" if closed_by_escape else "地图键：地图关闭，投影已隐藏")
            return
        self.map_presence_text = "正在检查…"
        self.map_presence_started_at = key_detected_at
        self.map_probe_pending = True
        generation = self.map_generation
        expected_open = event == "opened"
        probe_session = getattr(self, "session_id", None)
        capture_delay = map_capture_delay_ms(self.settings)
        delay = max(0, math.ceil(capture_delay - (time.perf_counter()-key_detected_at)*1000))
        self.after(delay, lambda: RouteProjectionApp._start_map_presence_probe(
            self, generation, 0, expected_open, probe_session))
        action = '匹配三层完整原图' if self.settings.get('registration_backend') == 'template' else '截图识别楼层'
        self._set_status(f"地图键已按下，等待 {capture_delay} ms 后{action}…")
        self._log(f"地图键 {self.settings['map_key']} 按下：等待 {capture_delay} ms 后{action}")

    def _handle_template_floor_click(self):
        """Invalidate on a floor-control click; rematch after release without naming the floor."""
        if (self.settings.get('registration_backend') != 'template' or not self.map_state.is_open
                or self.settings_dialog is not None or self.closed
                or not game_window_is_foreground(self.settings['game_window_titles'])):
            self._template_floor_click_active = False
            return
        down = floor_control_mouse_down(self.settings['game_window_titles'], self.settings['capture_reference_size'])
        previous = getattr(self, '_template_floor_click_active', False)
        self._template_floor_click_active = down
        if down and not previous:
            self.map_generation += 1
            self.map_probe_pending = False
            self.reference_context = None
            self.hide_overlay()
        elif previous and not down:
            generation, session = self.map_generation, self.session_id
            self.map_presence_started_at = time.perf_counter()
            self.map_probe_pending = True
            self._log('定尺度模板匹配：检测到楼层按钮操作，重新匹配完整原图')
            self.after(map_capture_delay_ms(self.settings),
                       lambda: RouteProjectionApp._start_map_presence_probe(self, generation, 0, True, session))

    def _start_map_presence_probe(self, generation: int, attempt: int, expected_open: bool,
                                  probe_session: str | None = None,
                                  retry_scheduled_at: float | None = None, retry_requested_ms: float = 0.) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if (generation != self.map_generation or self.closed
                or (probe_session is not None and probe_session != getattr(self, "session_id", None))):
            return
        key_started = getattr(self, 'map_presence_started_at', None)
        capture_delay = map_capture_delay_ms(self.settings)
        if attempt == 0 and key_started is not None:
            remaining_ms = capture_delay - (time.perf_counter()-key_started)*1000
            if remaining_ms > .001:
                # A coarse Qt timer may fire early; never capture before the deadline.
                self.after(max(1, math.ceil(remaining_ms)),
                           lambda: RouteProjectionApp._start_map_presence_probe(
                               self, generation, attempt, expected_open, probe_session))
                return
        if self.settings.get('registration_backend') == 'template':
            if not self.map_state.is_open or not game_window_is_foreground(self.settings['game_window_titles']):
                self.map_probe_pending = False
                return
            self.map_probe_pending = False
            self.opening_floor_sample = None
            self.floor_monitor_active = False
            self.map_presence_text = '定尺度模板匹配：跳过楼层与 mylocation 检测'
            self.map_presence_elapsed_ms = (time.perf_counter()-key_started)*1000 if key_started is not None else None
            self._refresh_status()
            self._log(f'定尺度模板匹配开图：设定等待 {capture_delay} ms，直接截图匹配原图')
            RouteProjectionApp._auto_if_open(self, generation)
            if not self.settings['auto_project']:
                self._set_status('定尺度模板匹配已就绪；自动投影已关闭，可按投影快捷键匹配', MUTED)
            return
        titles = list(self.settings["game_window_titles"])
        reference_size = list(self.settings["capture_reference_size"])
        probe_session = getattr(self, "session_id", None)
        dispatched_at = time.perf_counter()
        profile = dict(attempt=attempt + 1,
                       probe_mode='floor_only_configurable',
                       capture_delay_ms=capture_delay,
                       initial_requested_ms=capture_delay if attempt == 0 else 0.,
                       initial_dispatch_ms=(dispatched_at-key_started)*1000 if attempt==0 and key_started is not None else 0.,
                       retry_wait_ms=(dispatched_at-retry_scheduled_at)*1000 if retry_scheduled_at is not None else 0.,
                       retry_requested_ms=retry_requested_ms)

        def worker() -> None:
            worker_started = time.perf_counter()
            profile['thread_start_ms'] = (worker_started-dispatched_at)*1000
            image = None
            region = None
            detection = None
            error = None
            floor_sample = None
            try:
                lookup_started = time.perf_counter()
                client = foreground_game_client(titles)
                profile['window_lookup_ms'] = (time.perf_counter()-lookup_started)*1000
                # Only a reliable selected-floor control may authorize projection.
                try:
                    captured_at = time.perf_counter()
                    floor, floor_image, floor_region = capture_floor_indicator(client, reference_size, timings=profile)
                    floor_sample = CapturedFloorSample(floor, floor_image, floor_region,
                        client.as_list(), reference_size, captured_at, generation, probe_session)
                except Exception as exc:
                    error = f"楼层探测失败：{exc}"
            except Exception as exc:
                error = str(exc)
            profile['worker_ms'] = (time.perf_counter()-worker_started)*1000
            event = ("map_presence", generation, attempt, expected_open,
                     detection, image, region, error, floor_sample, probe_session, profile)
            bridge = getattr(self, '_map_probe_bridge', None)
            profile['delivery'] = 'qt_signal' if bridge is not None else 'poll_queue'
            profile['_posted_at'] = time.perf_counter()
            if bridge is not None:
                try:bridge.ready.emit(event)
                except RuntimeError:pass  # Window was destroyed during capture.
            else:
                self.results.put(event)

        threading.Thread(target=worker, name="map-presence-probe", daemon=True).start()

    @pyqtSlot(object)
    def _receive_map_probe(self, event):
        if self.closed:return
        event[-1]['_received_at'] = time.perf_counter()
        # The old polling loop checked close keys before consuming results.
        # Keep that ordering when results now arrive between polling ticks.
        self._handle_map_key()
        self._apply_map_presence_probe(*event[1:])

    def _apply_map_presence_probe(self, generation: int, attempt: int, expected_open: bool,
                                  detection: MapPresence | None, image: np.ndarray | None,
                                  region: list[int] | None, error: str | None,
                                  floor_sample: CapturedFloorSample | None = None,
                                  probe_session: str | None = None,
                                  profile: dict | None = None) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if (generation != self.map_generation or self.closed
                or (probe_session is not None and probe_session != getattr(self, "session_id", None))):
            return
        apply_started = time.perf_counter()
        if profile is not None:
            received_at = profile.pop('_received_at', apply_started)
            profile['delivery_wait_ms'] = (received_at-profile.pop('_posted_at', received_at))*1000
            profile['close_key_check_ms'] = (apply_started-received_at)*1000
        floor_open = floor_sample is not None and floor_sample.detection.accepted
        # Ignore even an old marker result: mylocation no longer authorizes opening.
        detection, image, region = None, None, None
        observed = "open" if floor_open else "uncertain"
        started_at = getattr(self, "map_presence_started_at", None)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0 if started_at is not None else None
        retry = ((observed == "uncertain" or (observed == "open") != expected_open)
                 and elapsed_ms is not None and elapsed_ms < MAP_PROBE_WINDOW_MS)
        if not retry:
            self.map_presence_elapsed_ms = elapsed_ms
        diagnostic = dict(detection=detection, region=region, attempt=attempt, error=error,
                          floor_sample=floor_sample, observed_state=observed)
        score_text = f"{max(floor_sample.detection.scores.values(), default=0.):.1%}" if floor_sample else "—"
        source_text = "楼层标识"
        state_text = {"open": "已打开", "closed": "已关闭", "uncertain": "无法确认"}[observed]
        self.map_presence_text = f"{state_text} · {source_text} {score_text} · 第 {attempt + 1} 次"
        self._refresh_status()
        if retry:
            self._set_status("地图界面正在切换，稍后复核…", AMBER)
            RouteProjectionApp._finish_map_probe_profile(self, generation, probe_session, attempt, profile,
                                                        apply_started, image, diagnostic, False)
            delay = max(1, min(map_capture_delay_ms(self.settings), int(MAP_PROBE_WINDOW_MS - elapsed_ms)))
            scheduled_at = time.perf_counter()
            self.after(delay,
                       lambda: RouteProjectionApp._start_map_presence_probe(
                self, generation, attempt + 1, expected_open, probe_session, scheduled_at, delay))
            return
        if observed == "open":
            self.map_probe_pending = False
            self.map_state.is_open = True
            self.opening_floor_sample = floor_sample if floor_open else None
            RouteProjectionApp._map_toast(self, "地图已打开",
                f"已锁定 {self.locked_layout}，楼层已确认。" if getattr(self, "locked_layout", None)
                else "布局尚未确认，请按 F8 识别并确认。", success=bool(getattr(self, "locked_layout", None)))
            self.floor_probe_next = time.monotonic() + 0.6
            self._log(f"地图界面已确认：{source_text}相似度 {score_text}，校验耗时 {elapsed_ms:.0f} ms，截图区域 {floor_sample.region if floor_open else region}"
                      if elapsed_ms is not None else f"地图界面已确认：相似度 {score_text}，截图区域 {region}")
            if self.settings["auto_project"]:
                self._set_status("游戏地图已确认，开始截图配准…", GREEN)
                RouteProjectionApp._auto_if_open(self, generation)
            else:
                self._set_status("游戏地图已确认；自动投影已关闭", MUTED)
        else:
            self.map_probe_pending = False
            self.map_state.is_open = False
            self.opening_floor_sample = None
            self.hide_overlay()
            self.floor_monitor_active = False
            if observed == "closed":
                self._set_status("游戏地图已关闭，投影已隐藏", MUTED)
                self._log(f"地图界面未出现：相似度 {score_text}，投影已隐藏")
            else:
                self._set_status("无法确认游戏地图界面，已停止本次投影", AMBER)
                self._log(f"地图界面校验未确认：{error or score_text}，投影已隐藏")
        RouteProjectionApp._finish_map_probe_profile(self, generation, probe_session, attempt, profile,
                                                    apply_started, image, diagnostic, True)

    def _finish_map_probe_profile(self, generation, session, attempt, profile, apply_started,
                                 image, diagnostic, complete):
        finished = time.perf_counter()
        started = getattr(self, 'map_presence_started_at', None)
        elapsed = (finished-started)*1000 if started is not None else None
        timing = None
        if profile is not None:
            profile['main_apply_ms'] = (finished-apply_started)*1000
            key = (generation, session)
            trace = getattr(self, '_map_probe_trace', None)
            if attempt==0 or trace is None or trace['key']!=key:
                trace = {'key':key, 'rounds':[]}
                self._map_probe_trace = trace
            trace['rounds'].append(dict(profile))
            fields = ('initial_dispatch_ms','retry_wait_ms','thread_start_ms','window_lookup_ms',
                      'floor_region_ms','floor_capture_ms','floor_classify_ms','marker_region_ms',
                      'marker_capture_ms','marker_classify_ms','delivery_wait_ms','close_key_check_ms','main_apply_ms')
            totals = {name:sum(row.get(name,0.) for row in trace['rounds']) for name in fields}
            totals['other_ms'] = max(0., (elapsed or 0.)-sum(totals.values()))
            timing = dict(generation=generation,session_id=session,complete=complete,total_ms=elapsed,
                          probe_mode='floor_only_configurable',
                          capture_delay_ms=trace['rounds'][0].get('capture_delay_ms', map_capture_delay_ms(self.settings)),
                          rounds=list(trace['rounds']),totals_ms=totals,
                          note='From handled map key to applied decision; diagnostics run afterward. Substages exclude retry/dispatch overlap. Physical key polling delay is not measured.')
            if complete:
                self.map_presence_elapsed_ms = elapsed
                labels = [('initial_dispatch_ms',f"首次等待（设定{timing['capture_delay_ms']}ms）"),('thread_start_ms','线程启动'),
                          ('window_lookup_ms','窗口'),('floor_capture_ms','楼层截图'),('floor_classify_ms','楼层识别'),
                          ('floor_region_ms','区域换算'),('close_key_check_ms','关闭检查'),
                          ('delivery_wait_ms','回传等待'),('main_apply_ms','主线程'),('retry_wait_ms','重试等待'),
                          ('other_ms','其他')]
                self._log(f"地图校验分段（{len(trace['rounds'])} 轮，总计 {elapsed:.1f} ms）：" +
                          ' · '.join(f'{label} {totals[name]:.1f}' for name,label in labels))
                self._refresh_status()
        writer = getattr(self, '_map_probe_diagnostics', None)
        if writer is not None:
            writer.submit(image, **diagnostic, elapsed_ms=elapsed, timing=timing)

    def _auto_if_open(self, generation: int) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        if self.map_state.is_open and generation == self.map_generation and not self.closed and self.settings["auto_project"]:
            if game_window_is_foreground(self.settings["game_window_titles"]):
                if self.busy:
                    self._pending_auto_projection = (generation, getattr(self, 'session_id', None),
                        self.settings.get('registration_backend', 'auto'), getattr(self, 'locked_layout', None))
                else:
                    self.request_work("project", auto=True)
            else:
                self._set_status("游戏窗口不在前台，本次未截图", AMBER)

    def _apply_floor_probe(self, generation: int, detection: FloorDetection | None, error: str | None) -> None:
        if getattr(self, 'map_keys', None) is not None:
            self._handle_map_key()
        self.floor_probe_busy = False
        self.floor_probe_next = time.monotonic() + 0.18
        if (generation != self.map_generation or not self.map_state.is_open
                or not self.floor_monitor_active or self.busy):
            return
        if error or detection is None or not detection.accepted:
            self.floor_probe_unknown_count += 1
            self.floor_probe_candidate = None
            self.floor_probe_count = 0
            if self.floor_probe_unknown_count >= 2:
                self.hide_overlay()
                self.reference_context = None
                self.floor_probe_needs_projection = True
                self.floor_detect_var.set("楼层标识暂时无法确认，已隐藏投影")
            return
        self.floor_probe_unknown_count = 0
        floor = detection.floor
        if floor != self.floor_var.get():
            self.hide_overlay()
            self.reference_context = None
            self.floor_probe_needs_projection = True
            self._set_status("检测到楼层切换，正在确认新楼层…", AMBER)
        if floor == self.floor_probe_candidate:
            self.floor_probe_count += 1
        else:
            self.floor_probe_candidate = floor
            self.floor_probe_count = 1
        if self.floor_probe_needs_projection and self.floor_probe_count >= 2:
            self.floor_probe_needs_projection = False
            self.floor_probe_next = time.monotonic() + 0.6
            self._log(f"楼层切换已确认：{FLOOR_NAMES[floor]}，重新截图配准")
            self.request_work("project", auto=True)

    def _start_floor_probe_if_due(self) -> None:
        if self.settings.get('registration_backend') == 'template':
            return
        if (self.floor_probe_busy or self.busy or not self.map_state.is_open or not self.floor_monitor_active
                or not self.locked_layout
                or not self.settings["auto_floor"] or self.settings_dialog is not None
                or time.monotonic() < self.floor_probe_next
                or not game_window_is_foreground(self.settings["game_window_titles"])):
            return
        self.floor_probe_busy = True
        generation = self.map_generation
        titles = list(self.settings["game_window_titles"])
        reference_size = list(self.settings["capture_reference_size"])

        def worker() -> None:
            try:
                client = foreground_game_client(titles)
                detection, _image, _region = capture_floor_indicator(client, reference_size)
                self.results.put(("floor_probe", generation, detection, None))
            except Exception as exc:
                self.results.put(("floor_probe", generation, None, str(exc)))

        threading.Thread(target=worker, name="floor-indicator-probe", daemon=True).start()

    def _tick(self) -> None:
        if self.closed:
            return
        writer = getattr(self, '_map_probe_diagnostics', None)
        error = writer.pop_error() if writer is not None else None
        if error:self._log(f"地图界面诊断保存失败：{error}")
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
                if self.settings_dialog is None:
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
            if event[0] == "model":
                _, profile_key, bundle, error = event
                if bundle:
                    self.model_cache[profile_key] = bundle
                    self._log(f"模型就绪：{profile_key} / CPU")
                else:
                    self._set_status(f"模型加载失败：{error}", BAD)
            elif event[0] == "registration_ready":
                _, backend, info = event
                if backend == self.settings.get("registration_backend", "auto"):
                    if info['effective_backend'] == 'cuda':
                        self._log(f"GPU 配准就绪：{info['device']} · 第四轮算法")
                    elif info['effective_backend'] == 'template':
                        self._log("定尺度模板匹配就绪：使用完整原图及保存尺度，按屏幕比例修正；候选直接显示")
                    else:
                        self._log("CPU 配准就绪：原正式版算法" +
                                  (f" · GPU 回退原因：{info['fallback_reason']}" if info.get('fallback_reason') else ""))
            elif event[0] == "saved":
                _, saved_result, directory, error = event
                if saved_result is self.last_result:
                    self.path_var.set(f"诊断文件：{directory}" if directory else f"诊断保存失败：{error}")
                    self._log(self.path_var.get())
            elif event[0] == "floor_probe":
                self._apply_floor_probe(*event[1:])
            elif event[0] == "map_presence":
                self._apply_map_presence_probe(*event[1:])
            elif event[0] == "reference_ready":
                self._apply_reference_ready(*event[1:])
            elif event[0] == 'retrieval_index':
                if (event[1] == self.profile.key and
                        self.settings.get('layout_recognition_backend', 'cnn') == 'icon_structure'):
                    self._log(f'识别资源准备失败：{event[3]}' if event[3] else f'图标与结构识别资源就绪：{event[2][:12]}')
            elif event[0] == "preloaded_template":
                _, key, prepared, error = event
                if (prepared is not None and key[0] == self.profile.key
                        and key[1] == self.locked_layout):
                    self.prepared_cache.setdefault(key, prepared)
                    self._log(f"模板预热完成：{key[1]} / {FLOOR_NAMES[key[2]]}")
                elif error and key[0] == self.profile.key and key[1] == self.locked_layout:
                    self._log(f"模板预热失败：{key[1]} / {key[2]} · {error}")
            else:
                self._apply_result(*event[1:])
                RouteProjectionApp._drain_pending_projection(self)
        self._start_floor_probe_if_due()
        self.after(25, self._tick)

    def _shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        writer = getattr(self, '_map_probe_diagnostics', None)
        if writer is not None:writer.close()
        self.hide_overlay()
        if getattr(self, 'map_keys', None) is not None:
            self.map_keys.stop()
        if self.hotkeys:
            self.hotkeys.stop()
        self.overlay.destroy()
        self.reference_overlay.destroy()
        self.toast.close()

    def closeEvent(self, event):
        page = self.settings_dialog
        if page is not None and (page._settings_dirty or page._settings_save_job is not None):
            if not page._persist_settings(show_error=True):
                event.ignore()
                return
        self._shutdown()
        event.accept()
        QApplication.instance().quit()
