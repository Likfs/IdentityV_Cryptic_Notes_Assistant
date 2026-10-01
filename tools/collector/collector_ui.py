"""Qt sample collector. All file/capture work is serialized outside the UI thread."""
from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mss
from PIL import Image
from PyQt6.QtCore import Qt, QRect, QSize, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QFrame, QGraphicsScene, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from tools.collector.fixed_roi_collector import (
    DEFAULT_CONFIG_PATH, DEFAULT_LABELS_PATH, PENDING_LAYOUT_ID, PENDING_LAYOUT_NAME,
    Category, CaptureRegion, ChineseVoiceNotifier, GlobalHotkeyManager, LayoutCategory,
    SampleWriter, ScreenRegionBorder, VOICE_PROMPTS, grab_region, image_sha256,
    load_capture_config, load_category_config, validate_region_inside_screen, virtual_screen_bounds,
)
from tools.collector.capture_guard import CollectorCaptureGuard
from tools.common.capture_config import save_capture_config, capture_settings_path


STYLE = """
QMainWindow, QDialog { background: #f6faff; }
QWidget { font-family: 'Microsoft YaHei UI'; font-size: 13px; color: #172442; }
QFrame#card { background: white; border: 1px solid #dbe7f5; border-radius: 8px; }
QWidget#cardHeader { background: #eff6ff; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QLabel#heading { font-size: 16px; font-weight: bold; }
QLabel#title { font-size: 19px; font-weight: bold; }
QLabel#muted { color: #73839b; }
QLabel#state { font-size: 23px; font-weight: bold; padding: 4px 0 12px; }
QPushButton, QToolButton { background: #ffffff; border: 1px solid #c7d9ef; border-radius: 6px; padding: 8px 10px; }
QPushButton:hover, QToolButton:hover { background: #edf5ff; border-color: #398aff; }
QPushButton:pressed, QToolButton:pressed { background: #dfedff; }
QPushButton:disabled { color: #97a5b9; background: #f6f8fc; border-color: #e2e8f0; }
QPushButton#primary { color: white; background: #0878f9; border-color: #0878f9; }
QPushButton#primary:hover { background: #0066dd; }
QPushButton#hint { color: #0766cf; background: #eef6ff; text-align: left; }
QComboBox, QDoubleSpinBox, QSpinBox { background: white; border: 1px solid #c9d9ed; border-radius: 5px; padding: 6px; min-height: 22px; }
QComboBox:disabled { color: #8492a7; }
QComboBox::drop-down { width: 25px; border: none; }
QComboBox::down-arrow, QDoubleSpinBox::down-arrow, QSpinBox::down-arrow { image: url(__DOWN_ICON__); width: 10px; height: 6px; }
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow { image: url(__UP_ICON__); width: 10px; height: 6px; }
QDoubleSpinBox::up-button, QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 24px; border: none; }
QDoubleSpinBox::down-button, QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 24px; border: none; }
QCheckBox { spacing: 9px; padding: 4px 0; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #8199b9; border-radius: 3px; background: white; }
QCheckBox::indicator:checked { background: #0878f9; border-color: #0878f9; image: url(__CHECK_ICON__); }
QListWidget { background: white; border: none; outline: none; }
QListWidget::item { padding: 6px; border-bottom: 1px solid #edf2f8; }
QListWidget::item:selected { background: #e8f2ff; color: #105cbe; }
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: transparent; }
QStatusBar { background: #edf5ff; border-top: 1px solid #dce8f6; }
QToolTip { background: #172442; color: white; padding: 6px; border: none; }
"""


def pixmap_from_image(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    raw = rgb.tobytes()
    return QPixmap.fromImage(QImage(raw, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy())


def line_icon(kind: str, size: int = 24) -> QIcon:
    """Small vector icons, drawn locally so the tool has no asset dependencies."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 24, size / 24)
    p.setPen(QPen(QColor("#0878f9"), 1.8))
    if kind == "preview":
        p.drawRoundedRect(2, 3, 20, 14, 1, 1)
        p.drawLine(12, 17, 12, 21)
        p.drawLine(7, 21, 17, 21)
    elif kind == "play":
        p.drawLine(8, 5, 18, 12); p.drawLine(18, 12, 8, 19); p.drawLine(8, 19, 8, 5)
    elif kind == "pause":
        p.drawLine(8, 5, 8, 19); p.drawLine(16, 5, 16, 19)
    elif kind == "stop":
        p.fillRect(6, 6, 12, 12, QColor("#0878f9"))
    elif kind == "check":
        p.drawLine(4, 12, 9, 17); p.drawLine(9, 17, 20, 5)
    elif kind == "camera":
        p.drawRoundedRect(2, 6, 20, 14, 2, 2); p.drawEllipse(8, 9, 8, 8); p.drawLine(8, 3, 16, 3)
    else:
        for x in (3, 14):
            for y in (3, 14):
                p.drawRoundedRect(x, y, 7, 7, 1, 1)
    p.end()
    return QIcon(pm)


class PreviewView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#eef0f3"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.fitting = True
        self.image_item = None
        self.empty_text = self.scene().addText("暂无截图\n点击「预览」或按 F5")
        self.empty_text.setDefaultTextColor(QColor("#8290a4"))

    def show_image(self, image: Image.Image):
        pm = pixmap_from_image(image)
        if self.image_item is None:
            self.scene().clear()
            self.image_item = self.scene().addPixmap(pm)
        else:
            self.image_item.setPixmap(pm)
        self.scene().setSceneRect(self.image_item.boundingRect())
        if self.fitting:
            self.fit()

    def fit(self):
        self.fitting = True
        if self.image_item is not None:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        self.fitting = False
        self.resetTransform()

    def zoom(self, factor):
        self.fitting = False
        target = self.transform().m11() * factor
        if 0.03 <= target <= 16:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitting:
            self.fit()


class RegionView(PreviewView):
    """Select in physical image coordinates, independent of Qt display scaling."""
    def __init__(self, image, selected):
        super().__init__()
        self.show_image(image)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.selected = selected
        self.origin = None
        self.selection = self.scene().addRect(0, 0, 0, 0, QPen(QColor("#0878f9"), 3))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = self.mapToScene(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            from PyQt6.QtCore import QRectF
            rect = QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized()
            self.selection.setRect(rect.intersected(self.image_item.boundingRect()))

    def mouseReleaseEvent(self, event):
        if self.origin is not None:
            self.mouseMoveEvent(event)
            rect = self.selection.rect().toAlignedRect()
            self.origin = None
            if rect.width() > 1 and rect.height() > 1:
                self.selected(rect)


class RegionDialog(QDialog):
    def __init__(self, region, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择截图区域")
        self.resize(1050, 760)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("在桌面快照上拖动框选，也可以输入物理像素坐标。"))
        with mss.mss() as sct:
            self.bounds = virtual_screen_bounds(sct)
            shot = sct.grab(self.bounds.as_mss_monitor())
            image = Image.frombytes("RGB", shot.size, shot.rgb)
        self.view = RegionView(image, self.set_selection)
        layout.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.fields = []
        for name, value in zip(("左", "上", "宽", "高"), region.as_list()):
            row.addWidget(QLabel(name))
            field = QSpinBox()
            field.setRange(-100000 if len(self.fields) < 2 else 1, 100000)
            field.setValue(value)
            self.fields.append(field)
            row.addWidget(field)
        layout.addLayout(row)
        self.error = QLabel("")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("使用此区域")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_selection(self, rect: QRect):
        for field, value in zip(self.fields, (rect.x() + self.bounds.left, rect.y() + self.bounds.top, rect.width(), rect.height())):
            field.setValue(value)

    def selected_region(self):
        return CaptureRegion(*(field.value() for field in self.fields))

    def validate(self):
        try:
            validate_region_inside_screen(self.selected_region(), self.bounds)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()


def card(title, icon="grid"):
    frame = QFrame()
    frame.setObjectName("card")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    header = QWidget()
    header.setObjectName("cardHeader")
    row = QHBoxLayout(header)
    row.setContentsMargins(14, 10, 14, 10)
    pict = QLabel()
    pict.setPixmap(line_icon(icon).pixmap(24, 24))
    row.addWidget(pict)
    label = QLabel(title)
    label.setObjectName("heading")
    row.addWidget(label)
    row.addStretch()
    outer.addWidget(header)
    body = QVBoxLayout()
    body.setContentsMargins(14, 12, 14, 14)
    body.setSpacing(10)
    outer.addLayout(body, 1)
    return frame, body, row, label


class FixedRoiCollectorApp(QMainWindow):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH, labels_path=DEFAULT_LABELS_PATH, *, writer=None, services=True):
        super().__init__()
        self.config_path = capture_settings_path(Path(config_path))
        self.config = load_capture_config(self.config_path)
        self.categories = load_category_config(labels_path)
        self.region = CaptureRegion.from_value(self.config["capture_roi"])
        self.writer = writer or SampleWriter()
        self.screen_border = ScreenRegionBorder(self.region)
        self.voice = ChineseVoiceNotifier()
        self.hotkeys = GlobalHotkeyManager()
        self.services = services
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collector-work")
        self.results = queue.Queue()
        self.busy = False
        self.closing = False
        self.running = False
        self.paused = False
        self.last_hash = ""
        self.pending_ids = []
        self.saved_count = self.deleted_count = self.skipped_count = 0
        self.elapsed = 0.0
        self.active_since = None
        self.setWindowTitle("固定区域样本采集器")
        self.setWindowIcon(line_icon("grid", 32))
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, available.width() - 40), min(940, available.height() - 60))
        self.setMinimumSize(1140, 720)
        style = STYLE
        for key, name in (("CHECK", "check"), ("DOWN", "down"), ("UP", "up")):
            style = style.replace(f"__{key}_ICON__", Path(__file__).with_name(name + ".svg").as_posix())
        self.setStyleSheet(style)
        self._build_ui()
        self.capture_guard = CollectorCaptureGuard(self)
        self.capture_guard.changed.connect(self.update_capture_status)
        self.capture_guard.ensure_excluded()
        self.capture_timer = QTimer(self)
        self.capture_timer.setSingleShot(True)
        self.capture_timer.timeout.connect(lambda: self.request_capture(True))
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(30)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.refresh_counts)
        self.clock_timer.start(250)
        self.shortcuts = []
        self.actions = {"preview": self.preview_once, "start": self.start_sampling,
                        "pause": self.toggle_pause, "stop": self.stop_sampling,
                        "capture": self.capture_and_save_once, "confirm": self.confirm_pending_layout,
                        "supplement": self.supplement_existing_layout, "rebuild": self.rebuild_all_jsonl}
        # Global shortcuts are handled exactly once; local fallbacks cover failed registrations.
        for action, _mods, _vk, key in GlobalHotkeyManager.HOTKEYS.values():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda a=action, k=key: self._local_action(a, k))
            self.shortcuts.append(shortcut)
        self.map_combo.currentIndexChanged.connect(self.context_changed)
        self.difficulty_combo.currentIndexChanged.connect(self.context_changed)
        self.context_changed()
        if services:
            self.voice.start()
            self.hotkeys.start()
            self.update_border()
            self.speak("app_started")
        self.statusBar().showMessage("就绪 · 截图先保存为待确认样本，确认布局后生成训练索引")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 8, 16, 14)
        root.setSpacing(12)
        top = QHBoxLayout()
        logo = QLabel(); logo.setPixmap(line_icon("grid", 28).pixmap(28, 28))
        top.addWidget(logo)
        title = QLabel("固定区域样本采集器"); title.setObjectName("title")
        top.addWidget(title)
        subtitle = QLabel("样本采集工作台"); subtitle.setObjectName("muted")
        top.addWidget(subtitle); top.addStretch()
        save = self.button("保存设置", self.save_settings)
        save.setToolTip("保存截图区域、采样间隔和选项到采集器配置")
        top.addWidget(save)
        root.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(420)
        left = QWidget()
        left_scroll.setWidget(left)
        left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(12)
        split.addWidget(left_scroll)

        frame, body, _, _ = card("采集设置")
        left_layout.addWidget(frame)
        roi_row = QHBoxLayout()
        self.roi_label = QLabel(); self.roi_label.setWordWrap(True)
        roi_row.addWidget(self.roi_label, 1)
        self.region_button = self.button("选择区域", self.select_region)
        roi_row.addWidget(self.region_button)
        body.addLayout(roi_row)
        form = QFormLayout(); form.setVerticalSpacing(7)
        self.map_combo = QComboBox(); self.difficulty_combo = QComboBox()
        for combo, key, default in ((self.map_combo, "maps", "default_map_id"), (self.difficulty_combo, "difficulties", "default_difficulty_id")):
            for category in self.categories[key]:
                combo.addItem(category.display, category)
            combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i).id == self.categories[default]))
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addRow("地图", self.map_combo); form.addRow("难度", self.difficulty_combo)
        self.next_label = QLabel(); self.next_label.setStyleSheet("padding: 8px; background: #f1f6fc; border-radius: 5px;")
        form.addRow("下一个新布局", self.next_label)
        existing = QHBoxLayout()
        self.existing_combo = QComboBox(); self.existing_combo.setMinimumContentsLength(12)
        self.existing_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        existing.addWidget(self.existing_combo, 1)
        self.refresh_button = self.button("↻", self.refresh_layouts)
        self.refresh_button.setToolTip("刷新现存布局和样本数")
        existing.addWidget(self.refresh_button)
        form.addRow("补充已有布局", existing)
        self.interval = QDoubleSpinBox(); self.interval.setRange(.1, 3600); self.interval.setDecimals(2)
        self.interval.setSingleStep(.1); self.interval.setValue(self.config["default_interval_seconds"])
        form.addRow("采样间隔（秒）", self.interval)
        body.addLayout(form)
        self.supplement_button = self.button("将待确认样本补充到所选布局\nCtrl+Alt+E", self.supplement_existing_layout)
        self.supplement_button.setObjectName("hint")
        body.addWidget(self.supplement_button)

        frame, body, _, _ = card("选项")
        left_layout.addWidget(frame)
        self.border_check = QCheckBox("显示屏幕截图边框")
        self.duplicates_check = QCheckBox("跳过完全相同的连续画面")
        self.voice_check = QCheckBox("启用中文语音提示")
        self.minimize_check = QCheckBox("开始连续采集后最小化窗口")
        for widget, value in ((self.border_check, self.config.get("show_capture_border", True)),
                              (self.duplicates_check, self.config["skip_exact_duplicates"]),
                              (self.voice_check, self.config["voice_prompts_enabled"]),
                              (self.minimize_check, self.config.get("minimize_on_start", False))):
            widget.setChecked(value); body.addWidget(widget)
        body.setSpacing(2)
        self.border_check.toggled.connect(self.update_border)

        frame, body, _, _ = card("快捷操作")
        left_layout.addWidget(frame)
        grid = QGridLayout(); grid.setSpacing(8)
        for col, (text, callback, icon) in enumerate((("预览\nF5", self.preview_once, "preview"),
                ("开始采集\nF6", self.start_sampling, "play"), ("暂停/继续\nF7", self.toggle_pause, "pause"),
                ("停止\nF8", self.stop_sampling, "stop"))):
            button = QToolButton()
            button.setText(text); button.setIcon(line_icon(icon))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, action=callback: action())
            button.setMinimumHeight(72)
            grid.addWidget(button, 0, col)
        grid.addWidget(self.button("立即采集一张  (F10)", self.capture_and_save_once, "camera"), 1, 0, 1, 4)
        self.confirm_button = self.button("确认并归档", self.confirm_pending_layout, "check")
        self.confirm_button.setMinimumHeight(64)
        grid.addWidget(self.confirm_button, 2, 0, 1, 2)
        grid.addWidget(self.button("重建全部 JSONL\nCtrl+Alt+R", self.rebuild_all_jsonl), 2, 2, 1, 2)
        body.addLayout(grid)
        left_layout.addStretch()

        frame, body, header, _ = card("截图预览", "preview")
        frame.setMinimumWidth(400)
        self.preview = PreviewView()
        for label, callback in (("适应窗口", self.preview.fit), ("1:1", self.preview.actual_size), ("−", lambda: self.preview.zoom(1/1.2)), ("＋", lambda: self.preview.zoom(1.2))):
            header.addWidget(self.button(label, callback))
        body.setContentsMargins(8, 8, 8, 8)
        body.addWidget(self.preview)
        split.addWidget(frame)
        right = QWidget(); right.setMinimumWidth(252)
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(4, 0, 0, 0); right_layout.setSpacing(12)
        split.addWidget(right)
        frame, body, _, _ = card("采集状态")
        right_layout.addWidget(frame)
        self.state_label = QLabel("● 已停止"); self.state_label.setObjectName("state")
        body.addWidget(self.state_label)
        self.count_labels = {}
        for key in ("已保存", "待确认", "已删除", "跳过", "运行时间"):
            row = QHBoxLayout(); label = QLabel(key); label.setStyleSheet("color: #536c90;")
            row.addWidget(label); row.addStretch()
            value = QLabel("0"); self.count_labels[key] = value; row.addWidget(value)
            body.addLayout(row)
        frame, body, header, self.recent_title = card("最近采集 (0)")
        clear = self.button("清空", self.clear_recent)
        clear.setToolTip("仅清空最近采集列表，不删除磁盘上的样本")
        header.addWidget(clear)
        self.recent = QListWidget(); self.recent.setIconSize(QSize(92, 74))
        self.recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self.recent_menu)
        self.recent.itemClicked.connect(self.view_recent)
        self.recent.setToolTip("点击预览；右键可永久删除该样本。最多显示最近 60 张。")
        self.empty_recent = QLabel("暂无采集记录\n\n开始采集后将显示缩略图")
        self.empty_recent.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_recent.setObjectName("muted")
        self.empty_recent.setMinimumHeight(200)
        body.addWidget(self.empty_recent, 1); body.addWidget(self.recent, 1)
        self.recent.hide()
        right_layout.addWidget(frame, 1)
        split.setSizes([435, 655, 290])
        self.context_label = QLabel()
        self.capture_status_label = QLabel("捕获保护：检查中")
        self.statusBar().addPermanentWidget(self.capture_status_label)
        self.statusBar().addPermanentWidget(self.context_label)
        self.update_roi_label()

    def update_capture_status(self):
        protected = all(widget.property("captureExcluded") is True for widget in self.capture_guard.windows())
        self.capture_status_label.setText("捕获保护：已启用" if protected else "捕获保护：未生效")
        self.capture_status_label.setToolTip(
            "窗口已从兼容的系统截图和录屏中排除（Windows 10 2004 及以上）。"
            if protected else "Windows 未能启用窗口捕获排除，将阻止采集，避免把采集器保存为样本。")

    @staticmethod
    def button(text, callback, icon=None):
        button = QPushButton(text)
        if icon:
            button.setIcon(line_icon(icon))
        button.clicked.connect(lambda _checked=False: callback())
        return button

    def context(self):
        return self.map_combo.currentData(), self.difficulty_combo.currentData()

    def context_changed(self):
        self.pause_for_change()
        map_category, difficulty = self.context()
        self.pending_ids = self.writer.pending_sample_ids(map_category.id, difficulty.id)
        self.saved_count = len(self.pending_ids)
        self.deleted_count = self.skipped_count = 0
        self.last_hash = ""
        self.clear_recent()
        self.refresh_layouts()
        self.refresh_counts()

    def refresh_layouts(self):
        map_category, difficulty = self.context()
        next_layout = self.writer.next_layout(map_category, difficulty)
        self.next_label.setText(f"{next_layout.id}（{next_layout.name}）")
        self.confirm_button.setText(f"确认并归档为\n{next_layout.id}  (Ctrl+Alt+N)")
        previous = self.existing_combo.currentData()
        self.existing_combo.clear(); self.existing_combo.addItem("请选择已有布局", None)
        for layout_id in self.writer.known_layout_ids(map_category.id, difficulty.id):
            count = self.writer.layout_sample_count(map_category.id, difficulty.id, layout_id)
            self.existing_combo.addItem(f"{layout_id}（{count} 张）", layout_id)
        self.existing_combo.setCurrentIndex(max(0, self.existing_combo.findData(previous)))
        self.context_label.setText(f"下一布局：{next_layout.id}   |   {map_category.name} / {difficulty.name}")

    def update_roi_label(self):
        r = self.region
        self.roi_label.setText(f"固定区域\n左 {r.left} / 上 {r.top} / {r.width} × {r.height}")

    def update_border(self):
        if not self.services:
            return
        try:
            if self.border_check.isChecked():
                self.screen_border.show()
            else:
                self.screen_border.hide()
        except Exception as exc:
            self.border_check.setChecked(False)
            self.statusBar().showMessage(f"显示边框失败：{exc}")

    def select_region(self):
        if not self.ready_for_edit():
            return
        self.screen_border.hide()
        self.hide()
        QTimer.singleShot(180, self._select_region_dialog)

    def _select_region_dialog(self):
        try:
            dialog = RegionDialog(self.region, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.region = dialog.selected_region()
                self.screen_border.region = self.region
                self.last_hash = ""
                self.update_roi_label()
                self.save_settings()
        except Exception as exc:
            self.report_error(exc)
        finally:
            self.show()
            self.raise_()
            self.update_border()

    def save_settings(self):
        config = dict(self.config, capture_roi=self.region.as_list(), default_interval_seconds=self.interval.value(),
                      skip_exact_duplicates=self.duplicates_check.isChecked(), voice_prompts_enabled=self.voice_check.isChecked(),
                      show_capture_border=self.border_check.isChecked(), minimize_on_start=self.minimize_check.isChecked())
        try:
            save_capture_config(config, self.config_path)
            self.config = config
            self.statusBar().showMessage("采集设置已保存")
        except Exception as exc:
            self.report_error(exc)

    def speak(self, key):
        if self.services and self.voice_check.isChecked():
            self.voice.speak(VOICE_PROMPTS[key])

    def report_error(self, exc):
        self.statusBar().showMessage(f"操作失败：{exc}")
        self.speak("operation_failed")

    def _local_action(self, action, key):
        if not self.services or not self.hotkeys.ready or any(error.startswith(key + " ") for error in self.hotkeys.errors):
            self.actions[action]()

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.map_combo, self.difficulty_combo, self.region_button, self.refresh_button):
            widget.setEnabled(not busy)

    def submit(self, work, done):
        if self.busy or self.closing:
            self.statusBar().showMessage("当前操作尚未完成，请稍候")
            return False
        self.set_busy(True)
        future = self.executor.submit(work)
        future.add_done_callback(lambda result: self.results.put((result, done)))
        return True

    def _poll(self):
        while not self.results.empty():
            future, done = self.results.get_nowait()
            self.set_busy(False)
            try:
                done(future.result())
            except Exception as exc:
                self.stop_sampling()
                self.report_error(exc)
            if self.closing:
                self.close()
                return
            if self.running and not self.paused:
                self.capture_timer.start(round(self.interval.value() * 1000))
        while not self.hotkeys.events.empty():
            action = self.hotkeys.events.get_nowait()
            if not self.closing and action in self.actions:
                self.actions[action]()
        if self.hotkeys.errors:
            self.state_label.setToolTip("；".join(self.hotkeys.errors))

    def preview_once(self):
        self.request_capture(False)

    def capture_and_save_once(self):
        self.request_capture(True, single=True)

    def request_capture(self, save, single=False):
        if not self.capture_guard.ensure_excluded():
            self.stop_sampling()
            self.report_error(RuntimeError("窗口捕获保护未生效，已取消截图；请确认系统为 Windows 10 2004 或更新版本"))
            return
        region = self.region
        map_category, difficulty = self.context()
        previous = self.last_hash
        skip = self.duplicates_check.isChecked()
        def work():
            image = grab_region(region)
            digest = image_sha256(image)
            record = None
            duplicate = save and skip and digest == previous
            if save and not duplicate:
                record = self.writer.save(image, region=region, map_category=map_category, difficulty_category=difficulty,
                                          layout_category=LayoutCategory(PENDING_LAYOUT_ID, PENDING_LAYOUT_NAME, map_category.id, difficulty.id))
            return image, digest, record, duplicate
        def done(result):
            image, digest, record, duplicate = result
            self.preview.show_image(image)
            if record:
                self.last_hash = digest
                self.pending_ids.append(record["sample_id"])
                self.saved_count += 1
                self.add_recent(image, record)
                self.statusBar().showMessage("已保存待确认样本；确认或补充布局后生成 JSONL")
                if single:
                    self.speak("single_saved")
            elif duplicate:
                self.skipped_count += 1
                self.statusBar().showMessage("当前画面与上一张完全相同，已跳过")
                if single:
                    self.speak("duplicate_skipped")
            else:
                self.statusBar().showMessage(f"预览成功：{image.width} × {image.height}")
                self.speak("preview_success")
            self.refresh_counts()
        self.submit(work, done)

    def add_recent(self, image, record):
        thumbnail = image.copy(); thumbnail.thumbnail((100, 80))
        item = QListWidgetItem(QIcon(pixmap_from_image(thumbnail)), f"{record['sample_id'][9:15]}\n待确认布局")
        item.setData(Qt.ItemDataRole.UserRole, dict(record))
        item.setToolTip(record["roi_image_path"])
        self.recent.insertItem(0, item)
        while self.recent.count() > 60:
            self.recent.takeItem(self.recent.count() - 1)
        self.update_recent()

    def update_recent(self):
        count = self.recent.count()
        self.recent_title.setText(f"最近采集 ({count})")
        self.recent.setVisible(count > 0)
        self.empty_recent.setVisible(count == 0)

    def clear_recent(self):
        self.recent.clear()
        self.update_recent()

    def view_recent(self, item):
        record = item.data(Qt.ItemDataRole.UserRole)
        path = self.writer.project_root / record["roi_image_path"]
        try:
            with Image.open(path) as image:
                self.preview.show_image(image)
        except Exception as exc:
            self.report_error(exc)

    def recent_menu(self, point):
        item = self.recent.itemAt(point)
        if item is None:
            return
        menu = QMenu(self)
        delete = menu.addAction("永久删除此样本")
        delete.setEnabled(not self.busy)
        if menu.exec(self.recent.mapToGlobal(point)) == delete:
            self.delete_recent(item)

    def delete_recent(self, item):
        if not self.ready_for_edit():
            return
        record = item.data(Qt.ItemDataRole.UserRole)
        sample_id = record["sample_id"]
        def done(_result):
            if sample_id in self.pending_ids:
                self.pending_ids.remove(sample_id)
            self.saved_count = max(0, self.saved_count - 1)
            self.deleted_count += 1
            self.last_hash = ""
            for row in range(self.recent.count() - 1, -1, -1):
                if self.recent.item(row).data(Qt.ItemDataRole.UserRole)["sample_id"] == sample_id:
                    self.recent.takeItem(row)
            self.update_recent(); self.refresh_counts(); self.refresh_layouts()
            self.statusBar().showMessage("样本已永久删除，索引已同步更新")
            self.speak("sample_deleted")
        self.submit(lambda: self.writer.delete_sample(sample_id), done)

    def pause_for_change(self):
        if self.running and not self.paused:
            self.toggle_pause()

    def ready_for_edit(self):
        self.pause_for_change()
        if self.busy or self.closing:
            self.statusBar().showMessage("正在完成当前截图或文件操作，请稍后重试")
            return False
        return True

    def confirm_pending_layout(self):
        if self.ready_for_edit():
            self.archive_pending(self.writer.next_layout(*self.context()))

    def supplement_existing_layout(self):
        if not self.ready_for_edit():
            return
        layout_id = self.existing_combo.currentData()
        if not layout_id:
            self.statusBar().showMessage("请先在「补充已有布局」中选择目标布局")
            return
        map_category, difficulty = self.context()
        if layout_id not in self.writer.known_layout_ids(map_category.id, difficulty.id):
            self.statusBar().showMessage("目标布局已不存在，请刷新列表")
            return
        self.archive_pending(LayoutCategory(layout_id, layout_id, map_category.id, difficulty.id), supplement=True)

    def archive_pending(self, layout, supplement=False):
        if not self.pending_ids:
            self.statusBar().showMessage("没有待确认样本")
            self.speak("no_pending")
            return
        ids = self.pending_ids.copy()
        map_category, difficulty = self.context()
        def done(updated):
            by_id = {record["sample_id"]: record for record in updated}
            self.saved_count = max(0, self.saved_count - len(set(ids) - by_id.keys()))
            self.pending_ids.clear()
            self.last_hash = ""
            for row in range(self.recent.count() - 1, -1, -1):
                item = self.recent.item(row)
                old = item.data(Qt.ItemDataRole.UserRole)
                if old["sample_id"] in by_id:
                    record = by_id[old["sample_id"]]
                    item.setData(Qt.ItemDataRole.UserRole, record)
                    item.setText(f"{record['sample_id'][9:15]}\n{layout.id}")
                    item.setToolTip(record["roi_image_path"])
                elif old["sample_id"] in ids:
                    self.recent.takeItem(row)
            self.refresh_layouts(); self.refresh_counts(); self.update_recent()
            self.statusBar().showMessage(f"已归档 {len(updated)} 张到 {layout.id}；忽略 {len(ids) - len(updated)} 张已手动删除的图片")
            self.speak(("layout_supplemented" if supplement else "layout_confirmed") if updated else "layout_empty")
        self.submit(lambda: self.writer.assign_pending_layout(ids, map_category=map_category, difficulty_category=difficulty, layout_category=layout), done)

    def rebuild_all_jsonl(self):
        if not self.ready_for_edit():
            return
        def done(result):
            self.pending_ids = self.writer.pending_sample_ids(*(category.id for category in self.context()))
            self.refresh_layouts(); self.refresh_counts()
            self.statusBar().showMessage(f"JSONL 重建完成：{result['layouts']} 个布局，{result['images']} 张图片")
            self.speak("jsonl_rebuilt")
        self.submit(self.writer.rebuild_all_layout_manifests, done)

    def start_sampling(self):
        if self.running:
            if self.paused:
                self.toggle_pause()
            return
        if self.busy or self.closing:
            self.statusBar().showMessage("当前操作尚未完成，请稍后开始采集")
            return
        self.running = True; self.paused = False; self.last_hash = ""
        self.elapsed = 0; self.active_since = time.monotonic()
        self.speak("sampling_started")
        if self.minimize_check.isChecked():
            self.showMinimized()
            self.capture_timer.start(200)
        else:
            self.request_capture(True)
        self.refresh_counts()

    def toggle_pause(self):
        if not self.running:
            self.statusBar().showMessage("连续采集尚未开始")
            return
        self.paused = not self.paused
        if self.paused:
            self.freeze_clock(); self.capture_timer.stop(); self.speak("sampling_paused")
        else:
            self.active_since = time.monotonic()
            if not self.busy:
                self.capture_timer.start(round(self.interval.value() * 1000))
            self.speak("sampling_resumed")
        self.refresh_counts()

    def freeze_clock(self):
        if self.active_since is not None:
            self.elapsed += time.monotonic() - self.active_since
            self.active_since = None

    def stop_sampling(self):
        was_running = self.running
        self.running = self.paused = False
        self.freeze_clock(); self.capture_timer.stop(); self.refresh_counts()
        if was_running:
            self.speak("sampling_stopped")
        self.statusBar().showMessage("采集已停止；已开始的当前截图会完成保存")

    def refresh_counts(self):
        elapsed = self.elapsed + (time.monotonic() - self.active_since if self.active_since is not None else 0)
        seconds = int(elapsed)
        values = (self.saved_count, len(self.pending_ids), self.deleted_count, self.skipped_count,
                  f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}")
        for label, value in zip(self.count_labels.values(), values):
            label.setText(str(value))
        state, color = ("已停止", "#526780") if not self.running else (("已暂停", "#c88618") if self.paused else ("采集中", "#13a75c"))
        self.state_label.setText(f"● {state}")
        self.state_label.setStyleSheet(f"color: {color};")

    def closeEvent(self, event):
        self.stop_sampling()
        if self.busy:
            self.closing = True
            self.setEnabled(False)
            self.statusBar().showMessage("正在完成当前文件操作，完成后自动关闭")
            event.ignore()
            return
        self.closing = True
        self.poll_timer.stop(); self.clock_timer.stop()
        self.capture_guard.detach()
        self.screen_border.hide(); self.hotkeys.stop(); self.voice.stop()
        self.executor.shutdown(wait=False)
        event.accept()


def run_app(config_path=DEFAULT_CONFIG_PATH, labels_path=DEFAULT_LABELS_PATH):
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = FixedRoiCollectorApp(config_path, labels_path)
    window.show()
    app.exec()
