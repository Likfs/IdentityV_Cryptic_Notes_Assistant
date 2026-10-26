"""PyQt6 widgets for the route projection application."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, QPropertyAnimation
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPixmap, QPainter
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFormLayout, QFrame, QGridLayout,
    QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QTabWidget,
)

from src.app_catalog import ACTION_KEYS, MAP_KEYS, load_settings
from src.app_catalog import map_capture_delay_ms, MAX_MAP_CAPTURE_DELAY_MS, DEFAULT_MAP_CAPTURE_DELAY_MS
from src.capture.region import CaptureRegion
from src.layout_retrieval import BACKEND_LABELS
from src.chest_counter import CHEST_ACTIONS, CHEST_HOTKEY_FIELDS
from src.capture.visibility import apply_window_capture_policy
from src.hotkeys import Hotkey, validate_action_hotkeys, KEYPAD_VKS


# Graphite surfaces and a restrained sage accent, shared by the native windows.
BG = "#14181e"
CARD = "#1c222a"
CARD_2 = "#272f39"
INK = "#edf1f5"
MUTED = "#aeb9c7"
ACCENT = "#a9cec5"
GREEN = "#b9d8c7"
AMBER = "#e4c48a"
BAD = "#efaaa9"

STYLE = f"""
QWidget {{ background: transparent; color: {INK}; font-family: 'Microsoft YaHei UI'; font-size: 14px; }}
QLabel {{ background: transparent; }}
QFrame#titlebar {{ background: {BG}; border-bottom: 1px solid #303944; }}
QFrame#card {{ background: {CARD}; border: 1px solid #303944; border-radius: 10px; }}
QFrame#subcard {{ background: #202831; border: 1px solid #3c4753; border-radius: 8px; }}
QFrame#separator {{ background: #333d48; border: none; min-height: 1px; max-height: 1px; }}
QFrame#segmented {{ background: #151b22; border: 1px solid #313b47; border-radius: 6px; }}
QFrame#segmented QPushButton {{ border: 1px solid transparent; background: transparent; padding: 5px 4px; min-height: 22px; color: {MUTED}; }}
QFrame#segmented QPushButton:checked {{ background: #35423f; border-color: #576d65; color: #e3f0eb; }}
QLabel#section {{ font-size: 15px; font-weight: 600; }}
QLabel#subheading {{ font-size: 13px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#layoutNumber {{ font-size: 30px; font-weight: 500; }}
QLabel#counterGold {{ color: #e4c48a; font-size: 20px; font-weight: 600; }}
QLabel#counterPurple {{ color: #c3acdb; font-size: 20px; font-weight: 600; }}
QLabel#good {{ color: {GREEN}; }}
QLabel#warning {{ color: {AMBER}; }}
QComboBox, QLineEdit, QSpinBox {{ background: {CARD_2}; border: 1px solid #46525f; border-radius: 6px; padding: 5px 9px; min-height: 24px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CARD_2}; color: {INK}; selection-background-color: #40584f; }}
QPushButton {{ background: {CARD_2}; border: 1px solid #46525f; border-radius: 6px; padding: 6px 12px; min-height: 24px; }}
QPushButton:hover {{ background: #333e4a; border-color: #617182; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton:checked {{ background: #35423f; border-color: #637e73; }}
QPushButton:disabled {{ color: #84909e; background: #222932; border-color: #333d48; }}
QPushButton#primary {{ background: #b3d4ca; color: #203b32; border-color: #c3dfd6; font-weight: 600; }}
QPushButton#primary:hover {{ background: #c5e2d9; }}
QPushButton#primary:disabled {{ background: #53665f; color: #b2c1ba; border-color: #63766e; }}
QPushButton#windowControl {{ background: transparent; border: none; font-size: 18px; min-width: 26px; padding: 3px; }}
QPushButton#windowControl:hover {{ background: #343e49; }}
QPushButton#quiet {{ color: {MUTED}; background: transparent; border-color: transparent; padding: 4px; }}
QPushButton#quiet:hover {{ color: {INK}; background: #29323c; }}
QPushButton#disclosure {{ background: transparent; border: none; padding: 5px 0; text-align: left; color: #ccd7e2; font-size: 13px; }}
QPushButton#disclosure:checked {{ background: transparent; }}
QPushButton#hotkeyCapture:focus {{ border: 2px solid {ACCENT}; background: #354940; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; background: #151b22; border: 1px solid #9baab9; border-radius: 4px; }}
QCheckBox::indicator:checked {{ background: #597e70; border-color: #b7d9cc; image: url("{(Path(__file__).resolve().parents[1] / 'assets/app_ui/icons/check.svg').as_posix()}"); }}
QCheckBox::indicator:disabled {{ border-color: #67717d; background: #303b46; }}
QCheckBox:focus {{ color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: #4d5a68; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; background: {ACCENT}; border-radius: 7px; margin: -5px 0; }}
QTableWidget {{ background: {CARD}; alternate-background-color: {CARD_2}; gridline-color: #38424e; border: 1px solid #46525f; }}
QHeaderView::section {{ background: {CARD_2}; color: {INK}; padding: 6px; border: none; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 10px 16px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: #d8ede5; border-bottom-color: {ACCENT}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 7px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #46535f; border-radius: 3px; min-height: 28px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QToolTip {{ background: #29343f; color: {INK}; border: 1px solid #617182; padding: 6px; }}
"""


class CollapsibleSection(QWidget):
    """Keep secondary controls available without occupying the main workspace."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        self.toggle = QPushButton()
        self.toggle.setObjectName('disclosure')
        self.toggle.setCheckable(True)
        self.toggle.setAccessibleName(title)
        self.toggle.toggled.connect(self.setExpanded)
        box.addWidget(self.toggle)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 4, 0, 4)
        self.body.setSpacing(12)
        box.addWidget(self.content)
        self.setExpanded(False)

    def setExpanded(self, expanded):
        self.toggle.setChecked(expanded)
        self.toggle.setText(('−  ' if expanded else '+  ') + self.title)
        self.content.setVisible(expanded)


class SegmentedSelector(QFrame):
    """Present an existing combo's choices without adding another state model."""

    def __init__(self, combo, parent=None):
        super().__init__(parent)
        self.combo = combo
        self.setObjectName('segmented')
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(3, 3, 3, 3)
        self.row.setSpacing(3)
        self.buttons = []
        combo.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args):
        labels = [self.combo.itemText(i) for i in range(self.combo.count())]
        if labels != [button.text() for button in self.buttons]:
            for button in self.buttons:
                self.row.removeWidget(button)
                button.hide()
                button.deleteLater()
            self.buttons = []
            for index, label in enumerate(labels):
                button = QPushButton(label)
                button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                button.setCheckable(True)
                button.setAccessibleName(label)
                button.clicked.connect(lambda _checked=False, i=index: self.combo.setCurrentIndex(i))
                self.row.addWidget(button, 1)
                self.buttons.append(button)
        for index, button in enumerate(self.buttons):
            button.setChecked(index == self.combo.currentIndex())
            button.setEnabled(self.combo.isEnabled())


class EntrancePreview(QLabel):
    """Fit an entrance image to its card without cropping or stretching it."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setObjectName("entrance_preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.source = QPixmap(str(path)) if path is not None else QPixmap()
        self.setStyleSheet(f"background: {BG}; color: {MUTED}; border: none; border-radius: 6px;")
        if self.source.isNull():
            self.setText("入口图片暂不可用\n请核对布局编号；不确定时取消")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.source.isNull():
            self.setPixmap(self.source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))


class ManualLayoutConfirmationDialog(QDialog):
    """Confirm a user-selected layout against its entrance image."""

    def __init__(self, title, text, entrance_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setStyleSheet(STYLE + f"QDialog {{ background: {BG}; }}")
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 20, 24, 20)
        body.setSpacing(16)
        heading = QLabel(title)
        heading.setObjectName("section")
        body.addWidget(heading)
        description = QLabel(text)
        description.setWordWrap(True)
        body.addWidget(description)
        preview = EntrancePreview(entrance_path, self)
        preview.setMinimumHeight(180)
        body.addWidget(preview, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        confirm = QPushButton("确认使用")
        confirm.setObjectName("primary")
        confirm.clicked.connect(self.accept)
        buttons.addWidget(confirm)
        body.addLayout(buttons)
        cancel.setFocus()
        bounds = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        self.resize(min(680, bounds.width() - 48), min(600, bounds.height() - 64))


class LayoutConfirmationDialog(QDialog):
    """Choose between the two leading recognition candidates before locking."""

    def __init__(self, candidates, entrance_root, *, screen=None):
        super().__init__(None)
        self.setWindowTitle("确认锁定布局")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowSystemMenuHint)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLE + f"""
            QFrame#confirmationPanel {{ background: {BG}; border: 1px solid #40566c; border-radius: 14px; }}
            QLabel#confirmationTitle {{ font-size: 25px; font-weight: 700; }}
            QLabel#questionIcon {{ background: #354940; color: {INK}; border-radius: 28px; font-size: 38px; font-weight: 700; }}
            QPushButton#layoutCandidate {{ background: {CARD}; border: 2px solid #344a60; border-radius: 10px; padding: 0; }}
            QPushButton#layoutCandidate:hover {{ background: {CARD_2}; border-color: #6284a3; }}
            QPushButton#layoutCandidate:checked {{ background: #35423f; border-color: {ACCENT}; }}
            QPushButton#layoutCandidate:focus {{ border-color: #83bdff; }}
            QRadioButton {{ font-size: 22px; font-weight: 700; spacing: 12px; }}
            QRadioButton::indicator {{ width: 22px; height: 22px; border: 2px solid #8197af; border-radius: 13px; background: {BG}; }}
            QRadioButton::indicator:checked {{ border: 7px solid {ACCENT}; width: 12px; height: 12px; background: {INK}; }}
            QPushButton#primary, QPushButton#cancelConfirmation {{ min-width: 108px; min-height: 32px; font-size: 18px; border-radius: 7px; }}
        """)
        bounds = (screen or QGuiApplication.primaryScreen()).availableGeometry()
        width, height = min(960, bounds.width() - 40), min(590, bounds.height() - 40)
        compact = width < 700
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QFrame()
        panel.setObjectName("confirmationPanel")
        outer.addWidget(panel)
        body = QVBoxLayout(panel)
        body.setContentsMargins(24, 24, 24, 24)
        body.setSpacing(24)
        header = QHBoxLayout()
        header.setSpacing(20)
        icon = QLabel("?")
        icon.setObjectName("questionIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(56, 56)
        header.addWidget(icon)
        heading = QVBoxLayout()
        heading.setSpacing(8)
        title = QLabel("确认锁定布局")
        title.setObjectName("confirmationTitle")
        heading.addWidget(title)
        description = QLabel("识别到多个可能的布局，请确认正确的布局后继续。" if len(candidates) > 1
                             else "请核对入口图片，确认正确的布局后继续。")
        description.setObjectName("muted")
        description.setWordWrap(True)
        heading.addWidget(description)
        header.addLayout(heading, 1)
        self.close_button = QPushButton("×")
        self.close_button.setObjectName("windowControl")
        self.close_button.setAccessibleName("关闭确认框")
        self.close_button.setFixedSize(30, 30)
        self.close_button.setAutoDefault(False)
        self.close_button.clicked.connect(self.reject)
        header.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cards_layout = QGridLayout(content)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(18)
        scroll.setWidget(content)
        body.addWidget(scroll, 1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.cards = []
        self.selected_layout = candidates[0]["layout"] if candidates else None
        for index, item in enumerate(candidates):
            layout = item["layout"]
            card = QPushButton()
            card.setObjectName("layoutCandidate")
            card.setCheckable(True)
            card.setAutoDefault(False)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setMinimumSize(0, 240 if compact else 160)
            card.setStyleSheet(f"QPushButton#layoutCandidate {{ min-height: {240 if compact else 160}px; }}")
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            score_text = f"分数 {item['score']:.4f}"
            card.setAccessibleName(f"布局 {layout.removeprefix('layout_')}，{score_text}")
            card_body = QVBoxLayout(card)
            card_body.setContentsMargins(18, 16, 18, 16)
            card_body.setSpacing(14)
            row_widget = QWidget()
            row_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            radio = QRadioButton(f"布局 {layout.removeprefix('layout_')}")
            radio.setAutoExclusive(False)
            radio.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(radio, 1)
            score = QLabel(score_text)
            score.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(score)
            card_body.addWidget(row_widget)
            preview = EntrancePreview(item.get('preview_path') or (entrance_root / f"{layout}.png" if entrance_root else None))
            preview.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            card_body.addWidget(preview, 1)
            if item.get('preview_path'):
                floor_name = {'1st': '一楼', '2nd': '二楼', 'b1': '地下室'}.get(item.get('preview_floor'), '对应楼层')
                card_body.addWidget(QLabel(f'入口图待补，当前展示{floor_name}底图（匹配候选）'))
            def update(checked, *, choice=layout, indicator=radio, badge=score):
                indicator.setChecked(checked)
                badge.setStyleSheet(f"background: {'#254d78' if checked else CARD_2}; "
                                    f"color: {'#79b8ff' if checked else INK}; border-radius: 6px; "
                                    "padding: 5px 12px; font-size: 22px; font-weight: 700;")
                if checked:
                    self.selected_layout = choice
            card.toggled.connect(update)
            update(False)
            self.group.addButton(card)
            self.cards.append(card)
            cards_layout.addWidget(card, index if compact else 0, 0 if compact else index)
            card.setChecked(index == 0)

        footer = QVBoxLayout() if width < 850 else QHBoxLayout()
        footer.setSpacing(18)
        self.confirmation_required = QCheckBox('当前使用人工确认（可在设置中开启自动选择）')
        self.confirmation_required.setChecked(True)
        self.confirmation_required.setToolTip('请核对入口图片后确认布局；匹配分数不是正确概率。')
        self.confirmation_required.setEnabled(False)
        description.setText(('入口图待补，请对照候选楼层底图和布局编号确认。' if any(c.get('preview_path') for c in candidates)
                             else '请核对入口图片后确认布局。') + '匹配分数不是正确概率。'
                            + ('当前包含 50% 缩放候选；确认后请缩到最小再配准。'
                               if any(c.get('recognition_zoom') == '50_percent' for c in candidates) else '')
                            + ('两项分数接近，当前局部结构可能无法区分；可取消并多探索后重试。'
                               if len(candidates)>1 and abs(candidates[0]['score']-candidates[1]['score']) < .02 else ''))
        footer.addWidget(self.confirmation_required)
        if width >= 850:
            footer.addStretch(1)
        actions = QHBoxLayout()
        actions.setSpacing(14)
        actions.addStretch(1)
        self.confirm_button = QPushButton("确定")
        self.confirm_button.setObjectName("primary")
        self.confirm_button.setDefault(True)
        self.confirm_button.setEnabled(bool(candidates))
        self.confirm_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelConfirmation")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.confirm_button)
        actions.addWidget(self.cancel_button)
        footer.addLayout(actions)
        body.addLayout(footer)
        self.resize(width, height)


class Value:
    def __init__(self, value=None, callback=None):
        self.value = value
        self.callback = callback

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        if self.callback is not None:
            self.callback(value)


class HotkeyCaptureButton(QPushButton):
    """Record a primary key with optional Ctrl/Shift/Alt modifiers."""

    SPECIAL_KEYS = {
        Qt.Key.Key_Tab: "Tab", Qt.Key.Key_Space: "Space",
        Qt.Key.Key_Shift: "Shift", Qt.Key.Key_Control: "Ctrl",
        Qt.Key.Key_Alt: "Alt", Qt.Key.Key_CapsLock: "CapsLock",
        Qt.Key.Key_Return: 'Enter',
    }

    def __init__(self, value: str, allowed: tuple[str, ...], on_capture, parent=None, *, allow_modifiers=True):
        super().__init__(value, parent)
        self.setObjectName("hotkeyCapture")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.value = value
        self.allowed = allowed
        self.on_capture = on_capture
        self.capturing = False
        self.allow_modifiers = allow_modifiers
        self.clicked.connect(self.begin_capture)

    def begin_capture(self):
        self.capturing = True
        self.setText('请按快捷键 · Esc 取消' if self.allow_modifiers else '请按地图单键 · Esc 取消')
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    @staticmethod
    def key_name(event):
        key = event.key()
        native = event.nativeVirtualKey()
        if key == Qt.Key.Key_Enter:
            return 'NumEnter'
        native_keypad = next((name for name, vk in KEYPAD_VKS.items() if vk == native), None)
        if native_keypad:
            return native_keypad
        if event.modifiers() & Qt.KeyboardModifier.KeypadModifier:
            if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
                return f'Num{key - Qt.Key.Key_0}'
            return {Qt.Key.Key_Insert: 'Num0', Qt.Key.Key_End: 'Num1', Qt.Key.Key_Down: 'Num2',
                Qt.Key.Key_PageDown: 'Num3', Qt.Key.Key_Left: 'Num4', Qt.Key.Key_Clear: 'Num5',
                Qt.Key.Key_Right: 'Num6', Qt.Key.Key_Home: 'Num7', Qt.Key.Key_Up: 'Num8',
                Qt.Key.Key_PageUp: 'Num9', Qt.Key.Key_Delete: 'NumDecimal',
                Qt.Key.Key_Period: 'NumDecimal', Qt.Key.Key_Comma: 'NumDecimal',
                Qt.Key.Key_Plus: 'NumAdd', Qt.Key.Key_Minus: 'NumSubtract',
                Qt.Key.Key_Asterisk: 'NumMultiply', Qt.Key.Key_Slash: 'NumDivide',
                Qt.Key.Key_Return: 'NumEnter'}.get(key)
        if 0x30 <= native <= 0x39 or 0x41 <= native <= 0x5A:
            return chr(native)
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
            return f"F{key - Qt.Key.Key_F1 + 1}"
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z or Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)
        shifted_digits = {Qt.Key.Key_ParenRight: '0', Qt.Key.Key_Exclam: '1', Qt.Key.Key_At: '2',
                          Qt.Key.Key_NumberSign: '3', Qt.Key.Key_Dollar: '4', Qt.Key.Key_Percent: '5',
                          Qt.Key.Key_AsciiCircum: '6', Qt.Key.Key_Ampersand: '7', Qt.Key.Key_Asterisk: '8',
                          Qt.Key.Key_ParenLeft: '9'}
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and key in shifted_digits:
            return shifted_digits[key]
        return HotkeyCaptureButton.SPECIAL_KEYS.get(key)

    def event(self, event):
        if self.capturing and event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        if self.capturing and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Escape:
                self.capturing = False
                self.setText(self.value)
                return True
            if event.isAutoRepeat():
                return True
            modifiers = event.modifiers()
            flags = sum(flag for qt_flag, flag in (
                (Qt.KeyboardModifier.ControlModifier, 2), (Qt.KeyboardModifier.ShiftModifier, 4),
                (Qt.KeyboardModifier.AltModifier, 1)) if modifiers & qt_flag)
            if modifiers & Qt.KeyboardModifier.MetaModifier:
                self.setText('请使用 Ctrl／Shift／Alt 组合 · Esc 取消')
                return True
            if self.allow_modifiers and key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt):
                prefix = '+'.join(name for name, flag in (('Ctrl', 2), ('Shift', 4), ('Alt', 1)) if flags & flag)
                self.setText(prefix + '+… · Esc 取消')
                return True
            name = self.key_name(event)
            if name not in self.allowed:
                self.setText("不支持此键，请重按 · Esc 取消")
                return True
            if self.allow_modifiers:
                name = Hotkey(name, flags).label
            elif flags & ~{'Ctrl': 2, 'Shift': 4, 'Alt': 1}.get(name, 0):
                self.setText('游戏地图键仅支持单键 · Esc 取消')
                return True
            if self.on_capture(name) is False:
                self.setText("按键冲突，请重按 · Esc 取消")
                return True
            self.value = name
            self.capturing = False
            self.setText(name)
            self.clearFocus()
            return True
        return super().event(event)

    def focusOutEvent(self, event):
        self.cancel_capture()
        super().focusOutEvent(event)

    def cancel_capture(self):
        if self.capturing:
            self.capturing = False
            self.setText(self.value)


class DraggableTitleBar(QFrame):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setObjectName("titlebar")
        self._drag_offset = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window.windowHandle()
            if handle is not None and handle.startSystemMove():
                self._drag_offset = None
            else:
                self._drag_offset = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.window.toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class StatusLabel(QLabel):
    """A QLabel with the tiny configure API used by the processing controller."""

    def configure(self, *, text=None, fg=None, **_kwargs):
        if text is not None:
            self.setText(str(text))
        if fg is not None:
            self.setStyleSheet(f"color: {fg}; background: transparent;")


class ImagePanel(QFrame):
    def __init__(self, title: str, placeholder: str, parent=None):
        super().__init__(parent)
        self.setObjectName("subcard")
        self._pixmap = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 6, 7, 7)
        layout.setSpacing(3)
        self.heading = QLabel(title)
        self.heading.setObjectName("subheading")
        layout.addWidget(self.heading)
        self.image = QLabel(placeholder)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet("background: #1b252e; color: #aeb9c7; border-radius: 4px;")
        self.image.setMinimumSize(0, 80)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumWidth(0)
        layout.addWidget(self.image, 1)
        self.caption = QLabel("")
        self.caption.setObjectName("muted")
        self.caption.setMinimumHeight(20)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

    def set_image(self, bgr: np.ndarray | None, caption: str = ""):
        if bgr is None:
            self._pixmap = None
            self.image.clear()
            self.image.setText("暂无图像")
        else:
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA if bgr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
            height, width = rgba.shape[:2]
            fmt = QImage.Format.Format_RGBA8888 if rgba.shape[2] == 4 else QImage.Format.Format_RGB888
            self._pixmap = QPixmap.fromImage(QImage(rgba.data, width, height, rgba.strides[0], fmt).copy())
            self._redraw()
        self.caption.setText(caption)

    def _redraw(self):
        if self._pixmap is not None:
            self.image.setPixmap(self._pixmap.scaled(
                self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


class ScreenshotViewport(QLabel):
    """Pan a fitted screenshot locally without changing the game or its overlay."""

    def __init__(self, parent=None):
        super().__init__('等待本局截图', parent)
        self.zoom = 1.
        self.pan = QPointF()
        self._drag = None

    def reset_view(self):
        self.zoom = 1.
        self.pan = QPointF()
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#1b252e'))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y())
        painter.scale(self.zoom, self.zoom)
        # QLabel's pixmap is already fitted by ImagePanel, with a device-aware size.
        size = pixmap.deviceIndependentSize()
        painter.drawPixmap(QPointF(-size.width() / 2, -size.height() / 2), pixmap)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.zoom != 1:
            self._drag = (event.position(), QPointF(self.pan))
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self.pan = self._drag[1] + event.position() - self._drag[0]
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom != 1 else Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)


class ScreenshotPanel(ImagePanel):
    def __init__(self, parent=None):
        super().__init__('游戏截图', '等待本局截图', parent)
        old_image = self.image
        self.image = ScreenshotViewport(self)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(0, 80)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setStyleSheet('background: #1b252e; color: #aeb9c7; border-radius: 4px;')
        self.layout().replaceWidget(old_image, self.image)
        old_image.hide()
        old_image.deleteLater()
        tools = QHBoxLayout()
        tools.addStretch(1)
        self.zoom_out = QPushButton('−')
        self.zoom_out.setAccessibleName('缩小游戏截图')
        self.zoom_in = QPushButton('+')
        self.zoom_in.setAccessibleName('放大游戏截图')
        self.zoom_label = QLabel('适应窗口')
        self.zoom_label.setObjectName('muted')
        self.fit_button = QPushButton('复位')
        self.fit_button.setAccessibleName('截图适应窗口')
        self.zoom_out.clicked.connect(lambda: self.set_zoom(self.image.zoom - .25))
        self.zoom_in.clicked.connect(lambda: self.set_zoom(self.image.zoom + .25))
        self.fit_button.clicked.connect(self.reset_view)
        for button in (self.zoom_out, self.zoom_in):
            button.setStyleSheet('padding: 0; min-height: 0;')
            button.setFixedSize(28, 28)
        for widget in (self.zoom_out, self.zoom_label, self.zoom_in, self.fit_button):
            tools.addWidget(widget)
        self.layout().insertLayout(self.layout().count() - 1, tools)
        self.setToolTip('仅调整截图预览，不改变游戏缩放；放大后可拖动查看。')

    def set_zoom(self, value):
        if self._pixmap is None:
            return
        self.image.zoom = max(.5, min(3., value))
        if self.image.zoom == 1:
            self.image.pan = QPointF()
        self.image.setCursor(Qt.CursorShape.OpenHandCursor if self.image.zoom != 1 else Qt.CursorShape.ArrowCursor)
        self.zoom_label.setText('适应窗口' if self.image.zoom == 1 else f'{self.image.zoom:.0%}')
        self.image.update()

    def reset_view(self):
        self.image.reset_view()
        self.zoom_label.setText('适应窗口')

    def set_image(self, bgr, caption=''):
        self.reset_view()
        super().set_image(bgr, caption)


def exclude_window_from_capture(widget: QWidget) -> bool:
    return apply_window_capture_policy(widget)


def physical_to_qt_geometry(left: int, top: int, width: int, height: int):
    """Translate MSS physical pixels to Qt logical coordinates on each monitor."""
    if sys.platform != "win32":
        return left, top, width, height, 1.0

    class MonitorInfo(ctypes.Structure):
        _fields_ = (("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32))

    user32 = ctypes.windll.user32
    monitor_from_point = user32.MonitorFromPoint
    monitor_from_point.argtypes = (wintypes.POINT, wintypes.DWORD)
    monitor_from_point.restype = wintypes.HANDLE
    get_monitor_info = user32.GetMonitorInfoW
    get_monitor_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(MonitorInfo))
    get_monitor_info.restype = wintypes.BOOL
    monitor = monitor_from_point(wintypes.POINT(left + width // 2, top + height // 2), 2)
    info = MonitorInfo()
    info.cbSize = ctypes.sizeof(MonitorInfo)
    if not monitor or not get_monitor_info(monitor, ctypes.byref(info)):
        raise ctypes.WinError()
    name = info.szDevice.casefold().removeprefix("\\\\.\\")
    screens = QGuiApplication.screens()
    screen = next((item for item in screens if item.name().casefold().removeprefix("\\\\.\\") == name), None)
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    scale = screen.devicePixelRatio()
    geometry = screen.geometry()
    return (geometry.x() + round((left - info.rcMonitor.left) / scale),
            geometry.y() + round((top - info.rcMonitor.top) / scale),
            max(1, round(width / scale)), max(1, round(height / scale)), scale)


class NotificationToast(QWidget):
    """Transparent, non-interactive notification shown above the game window."""

    HOLD_MS = 3200
    FADE_MS = 600

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(620)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        self.panel = QFrame(self)
        self.panel.setStyleSheet("background: transparent; border: none;")
        outer.addWidget(self.panel)
        content = QVBoxLayout(self.panel)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
        self.title = QLabel(self.panel)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(f"font-size: 25px; font-weight: 700; color: {GREEN}; background: transparent;")
        content.addWidget(self.title)
        self.detail = QLabel(self.panel)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"font-size: 16px; color: {INK}; background: transparent;")
        content.addWidget(self.detail)
        for label in (self.title, self.detail):
            shadow = QGraphicsDropShadowEffect(label)
            shadow.setBlurRadius(12)
            shadow.setOffset(0, 2)
            shadow.setColor(QColor(0, 0, 0, 230))
            label.setGraphicsEffect(shadow)
        # A graphics effect on the top-level translucent window can remain
        # unpainted until its opacity changes. Animate native window opacity.
        self.fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade.finished.connect(self.hide)
        self._sequence = 0

    def show_message(self, title: str, detail: str, *, success: bool,
                     roi: list[int] | None = None, hold_ms: int | None = None,
                     fade_ms: int | None = None) -> None:
        hold_ms = self.HOLD_MS if hold_ms is None else hold_ms
        fade_ms = self.FADE_MS if fade_ms is None else fade_ms
        self._sequence += 1
        sequence = self._sequence
        self.fade.stop()
        self.title.setText(title)
        self.detail.setText(detail)
        self.title.setStyleSheet(
            f"font-size: 25px; font-weight: 700; color: {GREEN if success else BAD}; "
            "background: transparent;"
        )
        self.setWindowOpacity(1.0)
        screen = QGuiApplication.primaryScreen()
        if roi and len(roi) == 4:
            try:
                x, y, width, height, _scale = physical_to_qt_geometry(*map(int, roi))
                screen = QGuiApplication.screenAt(QPoint(x + width // 2, y + height // 2)) or screen
            except (OSError, ValueError):
                pass
        bounds = screen.geometry()
        self.setFixedWidth(min(620, max(200, bounds.width() - 16)))
        self.adjustSize()
        x = bounds.left() + (bounds.width() - self.width()) // 2
        y = bounds.top() + round(bounds.height() * 0.10)
        self.move(x, y)
        self.show()
        self.raise_()
        exclude_window_from_capture(self)

        def start_fade() -> None:
            if sequence != self._sequence or not self.isVisible():
                return
            self.fade.setDuration(fade_ms)
            self.fade.setStartValue(1.0)
            self.fade.setEndValue(0.0)
            self.fade.start()

        QTimer.singleShot(hold_ms, start_fade)


class RouteOverlayWindow(QWidget):
    """Transparent, mouse-through Qt window positioned over the game's map ROI."""

    def __init__(self, parent=None, *, clear_before_hide: bool = True):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._clear_before_hide = clear_before_hide
        self.capture_excluded = False
        self._pan_pixmap = None
        self._pan_offset = (0., 0.)
        self.label = QLabel(self)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.visible = False

    def show(self, bgra: np.ndarray, left: int = 0, top: int = 0):
        self._pan_pixmap = None
        self.label.show()
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("覆盖层必须为 BGRA 四通道图片")
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        x, y, logical_width, logical_height, scale = physical_to_qt_geometry(int(left), int(top), width, height)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(scale)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(0, 0, logical_width, logical_height)
        self.setGeometry(x, y, logical_width, logical_height)
        if not self.isVisible():
            self._prepare_native_window()
            super().show()
        self.visible = True

    def set_pan_image(self, rgba, origin, roi):
        """Cache the complete scaled source inside a fixed, clipping viewport."""
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        x, y, w, h, scale = physical_to_qt_geometry(*map(int, roi))
        self._pan_scale = scale
        self._pan_origin = tuple(origin)
        self._pan_offset = (0., 0.)
        self._pan_pixmap = QPixmap.fromImage(image)
        self._pan_pixmap.setDevicePixelRatio(scale)
        self.label.clear()
        self.label.hide()
        self.setGeometry(x, y, w, h)
        self.update()

    def set_pan_offset(self, dx, dy):
        if self._pan_pixmap is not None and (dx, dy) != self._pan_offset:
            self._pan_offset = (float(dx), float(dy))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pan_pixmap is not None:
            painter = QPainter(self)
            painter.setClipRect(self.rect())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            x = (self._pan_origin[0] + self._pan_offset[0]) / self._pan_scale
            y = (self._pan_origin[1] + self._pan_offset[1]) / self._pan_scale
            painter.drawPixmap(QPointF(x, y), self._pan_pixmap)
            painter.end()

    def _prepare_native_window(self):
        # Apply native styles before exposing the window's first frame.
        if sys.platform == "win32":
            hwnd = int(self.winId())
            get_style = ctypes.windll.user32.GetWindowLongW
            set_style = ctypes.windll.user32.SetWindowLongW
            get_style.argtypes = (wintypes.HWND, ctypes.c_int)
            get_style.restype = ctypes.c_long
            set_style.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
            set_style.restype = ctypes.c_long
            style = get_style(hwnd, -20)
            set_style(hwnd, -20, style | 0x20 | 0x80 | 0x08000000)
            self.capture_excluded = exclude_window_from_capture(self)

    def hide(self):
        if self._clear_before_hide:
            self._pan_pixmap = None
            self.label.clear()
            if self.isVisible():
                # Clear the previous surface while painting is still possible.
                # Hidden widgets cannot repaint; update() alone would be too late.
                self.repaint()
        super().hide()
        self.visible = False

    def destroy(self):
        self.close()
        self.visible = False


class ReferenceMapWindow(RouteOverlayWindow):
    """Unregistered floor reference, separate from the positioned route overlay."""

    def __init__(self):
        super().__init__(clear_before_hide=False)
        self.caption = QLabel(self)
        self.caption.setStyleSheet("color: white; background: #172331; padding: 3px; font-size: 12px;")

    def show_map(self, image: np.ndarray, roi: list[int], caption: str, opacity: float):
        width = 460
        scale = width / image.shape[1]
        thumb = cv2.resize(image, (width, max(1, round(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR)
        canvas = np.full((thumb.shape[0] + 54, thumb.shape[1], 4), 255, np.uint8)
        canvas[:54, :, :3] = (49, 35, 23)
        canvas[54:, :, :3] = thumb
        self.setWindowOpacity(opacity)
        super().show(canvas, 0, 0)
        self.caption.setText(caption)
        self.caption.setGeometry(0, 0, self.width(), max(36, round(54 / self.devicePixelRatioF())))
        self.caption.raise_()


class ChestCounterWindow(RouteOverlayWindow):
    """A mouse-through counter that stays visible while the game is foreground."""

    def __init__(self):
        super().__init__(clear_before_hide=False)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            'background: transparent; color: white; font-family: "Microsoft YaHei UI"; font-size: 16px;')

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(16, 26, 39, 225))
        painter.setPen(QColor('#526980'))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)
        painter.end()

    def show_counts(self, gold, purple, client):
        x, y, width, height, _ = physical_to_qt_geometry(*client.as_list())
        self.setGeometry(x + max(0, width - 296), y + min(100, max(0, height - 54)), 280, 54)
        self.label.setGeometry(self.rect())
        self.label.setText(f'已开启　<span style="color:#f5c879">金箱 {gold}</span>'
                           f'　<span style="color:#d9a0ff">紫箱 {purple}</span>')
        if not self.isVisible():
            self._prepare_native_window()
            QWidget.show(self)
        self.visible = True


class SettingsPage(QWidget):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.vars = {}
        self._settings_dirty = False
        self._settings_save_job = None
        self.save_status = Value("所有修改都会自动保存并立即生效", lambda text: self.status.setText(text))
        self.opacity_text = Value("", lambda text: self.opacity_label.setText(text))
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 18, 22, 18)
        title = QLabel("偏好设置")
        title.setObjectName("section")
        outer.addWidget(title)
        hint = QLabel('点击快捷键按钮，再按目标按键；程序操作可使用 Ctrl／Shift／Alt 组合，地图键保持单键。修改会自动保存。')
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        outer.addWidget(hint)
        self.status = QLabel(self.save_status.get())
        self.status.setObjectName("good")
        outer.addWidget(self.status)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.forms = {}
        self.key_forms = {}
        for key, label in (('general', '识别与显示'), ('keys', '快捷键'), ('capture', '截图设置')):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            content = QWidget()
            scroll.setWidget(content)
            body = QVBoxLayout(content)
            body.setContentsMargins(0, 14, 0, 0)
            if key == 'keys':
                columns = QHBoxLayout()
                columns.setSpacing(18)
                body.addLayout(columns)
                for group, title in (('map', '地图与路线'), ('round', '本局与计数')):
                    card = QFrame()
                    card.setObjectName('card')
                    column = QVBoxLayout(card)
                    column.setContentsMargins(20, 20, 20, 20)
                    column.setSpacing(20)
                    heading = QLabel(title)
                    heading.setObjectName('subheading')
                    column.addWidget(heading)
                    fields = QFormLayout()
                    fields.setSpacing(18)
                    fields.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                    column.addLayout(fields)
                    column.addStretch(1)
                    self.key_forms[group] = fields
                    columns.addWidget(card, 1)
                self.forms[key] = self.key_forms['map']
                self.keys_body = body
                body.addStretch(1)
                self.tabs.addTab(scroll, label)
                continue
            form_box = QFrame()
            form_box.setObjectName('card')
            form = QFormLayout(form_box)
            form.setContentsMargins(22, 20, 22, 20)
            form.setSpacing(18)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            body.addWidget(form_box)
            body.addStretch(1)
            self.forms[key] = form
            self.tabs.addTab(scroll, label)
        form = self.forms['general']

        settings = self.app.settings
        def entry(name, value):
            widget = QLineEdit(value)
            self.vars[name] = Value(value)
            widget.textChanged.connect(lambda text, key=name: (self.vars[key].set(text), self._schedule_settings_save()))
            return widget

        def check(name, text, value):
            widget = QCheckBox(text)
            widget.setChecked(value)
            self.vars[name] = Value(value)
            widget.toggled.connect(lambda checked, key=name: (self.vars[key].set(checked), self._schedule_settings_save()))
            return widget

        def capture(name, allowed, value):
            self.vars[name] = Value(value)
            button = HotkeyCaptureButton(value, allowed, lambda key: self._capture_key(name, key),
                                         allow_modifiers=name != 'map_key')
            self.hotkey_buttons[name] = button
            return button

        self.hotkey_buttons = {}
        self.layout_recognition_combo = QComboBox()
        for key, label in BACKEND_LABELS.items():
            self.layout_recognition_combo.addItem(label, key)
        self.layout_recognition_combo.setCurrentIndex(
            self.layout_recognition_combo.findData(settings['layout_recognition_backend']))
        self.vars['layout_recognition_backend'] = Value(settings['layout_recognition_backend'])
        self.layout_recognition_combo.currentIndexChanged.connect(lambda _index: (
            self.vars['layout_recognition_backend'].set(self.layout_recognition_combo.currentData()),
            self._update_recognition_controls(), self._schedule_settings_save()))
        form.addRow('布局识别方法', self.layout_recognition_combo)
        self.auto_select_layout_check = check('auto_select_layout', '自动选择识别布局，无需确认', settings['auto_select_layout'])
        self.auto_select_layout_check.setToolTip('默认关闭。开启后按最高匹配分数选择，可能选错；分数不是正确概率。可重新识别或手动改选布局。')
        form.addRow('布局选择', self.auto_select_layout_check)
        self.recognition_note = QLabel()
        self.recognition_note.setWordWrap(True)
        form.addRow('', self.recognition_note)
        form = self.forms['keys']
        form.addRow("游戏地图键", capture("map_key", MAP_KEYS, settings["map_key"]))
        form.addRow("识别快捷键", capture("recognize_hotkey", ACTION_KEYS, settings["recognize_hotkey"]))
        form.addRow("投影快捷键", capture("project_hotkey", ACTION_KEYS, settings["project_hotkey"]))
        form.addRow("常驻参考图显隐", capture("reference_hotkey", ACTION_KEYS, settings["reference_hotkey"]))
        form.addRow('参考图下一层', capture('reference_floor_hotkey', ACTION_KEYS, settings['reference_floor_hotkey']))
        reference_note = QLabel('H 显隐、J 切层为默认按键，可修改。配准成功后同步参考楼层，仍可用 J 手动切换；关地图后常驻，不影响投影，也不持续追踪角色楼层。')
        reference_note.setWordWrap(True)
        self.keys_body.insertWidget(self.keys_body.count() - 1, reference_note)
        form = self.key_forms['round']
        form.addRow('下一局', capture('next_round_hotkey', ACTION_KEYS, settings['next_round_hotkey']))
        next_round_note = QLabel('下一局快捷键仅在游戏前台生效；请先关闭游戏地图。换局会清零计数、重置布局和取消旧任务，撤销仅恢复计数。')
        next_round_note.setObjectName('muted')
        next_round_note.setWordWrap(True)
        self.keys_body.insertWidget(self.keys_body.count() - 1, next_round_note)
        for action, (label, _, _, _) in CHEST_ACTIONS.items():
            field = f'{action}_hotkey'
            form.addRow('宝箱计数 · ' + label, capture(field, ACTION_KEYS, settings[field]))
        form = self.forms['general']
        form.addRow('宝箱计数条', check('chest_counter_visible', '游戏前台时显示（关地图后仍显示）', settings['chest_counter_visible']))
        self.capture_exclusion_check = check('capture_exclusion_enabled', '在截图和录屏中隐藏本程序', settings['capture_exclusion_enabled'])
        form.addRow('防捕获', self.capture_exclusion_check)
        self.capture_status_label = QLabel(self.app.capture_policy.status)
        self.capture_status_label.setWordWrap(True)
        form.addRow('', self.capture_status_label)
        capture_note = QLabel('关闭后允许捕获；OBS 游戏捕获未必包含独立悬浮窗。内部截图期间会临时排除本程序，录屏中可能短暂消隐。')
        capture_note.setWordWrap(True)
        form.addRow('', capture_note)
        form.addRow("自动投影", check("auto_project", "地图打开时自动配准并投影", settings["auto_project"]))
        form.addRow("地图拖动跟随（试用）", check("template_motion_tracking", "拖动地图时路线同步平移（不支持缩放）", settings.get("template_motion_tracking", True)))
        form = self.forms['capture']
        self.map_capture_delay_spin = QSpinBox()
        self.map_capture_delay_spin.setRange(0, MAX_MAP_CAPTURE_DELAY_MS)
        self.map_capture_delay_spin.setSuffix(" ms")
        self.map_capture_delay_spin.setKeyboardTracking(False)
        self.map_capture_delay_spin.setValue(map_capture_delay_ms(settings))
        self.map_capture_delay_spin.setToolTip(f"默认 {DEFAULT_MAP_CAPTURE_DELAY_MS} ms。从松开地图键开始计算首次截图等待，修改自动保存，下次开图生效。")
        self.vars["map_capture_delay_ms"] = Value(self.map_capture_delay_spin.value())
        self.map_capture_delay_spin.valueChanged.connect(lambda value: (
            self.vars["map_capture_delay_ms"].set(value), self._schedule_settings_save()))
        form.addRow("截图延迟", self.map_capture_delay_spin)
        form.addRow("配准方式", QLabel("定尺度模板匹配 · 单层自动匹配"))
        form.addRow("游戏窗口标题", entry("window_titles", ", ".join(settings["game_window_titles"])))
        form.addRow("截图区域 左, 上, 宽, 高", entry("roi", ", ".join(map(str, settings["capture_roi"]))))
        form.addRow("参考分辨率 宽, 高", entry("reference_size", ", ".join(map(str, settings["capture_reference_size"]))))
        form = self.forms['general']
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(round(settings["opacity"] * 100))
        self.vars["opacity"] = Value(slider.value())
        slider.valueChanged.connect(lambda value: (self.vars["opacity"].set(value), self._update_opacity_text(), self._schedule_settings_save()))
        self.opacity_label = QLabel()
        self._update_opacity_text()
        opacity_layout.addWidget(slider, 1)
        opacity_layout.addWidget(self.opacity_label)
        form.addRow("投影不透明度", opacity_row)

        close_button = QPushButton("←  返回首页")
        close_button.clicked.connect(self.close)
        outer.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.tabs.currentChanged.connect(self._cancel_key_capture)
        self._update_recognition_controls()

    def _cancel_key_capture(self, *_args):
        for button in self.hotkey_buttons.values():
            button.cancel_capture()

    def _update_recognition_controls(self):
        if self.app.profile.key == 'a_bringer_of_doom/hard':
            self.recognition_note.setText(
                ('困难支持最小或 50% 缩放识别，CPU 比较各层底图。' if self.layout_recognition_combo.currentData() == 'pure_structure'
                 else '困难支持最小或 50% 缩放的二楼图标与结构识别。')
                + '选定布局后请缩到最小再配准；识别尺度不会用于投影。默认人工确认，开启自动选择后采用首选候选。匹配分数不是正确概率。')
            return
        self.recognition_note.setText(
            '请将游戏地图缩到最小再识别；CPU 搜索当前难度的各层底图，不自动适配其他缩放。默认人工确认，可开启自动选择；分数不是正确概率。'
            if self.layout_recognition_combo.currentData() == 'pure_structure' else
            '请在二楼地图识别，选定布局后可切层配准投影。默认人工确认，可开启自动选择；匹配分数不是正确概率。')

    def _capture_key(self, name: str, key: str) -> bool:
        names = ("map_key", "recognize_hotkey", "project_hotkey", "reference_hotkey", 'reference_floor_hotkey', 'next_round_hotkey', *CHEST_HOTKEY_FIELDS)
        values = {field: key if field == name else self.vars[field].get() for field in names}
        try:
            validate_action_hotkeys(values['map_key'], {field: value for field, value in values.items() if field != 'map_key'})
        except ValueError as exc:
            self.save_status.set(str(exc))
            return False
        self.vars[name].set(key)
        self._schedule_settings_save()
        return True

    def _update_opacity_text(self):
        self.opacity_text.set(f"{self.vars['opacity'].get()}% 不透明")

    def _schedule_settings_save(self):
        self._settings_dirty = True
        self.save_status.set("修改待保存…")
        if self._settings_save_job is not None:
            self._settings_save_job.stop()
        self._settings_save_job = QTimer(self)
        self._settings_save_job.setSingleShot(True)
        self._settings_save_job.timeout.connect(self._persist_settings)
        self._settings_save_job.start(600)

    def _collect_settings(self):
        fields = ('recognize_hotkey', 'project_hotkey', 'reference_hotkey', 'reference_floor_hotkey', 'next_round_hotkey', *CHEST_HOTKEY_FIELDS)
        shortcuts = validate_action_hotkeys(self.vars['map_key'].get(), {
            name: self.vars[name].get() if name in self.vars else self.app.settings[name] for name in fields})
        roi = [int(part.strip()) for part in self.vars["roi"].get().split(",")]
        CaptureRegion.from_value(roi)
        reference = [int(part.strip()) for part in self.vars["reference_size"].get().split(",")]
        if len(reference) != 2 or any(number <= 0 for number in reference):
            raise ValueError("游戏参考分辨率必须为：宽, 高")
        from src.route_projection_core import game_capture_region, percent_to_opacity
        game_capture_region(CaptureRegion(0, 0, *reference), roi, reference)
        updated = dict(self.app.settings)
        updated.update({name: value.get() for name, value in self.vars.items()
                        if name not in ("roi", "reference_size", "window_titles", "opacity")})
        updated.update(shortcuts)
        updated["capture_roi"] = roi
        updated["capture_reference_size"] = reference
        updated["game_window_titles"] = [part.strip() for part in self.vars["window_titles"].get().split(",") if part.strip()]
        if not updated["game_window_titles"]:
            raise ValueError("请填写游戏窗口标题关键词，以便只截取游戏画面")
        updated["opacity"] = percent_to_opacity(round(self.vars["opacity"].get()))
        updated.update(registration_backend="template", layout_recognition_backend=self.vars["layout_recognition_backend"].get(),
                       auto_floor=False, skip_layout_confirmation=False)
        return updated

    def _persist_settings(self, *, show_error=False):
        if self._settings_save_job is not None:
            self._settings_save_job.stop()
            self._settings_save_job = None
        try:
            updated = self._collect_settings()
            self.app.apply_settings(updated)
            persisted = load_settings(self.app.catalog, self.app.settings_path)
            fields = ("map_key", "game_window_titles", "auto_project", "auto_floor", "template_motion_tracking", "map_capture_delay_ms", "recognize_hotkey",
                      "project_hotkey", "reference_hotkey", "reference_opacity", "capture_roi", "capture_reference_size", "route_versions", "opacity", "skip_layout_confirmation", "auto_select_layout", "layout_recognition_backend",
                      *CHEST_HOTKEY_FIELDS, 'reference_floor_hotkey', 'next_round_hotkey', 'chest_counter_visible', 'capture_exclusion_enabled')
            if any(persisted[name] != updated[name] for name in fields):
                raise OSError(f"设置未完整写入配置文件：{self.app.settings_path}")
        except Exception as exc:
            self.save_status.set(f"尚未保存：{exc}")
            if show_error:
                QMessageBox.critical(self, "设置无效", str(exc))
            return False
        self._settings_dirty = False
        self.opacity_text.set(f"{round(updated['opacity'] * 100)}% 已保存")
        self.save_status.set("已自动保存并立即生效")
        return True

    def closeEvent(self, event):
        if (self._settings_dirty or self._settings_save_job is not None) and not self._persist_settings(show_error=True):
            event.ignore()
            return
        self.app._leave_settings(self)
        event.accept()
