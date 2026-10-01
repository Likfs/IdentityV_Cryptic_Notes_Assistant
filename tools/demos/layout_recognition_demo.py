from __future__ import annotations

import ctypes
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import mss
from mss.exception import ScreenShotError
from PIL import Image, ImageTk

from src.capture.region import CaptureRegion, enable_physical_pixel_coordinates
from tools.common.capture_config import DEFAULT_CONFIG_PATH as DEFAULT_CAPTURE_CONFIG, load_capture_config
from src.layout_classifier import explored_ratio, load_classifier, predict_image, summarize_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = DEFAULT_CAPTURE_CONFIG
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'workspace/checkpoints/layout_classifier.pt'
PRE_RESIZED_MODEL_PATH = PROJECT_ROOT / "models/releases/layout_classifier_pre_resized.pt"
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000


class RecognitionHotkey:
    HOTKEY_ID = 0x6201
    VK_F9 = 0x78
    LABEL = "F9"

    def __init__(self) -> None:
        self.events: queue.Queue[str] = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.thread_id: Optional[int] = None
        self.ready = False
        self.error = ""
        self._stop_event = threading.Event()

    def start(self) -> None:
        if sys.platform != "win32":
            self.error = "Windows 全局快捷键不可用；窗口激活时仍可使用 F9。"
            return
        self.thread = threading.Thread(target=self._run, name="layout-demo-hotkey", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.thread_id is not None:
            try:
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
            except Exception:
                pass

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self.thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        if not user32.RegisterHotKey(None, self.HOTKEY_ID, MOD_NOREPEAT, self.VK_F9):
            self.error = "F9 注册失败，可能被其他程序占用。"
            self.thread_id = None
            return
        self.ready = True
        message = wintypes.MSG()
        try:
            while not self._stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                if message.message == WM_HOTKEY and int(message.wParam) == self.HOTKEY_ID:
                    self.events.put("recognize")
        finally:
            user32.UnregisterHotKey(None, self.HOTKEY_ID)
            self.ready = False
            self.thread_id = None


def layout_display_name(label: str) -> str:
    match = re.fullmatch(r"layout_(\d+)", label)
    return f"布局 {int(match.group(1)):02d}" if match else label


def available_model_options(
    current_path: Path,
    original_path: Path = DEFAULT_MODEL_PATH,
    pre_resized_path: Path = PRE_RESIZED_MODEL_PATH,
    experiment_root: Path = PROJECT_ROOT / 'workspace/reports/classification',
) -> dict[str, Path]:
    """List independently trained models without treating historical backups as active choices."""
    choices: dict[str, Path] = {}
    for label, path in (
        ("原图增强（正式模型）", original_path),
        ("预缩放后增强", pre_resized_path),
    ):
        if path.is_file():
            choices[label] = path.resolve()
    if not pre_resized_path.is_file():
        experimental = sorted(
            experiment_root.glob("resize_cache_ab_*/pre_resized_best.pt"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if experimental:
            choices["预缩放实验模型（仅验证）"] = experimental[0].resolve()
    resolved_current = current_path.resolve()
    if resolved_current.is_file() and resolved_current not in choices.values():
        choices[f"自定义：{resolved_current.name}"] = resolved_current
    return choices


def model_coverage_text(checkpoint: dict, project_root: Path = PROJECT_ROOT) -> str:
    """Describe the exact checkpoint classes and archived classes it cannot predict."""
    labels = [str(label) for label in checkpoint["labels"]]
    trained = set(labels)
    numbers = [int(match.group(1)) for label in labels if (match := re.fullmatch(r"layout_(\d+)", label))]
    gaps = {f"layout_{number:02d}" for number in range(1, max(numbers, default=0) + 1)} - trained
    roi_dir = project_root / 'workspace/datasets/layout_classification/images' / str(checkpoint.get("map_id", "")) / str(
        checkpoint.get("difficulty", "")
    )
    archived = {path.name for path in roi_dir.glob("layout_*") if path.is_dir()}
    unavailable = sorted(gaps | (archived - trained))
    text = f"模型支持（{len(labels)} 类）：{'、'.join(labels)}"
    if unavailable:
        details = []
        for label in unavailable:
            has_images = any((roi_dir / label).glob("*.png"))
            details.append(f"{label}{'（未训练）' if has_images else '（无样本）'}")
        text += f"；当前不能识别：{'、'.join(details)}"
    return text


def capture_region(region: CaptureRegion) -> Image.Image:
    try:
        with mss.MSS() as screen:
            shot = screen.grab(region.as_mss_monitor())
    except ScreenShotError as exc:
        raise RuntimeError(
            "Windows 截图失败。请确认屏幕没有锁定，并从已登录的桌面会话运行 Demo。"
        ) from exc
    image = Image.frombytes("RGB", shot.size, shot.rgb)
    if image.size != (region.width, region.height):
        raise RuntimeError(f"截图尺寸异常：得到 {image.size}，预期 {(region.width, region.height)}")
    return image


class LayoutRecognitionDemo(tk.Tk):
    PALETTE = ("#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c")

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        super().__init__()
        self.config_path = config_path.resolve()
        self.model_path = model_path.resolve()
        config = load_capture_config(self.config_path)
        self.region = CaptureRegion.from_value(config["capture_roi"])
        self.model, self.checkpoint, self.device = load_classifier(self.model_path)
        self.labels = [str(label) for label in self.checkpoint["labels"]]
        if not self.labels:
            raise ValueError("模型不包含任何布局类别")
        self.model_choices = available_model_options(self.model_path)
        initial_choice = next(
            (label for label, path in self.model_choices.items() if path == self.model_path),
            self.model_path.name,
        )

        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.result_queue: queue.Queue[tuple] = queue.Queue()
        self.recognition_busy = False
        self.closed = False
        self.hotkey = RecognitionHotkey()

        self.result_var = tk.StringVar(value="等待识别")
        self.detail_var = tk.StringVar(value="打开游戏地图后，按 F9 进行一次识别。")
        self.status_var = tk.StringVar(value="模型已载入，正在注册全局快捷键……")
        self.hotkey_status_var = tk.StringVar(value="全局 F9：注册中……")
        self.capture_info_var = tk.StringVar(
            value=f"ROI：左 {self.region.left} / 上 {self.region.top} / {self.region.width} × {self.region.height}"
        )
        self.model_choice_var = tk.StringVar(value=initial_choice)
        self.model_meta_var = tk.StringVar(value=f"{initial_choice} · 厄运之女 / 噩梦 · {len(self.labels)} 个布局")
        self.model_coverage_var = tk.StringVar(value=model_coverage_text(self.checkpoint))
        self.thresholds_var = tk.StringVar(value=self._thresholds_text())
        self.ranking_rows: list[dict] = []

        self.title("当前屏幕布局识别 Demo")
        self.geometry("1180x790+35+35")
        self.minsize(1020, 680)
        self._build_ui()
        if sys.platform != "win32":
            self.bind("<F9>", lambda _event: self.request_recognition())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.hotkey.start()
        self.after(50, self._poll_events)
        self.after(700, self._show_hotkey_status)
        self.after(450, self.request_recognition)

    def _build_ui(self) -> None:
        self.configure(background="#eef2f7")
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        title = ttk.Label(header, text="当前屏幕布局识别", font=("Microsoft YaHei UI", 19, "bold"))
        title.pack(side="left")
        ttk.Button(header, text="立即识别  F9", command=self.request_recognition).pack(side="right")
        ttk.Button(header, text="重新载入模型", command=self.reload_model).pack(side="right", padx=(0, 8))

        model_controls = ttk.Frame(root)
        model_controls.pack(fill="x", pady=(0, 8))
        ttk.Label(model_controls, text="推理模型：").pack(side="left")
        self.model_choice_box = ttk.Combobox(
            model_controls,
            textvariable=self.model_choice_var,
            values=list(self.model_choices),
            state="readonly",
            width=32,
        )
        self.model_choice_box.pack(side="left", padx=(0, 8))
        self.model_choice_box.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(model_controls, text="刷新模型列表", command=self.refresh_model_choices).pack(side="left")
        ttk.Button(model_controls, text="选择其他模型文件…", command=self.select_model_file).pack(
            side="left", padx=(8, 0)
        )

        meta = ttk.Frame(root)
        meta.pack(fill="x", pady=(0, 8))
        ttk.Label(meta, textvariable=self.model_meta_var, font=("Microsoft YaHei UI", 10)).pack(side="left")
        ttk.Label(meta, textvariable=self.hotkey_status_var).pack(side="right")
        ttk.Label(root, textvariable=self.model_coverage_var, wraplength=1120, justify="left").pack(
            fill="x", pady=(0, 8)
        )

        content = ttk.Panedwindow(root, orient="horizontal")
        content.pack(fill="both", expand=True)

        preview_frame = ttk.LabelFrame(content, text="当前固定区域截图", padding=8)
        ranking_frame = ttk.LabelFrame(content, text="布局置信度排序", padding=12)
        content.add(preview_frame, weight=3)
        content.add(ranking_frame, weight=2)

        self.preview_label = tk.Label(
            preview_frame,
            text="等待首次截图……",
            bg="#17202b",
            fg="#cbd5e1",
            font=("Microsoft YaHei UI", 13),
            anchor="center",
        )
        self.preview_label.pack(fill="both", expand=True)
        ttk.Label(preview_frame, textvariable=self.capture_info_var, anchor="w").pack(fill="x", pady=(7, 0))

        self.result_banner = tk.Label(
            ranking_frame,
            textvariable=self.result_var,
            bg="#334155",
            fg="white",
            font=("Microsoft YaHei UI", 18, "bold"),
            padx=10,
            pady=12,
        )
        self.result_banner.pack(fill="x")
        ttk.Label(
            ranking_frame,
            textvariable=self.detail_var,
            wraplength=330,
            justify="left",
        ).pack(fill="x", pady=(8, 14))

        self.ranking_container = ttk.Frame(ranking_frame)
        self.ranking_container.pack(fill="both", expand=True)
        self._build_ranking_rows()

        ttk.Label(ranking_frame, textvariable=self.thresholds_var, wraplength=330, foreground="#475569").pack(
            fill="x", pady=(15, 0)
        )

        ttk.Separator(root).pack(fill="x", pady=(10, 7))
        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x")

    def _build_ranking_rows(self) -> None:
        for child in self.ranking_container.winfo_children():
            child.destroy()
        self.ranking_rows.clear()
        for index in range(len(self.labels)):
            row = ttk.Frame(self.ranking_container)
            row.pack(fill="x", pady=6)
            top = ttk.Frame(row)
            top.pack(fill="x")
            rank_var = tk.StringVar(value=f"{index + 1}.")
            label_var = tk.StringVar(value="—")
            probability_var = tk.StringVar(value="0.0%")
            ttk.Label(top, textvariable=rank_var, width=3).pack(side="left")
            ttk.Label(top, textvariable=label_var, font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
            ttk.Label(top, textvariable=probability_var).pack(side="right")
            bar = tk.Canvas(row, height=15, bg="#dbe3ec", highlightthickness=0)
            bar.pack(fill="x", pady=(4, 0))
            self.ranking_rows.append(
                {
                    "rank": rank_var,
                    "label": label_var,
                    "probability": probability_var,
                    "bar": bar,
                }
            )

    def _thresholds_text(self) -> str:
        return (
            f"接受条件：探索 ≥ {float(self.checkpoint.get('minimum_explored_ratio', 0.10)):.0%}，"
            f"置信度 ≥ {float(self.checkpoint.get('confidence_threshold', 0.70)):.0%}，"
            f"前两名差值 ≥ {float(self.checkpoint.get('margin_threshold', 0.20)):.0%}"
        )

    def refresh_model_choices(self) -> None:
        self.model_choices = available_model_options(self.model_path)
        self.model_choice_box.configure(values=list(self.model_choices))
        label = next(
            (name for name, path in self.model_choices.items() if path == self.model_path),
            self.model_path.name,
        )
        self.model_choice_var.set(label)

    def _on_model_selected(self, _event=None) -> None:
        selected = self.model_choices.get(self.model_choice_var.get())
        if selected is not None:
            self.reload_model(selected)

    def select_model_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择布局分类模型",
            initialdir=str(self.model_path.parent),
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            self.reload_model(Path(path))

    def reload_model(self, selected_path: Optional[Path] = None) -> None:
        if self.recognition_busy:
            self.status_var.set("正在识别，请稍后再重新载入模型。")
            self.refresh_model_choices()
            return
        target_path = (selected_path or self.model_path).resolve()
        try:
            model, checkpoint, device = load_classifier(target_path)
            labels = [str(label) for label in checkpoint["labels"]]
            if not labels:
                raise ValueError("模型不包含任何布局类别")
        except Exception as exc:
            self.refresh_model_choices()
            self.status_var.set(f"模型载入失败，仍使用原模型：{exc}")
            return
        self.model_path = target_path
        self.model, self.checkpoint, self.device, self.labels = model, checkpoint, device, labels
        self.refresh_model_choices()
        self.model_meta_var.set(f"{self.model_choice_var.get()} · 厄运之女 / 噩梦 · {len(labels)} 个布局")
        self.model_coverage_var.set(model_coverage_text(checkpoint))
        self.thresholds_var.set(self._thresholds_text())
        self._build_ranking_rows()
        self.result_var.set("等待识别")
        self.result_banner.configure(bg="#334155")
        self.detail_var.set("模型已更新。打开游戏地图后按 F9 识别。")
        self.status_var.set(f"模型已载入：{target_path.name} · {len(labels)} 类 · 设备 {device}")
        self.request_recognition()

    def request_recognition(self) -> None:
        if self.recognition_busy or self.closed:
            return
        self.recognition_busy = True
        self.status_var.set("正在截取当前屏幕并识别……")
        worker = threading.Thread(target=self._recognize_worker, name="layout-demo-inference", daemon=True)
        worker.start()

    def _recognize_worker(self) -> None:
        started = time.perf_counter()
        try:
            image = capture_region(self.region)
            model, checkpoint, device = self.model, self.checkpoint, self.device
            ranking = predict_image(model, checkpoint, device, image)
            summary = summarize_prediction(ranking, checkpoint, explored_ratio(image))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.result_queue.put(("success", image, ranking, summary, elapsed_ms, datetime.now()))
        except Exception as exc:
            self.result_queue.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                self.hotkey.events.get_nowait()
                self.request_recognition()
        except queue.Empty:
            pass

        try:
            while True:
                event = self.result_queue.get_nowait()
                if event[0] == "success":
                    self._apply_success(*event[1:])
                else:
                    self._apply_error(event[1])
        except queue.Empty:
            pass
        if not self.closed:
            self.after(50, self._poll_events)

    def _apply_success(self, image, ranking, summary, elapsed_ms, captured_at) -> None:
        self.recognition_busy = False
        preview = image.copy()
        preview.thumbnail((760, 650), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

        if summary["accepted"]:
            label = layout_display_name(summary["result"])
            self.result_var.set(label)
            self.result_banner.configure(bg="#15803d")
            self.detail_var.set(
                f"识别成功 · 置信度 {summary['confidence']:.1%} · "
                f"前两名差值 {summary['margin']:.1%} · 已探索 {summary['explored_ratio']:.1%}"
            )
        else:
            best_label = layout_display_name(ranking[0][0])
            self.result_var.set("暂无法判断")
            self.result_banner.configure(bg="#b45309")
            self.detail_var.set(
                f"当前最高：{best_label} {summary['confidence']:.1%} · "
                f"前两名差值 {summary['margin']:.1%} · 已探索 {summary['explored_ratio']:.1%}。"
                "请继续探索后再按 F9。"
            )

        for index, (row, (label, probability)) in enumerate(zip(self.ranking_rows, ranking)):
            row["label"].set(f"{layout_display_name(label)}  ({label})")
            row["probability"].set(f"{probability:.1%}")
            canvas = row["bar"]
            canvas.update_idletasks()
            width = max(1, canvas.winfo_width())
            canvas.delete("all")
            canvas.create_rectangle(
                0,
                0,
                width * probability,
                15,
                fill=self.PALETTE[index % len(self.PALETTE)],
                outline="",
            )

        self.status_var.set(
            f"{captured_at:%H:%M:%S} 完成 · 推理 {elapsed_ms:.0f} ms · "
            f"设备 {self.device} · 按 F9 再次识别"
        )

    def _apply_error(self, exc: Exception) -> None:
        self.recognition_busy = False
        self.result_var.set("识别失败")
        self.result_banner.configure(bg="#b91c1c")
        self.detail_var.set(str(exc))
        self.status_var.set(f"失败：{exc}")

    def _show_hotkey_status(self) -> None:
        if self.hotkey.ready:
            self.hotkey_status_var.set("全局 F9：已启用")
        elif self.hotkey.error:
            self.hotkey_status_var.set(self.hotkey.error)

    def close(self) -> None:
        self.closed = True
        self.hotkey.stop()
        self.destroy()


def run_demo(config_path: Path = DEFAULT_CONFIG_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    enable_physical_pixel_coordinates()
    try:
        app = LayoutRecognitionDemo(config_path, model_path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("布局识别 Demo 启动失败", str(exc), parent=root)
        root.destroy()
        raise
    app.mainloop()
