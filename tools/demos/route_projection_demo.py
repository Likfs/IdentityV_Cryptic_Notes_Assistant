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

import cv2
import mss
import numpy as np
from mss.exception import ScreenShotError
from PIL import Image, ImageTk

from src.capture.region import CaptureRegion, enable_physical_pixel_coordinates
from tools.common.capture_config import DEFAULT_CONFIG_PATH as DEFAULT_CAPTURE_CONFIG, load_capture_config
from src.layout_classifier import explored_ratio, load_classifier, predict_image, summarize_prediction
from tools.demos.layout_recognition_demo import DEFAULT_MODEL_PATH, model_coverage_text
from tools.common.tk_overlay import RouteOverlayWindow
from tools.demos.projection_service import DemoProjectionService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_CONFIG = DEFAULT_CAPTURE_CONFIG
DEMO_CONFIG = PROJECT_ROOT / 'workspace/tool_state/route_projection_demo.json'
CLEAN_ROOT = PROJECT_ROOT / "assets/maps/a_bringer_of_doom/nightmare/clean"
CALIBRATION_ROOT = PROJECT_ROOT / "assets/maps/a_bringer_of_doom/nightmare/routes"
TEMPLATE_CACHE = PROJECT_ROOT / 'workspace/cache/route_projection_templates'
RUN_ROOT = PROJECT_ROOT / 'workspace/reports/route_projection/demo_runs'
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
HOTKEY_CHOICES = tuple(f"F{number}" for number in range(1, 13))
FLOOR_LABELS = {"2楼": "2nd", "1楼": "1st", "地下室": "b1"}
AUTO_LAYOUT = "自动（最近识别）"


def _default_demo_config() -> dict:
    return json.loads((PROJECT_ROOT / 'tools/configs/route_projection_demo.json').read_text(encoding='utf-8'))


def load_demo_config(path: Path = DEMO_CONFIG) -> dict:
    config = _default_demo_config()
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict):
            config.update(loaded)
    if config["recognize_hotkey"] not in HOTKEY_CHOICES:
        config["recognize_hotkey"] = "F9"
    if config["project_hotkey"] not in HOTKEY_CHOICES:
        config["project_hotkey"] = "F10"
    if config["recognize_hotkey"] == config["project_hotkey"]:
        config["project_hotkey"] = "F10" if config["recognize_hotkey"] != "F10" else "F11"
    if config["floor"] not in FLOOR_LABELS.values():
        config["floor"] = "2nd"
    config["opacity"] = float(np.clip(float(config["opacity"]), 0.10, 1.0))
    config["opencv_threads"] = max(1, int(config["opencv_threads"]))
    if config.get("registration_backend", "auto") not in ("cpu", "auto"):
        raise ValueError("Demo 配准后端必须为 cpu 或 auto")
    config.setdefault("registration_backend", "auto")
    return config


def save_demo_config(config: dict, path: Path = DEMO_CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture_region(region: CaptureRegion) -> Image.Image:
    try:
        with mss.MSS() as screen:
            shot = screen.grab(region.as_mss_monitor())
    except ScreenShotError as exc:
        raise RuntimeError("Windows 截图失败，请确认屏幕未锁定且游戏地图已打开。") from exc
    image = Image.frombytes("RGB", shot.size, shot.rgb)
    if image.size != (region.width, region.height):
        raise RuntimeError(f"截图尺寸异常：{image.size}，预期 {(region.width, region.height)}")
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def layout_display_name(label: str) -> str:
    match = re.fullmatch(r"layout_(\d+)", label)
    return f"布局 {int(match.group(1)):02d}" if match else label


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"无法编码诊断图片：{path}")
    encoded.tofile(str(path))


def available_asset_layouts() -> list[str]:
    return sorted(
        path.name
        for path in (CALIBRATION_ROOT / "route_lhp_solo").glob("layout_*")
        if (path / "calibration.json").is_file()
    )


def available_asset_floors() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    available: list[tuple[str, str]] = []
    review: list[tuple[str, str]] = []
    for layout in available_asset_layouts():
        calibration = CALIBRATION_ROOT / "route_lhp_solo" / layout / "calibration.json"
        floors = json.loads(calibration.read_text(encoding="utf-8"))["floors"]
        for floor in FLOOR_LABELS.values():
            target = available if floors[floor]["status"] == "PASS" and floors[floor].get("asset") else review
            target.append((layout, floor))
    return available, review


class GlobalActionHotkeys:
    IDS = {"recognize": 0x6401, "project": 0x6402}

    def __init__(self, recognize_key: str, project_key: str) -> None:
        self.keys = {"recognize": recognize_key, "project": project_key}
        self.events: queue.Queue[str] = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.thread_id: Optional[int] = None
        self.ready = False
        self.error = ""
        self._stop = threading.Event()

    @staticmethod
    def virtual_key(label: str) -> int:
        return 0x70 + int(label[1:]) - 1

    def start(self) -> None:
        if sys.platform != "win32":
            self.error = "全局快捷键仅在 Windows 可用；窗口激活时仍可使用所选按键。"
            return
        self.thread = threading.Thread(target=self._run, name="route-demo-hotkeys", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self.thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        registered = []
        try:
            for action, hotkey_id in self.IDS.items():
                key = self.keys[action]
                if not user32.RegisterHotKey(None, hotkey_id, MOD_NOREPEAT, self.virtual_key(key)):
                    raise RuntimeError(f"{key} 注册失败，可能被其他程序占用。")
                registered.append(hotkey_id)
            self.ready = True
            message = wintypes.MSG()
            while not self._stop.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                if message.message == WM_HOTKEY:
                    for action, hotkey_id in self.IDS.items():
                        if int(message.wParam) == hotkey_id:
                            self.events.put(action)
                            break
        except Exception as exc:
            self.error = str(exc)
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)
            self.ready = False
            self.thread_id = None


class RouteProjectionDemo(tk.Tk):
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        capture_config_path: Path = CAPTURE_CONFIG,
        demo_config_path: Path = DEMO_CONFIG,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path.resolve()
        self.capture_config_path = capture_config_path.resolve()
        self.demo_config_path = demo_config_path.resolve()
        self.demo_config = load_demo_config(self.demo_config_path)
        capture_config = load_capture_config(self.capture_config_path)
        self.region = CaptureRegion.from_value(capture_config["capture_roi"])
        cv2.setNumThreads(self.demo_config["opencv_threads"])
        self.model, self.checkpoint, self.device = load_classifier(self.model_path)
        self.labels = [str(label) for label in self.checkpoint["labels"]]
        self.asset_layouts = available_asset_layouts()
        self.available_floors, self.review_floors = available_asset_floors()
        self.projection_service = DemoProjectionService(backend or self.demo_config["registration_backend"], TEMPLATE_CACHE)
        self.result_queue: queue.Queue[tuple] = queue.Queue()
        self.operation_busy = False
        self.closed = False
        self.last_layout: str | None = None
        self.last_recognition: dict | None = None
        self.photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.hotkeys: GlobalActionHotkeys | None = None

        floor_label = next(label for label, value in FLOOR_LABELS.items() if value == self.demo_config["floor"])
        self.recognize_hotkey_var = tk.StringVar(value=self.demo_config["recognize_hotkey"])
        self.project_hotkey_var = tk.StringVar(value=self.demo_config["project_hotkey"])
        self.floor_var = tk.StringVar(value=floor_label)
        self.layout_mode_var = tk.StringVar(value=AUTO_LAYOUT)
        self.opacity_var = tk.DoubleVar(value=self.demo_config["opacity"])
        self.main_result_var = tk.StringVar(value="等待操作")
        self.recognition_var = tk.StringVar(value="尚未识别布局")
        self.registration_var = tk.StringVar(value="尚未执行配准")
        self.overlay_var = tk.StringVar(value="覆盖层：隐藏")
        self.hotkey_status_var = tk.StringVar(value="快捷键注册中……")
        self.status_var = tk.StringVar(value="打开游戏地图后先识别布局，再执行路线投影。")
        self.asset_var = tk.StringVar(
            value=f"路线资产：{len(self.asset_layouts)} 个布局 / {len(self.available_floors)} 层可用 / "
            f"{len(self.review_floors)} 层待复核"
        )
        self.roi_var = tk.StringVar(
            value=f"截图 ROI：左 {self.region.left}，上 {self.region.top}，{self.region.width}×{self.region.height}"
        )
        self.model_var = tk.StringVar(
            value=f"模型：{self.model_path.name} · {len(self.labels)} 类 · 设备 {self.device}"
        )
        self.model_coverage_var = tk.StringVar(value=model_coverage_text(self.checkpoint))
        self.metric_vars = {
            "structure": tk.StringVar(value="—"),
            "semantic": tk.StringVar(value="—"),
            "route": tk.StringVar(value="—"),
            "edge": tk.StringVar(value="—"),
            "scale": tk.StringVar(value="—"),
            "bbox": tk.StringVar(value="—"),
            "timing": tk.StringVar(value="—"),
            "paths": tk.StringVar(value="—"),
        }

        self.title("布局识别与路线投影 Demo")
        self.geometry("1420x900+25+25")
        self.minsize(1180, 760)
        self._build_ui()
        self.overlay = RouteOverlayWindow(self)
        self._install_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self._poll_events)
        self.after(700, self._refresh_hotkey_status)
        self._log("Demo 已启动。投影前请选择当前楼层。")

    def _build_ui(self) -> None:
        self.configure(background="#eef2f7")
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="布局识别与路线投影", font=("Microsoft YaHei UI", 19, "bold")).pack(side="left")
        ttk.Button(header, text="隐藏投影", command=self.hide_overlay).pack(side="right")
        ttk.Button(header, text="执行投影", command=self.request_projection).pack(side="right", padx=(0, 8))
        ttk.Button(header, text="识别布局", command=self.request_recognition).pack(side="right", padx=(0, 8))

        controls = ttk.LabelFrame(root, text="操作设置", padding=9)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="识别快捷键").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.recognize_hotkey_var, values=HOTKEY_CHOICES, width=6, state="readonly"
        ).grid(row=0, column=1, padx=(5, 14))
        ttk.Label(controls, text="投影快捷键").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.project_hotkey_var, values=HOTKEY_CHOICES, width=6, state="readonly"
        ).grid(row=0, column=3, padx=(5, 14))
        ttk.Button(controls, text="应用快捷键", command=self.apply_hotkeys).grid(row=0, column=4, padx=(0, 20))
        ttk.Label(controls, text="投影布局").grid(row=0, column=5, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.layout_mode_var,
            values=[AUTO_LAYOUT, *self.asset_layouts],
            width=22,
            state="readonly",
        ).grid(row=0, column=6, padx=(5, 14))
        ttk.Label(controls, text="楼层").grid(row=0, column=7, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.floor_var, values=list(FLOOR_LABELS), width=8, state="readonly"
        ).grid(row=0, column=8, padx=(5, 14))
        ttk.Label(controls, text="透明度").grid(row=0, column=9, sticky="w")
        ttk.Scale(controls, from_=0.10, to=1.0, variable=self.opacity_var, orient="horizontal", length=130).grid(
            row=0, column=10, padx=(5, 8)
        )
        ttk.Button(controls, text="选择模型…", command=self.select_model).grid(row=0, column=11)

        info = ttk.Frame(root)
        info.pack(fill="x", pady=(0, 8))
        self.result_banner = tk.Label(
            info,
            textvariable=self.main_result_var,
            bg="#334155",
            fg="white",
            font=("Microsoft YaHei UI", 16, "bold"),
            padx=12,
            pady=9,
        )
        self.result_banner.pack(side="left", fill="y")
        facts = ttk.Frame(info)
        facts.pack(side="left", fill="both", expand=True, padx=(12, 0))
        ttk.Label(facts, textvariable=self.recognition_var).pack(anchor="w")
        ttk.Label(facts, textvariable=self.registration_var).pack(anchor="w")
        ttk.Label(facts, textvariable=self.overlay_var).pack(anchor="w")
        ttk.Label(facts, textvariable=self.hotkey_status_var).pack(anchor="w")

        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True)
        preview_frame = ttk.LabelFrame(panes, text="图像诊断", padding=7)
        diagnose_frame = ttk.LabelFrame(panes, text="状态与排查信息", padding=9)
        panes.add(preview_frame, weight=3)
        panes.add(diagnose_frame, weight=2)

        self.preview_tabs = ttk.Notebook(preview_frame)
        self.preview_tabs.pack(fill="both", expand=True)
        self.preview_labels = {}
        for key, title in (("capture", "本次截图"), ("alignment", "配准叠图"), ("projection", "投影结果")):
            frame = ttk.Frame(self.preview_tabs)
            label = tk.Label(frame, text="暂无图像", bg="#17202b", fg="#cbd5e1", anchor="center")
            label.pack(fill="both", expand=True)
            self.preview_tabs.add(frame, text=title)
            self.preview_labels[key] = label

        ttk.Label(diagnose_frame, text="识别前五名", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.ranking = ttk.Treeview(diagnose_frame, columns=("layout", "probability"), show="headings", height=5)
        self.ranking.heading("layout", text="布局")
        self.ranking.heading("probability", text="概率")
        self.ranking.column("layout", width=190)
        self.ranking.column("probability", width=90, anchor="e")
        self.ranking.pack(fill="x", pady=(5, 10))

        metrics = ttk.LabelFrame(diagnose_frame, text="配准质量", padding=7)
        metrics.pack(fill="x")
        rows = (
            ("结构精度", "structure"),
            ("语义精度", "semantic"),
            ("路线与截图", "route"),
            ("边缘误差", "edge"),
            ("匹配尺度", "scale"),
            ("可见区域", "bbox"),
            ("耗时", "timing"),
            ("资产与诊断", "paths"),
        )
        for row, (label, key) in enumerate(rows):
            ttk.Label(metrics, text=f"{label}：").grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(metrics, textvariable=self.metric_vars[key], wraplength=390, justify="left").grid(
                row=row, column=1, sticky="w", pady=2
            )

        ttk.Label(diagnose_frame, text="运行日志", font=("Microsoft YaHei UI", 11, "bold")).pack(
            anchor="w", pady=(10, 4)
        )
        self.log_text = tk.Text(diagnose_frame, height=9, wrap="word", state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(footer, textvariable=self.model_var).pack(anchor="w")
        ttk.Label(footer, textvariable=self.model_coverage_var, wraplength=1370, justify="left").pack(anchor="w")
        ttk.Label(footer, textvariable=self.asset_var).pack(anchor="w")
        ttk.Label(footer, textvariable=self.roi_var).pack(anchor="w")

    def _install_hotkeys(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        recognize = self.recognize_hotkey_var.get()
        project = self.project_hotkey_var.get()
        self.hotkeys = GlobalActionHotkeys(recognize, project)
        self.hotkeys.start()
        if sys.platform != "win32":
            self.bind_all(f"<{recognize}>", lambda _event: self.request_recognition())
            self.bind_all(f"<{project}>", lambda _event: self.request_projection())

    def apply_hotkeys(self) -> None:
        recognize = self.recognize_hotkey_var.get()
        project = self.project_hotkey_var.get()
        if recognize == project:
            messagebox.showerror("快捷键冲突", "识别和投影不能使用同一个快捷键。", parent=self)
            return
        self.demo_config.update(
            {
                "recognize_hotkey": recognize,
                "project_hotkey": project,
                "floor": FLOOR_LABELS[self.floor_var.get()],
                "opacity": float(self.opacity_var.get()),
            }
        )
        save_demo_config(self.demo_config, self.demo_config_path)
        self._install_hotkeys()
        self.hotkey_status_var.set("快捷键正在重新注册……")
        self.after(500, self._refresh_hotkey_status)
        self._log(f"快捷键已更新：识别 {recognize}，投影 {project}")

    def _refresh_hotkey_status(self) -> None:
        if self.hotkeys is None:
            return
        if self.hotkeys.ready:
            self.hotkey_status_var.set(
                f"全局快捷键：识别 {self.hotkeys.keys['recognize']} / 投影 {self.hotkeys.keys['project']}"
            )
        elif self.hotkeys.error:
            self.hotkey_status_var.set(self.hotkeys.error)
        elif not self.closed:
            self.after(300, self._refresh_hotkey_status)

    def select_model(self) -> None:
        if self.operation_busy:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="选择布局分类模型",
            initialdir=str(self.model_path.parent),
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            model, checkpoint, device = load_classifier(Path(path).resolve())
        except Exception as exc:
            messagebox.showerror("模型载入失败", str(exc), parent=self)
            return
        self.model_path = Path(path).resolve()
        self.model, self.checkpoint, self.device = model, checkpoint, device
        self.labels = [str(label) for label in checkpoint["labels"]]
        self.last_layout = None
        self.last_recognition = None
        self.model_var.set(f"模型：{self.model_path.name} · {len(self.labels)} 类 · 设备 {self.device}")
        self.model_coverage_var.set(model_coverage_text(checkpoint))
        self._log(f"模型已切换：{self.model_path}")

    def _begin_operation(self, text: str) -> bool:
        if self.operation_busy or self.closed:
            self._log("已有操作正在运行，本次请求已忽略。")
            return False
        self.hide_overlay(log=False)
        self.operation_busy = True
        self.status_var.set(text)
        return True

    def request_recognition(self) -> None:
        if not self._begin_operation("正在隐藏旧投影、截图并识别布局……"):
            return
        threading.Thread(target=self._recognition_worker, name="route-demo-recognition", daemon=True).start()

    def _recognition_worker(self) -> None:
        try:
            time.sleep(0.06)
            started = time.perf_counter()
            image = capture_region(self.region)
            capture_ms = (time.perf_counter() - started) * 1000.0
            inference_started = time.perf_counter()
            ranking = predict_image(self.model, self.checkpoint, self.device, image)
            summary = summarize_prediction(ranking, self.checkpoint, explored_ratio(image))
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            self.result_queue.put(("recognition", image, ranking, summary, capture_ms, inference_ms))
        except Exception as exc:
            self.result_queue.put(("error", "布局识别", exc))

    def _selected_projection_layout(self) -> str | None:
        selected = self.layout_mode_var.get()
        return self.last_layout if selected == AUTO_LAYOUT else selected

    def request_projection(self) -> None:
        layout = self._selected_projection_layout()
        if not layout:
            self.status_var.set("尚无可用布局；请先执行布局识别，或在“投影布局”中手动选择。")
            self._log("投影被拒绝：没有已确认布局。")
            return
        floor = FLOOR_LABELS[self.floor_var.get()]
        calibration = CALIBRATION_ROOT / "route_lhp_solo" / layout / "calibration.json"
        if not calibration.is_file():
            self.status_var.set(f"{layout} 没有路线资产，无法投影。")
            self._log(f"投影被拒绝：缺少 {calibration}")
            return
        if (layout, floor) not in self.available_floors:
            self.status_var.set(f"{layout} / {floor} 的路线资产待复核，请先查看标定预览。")
            self._log(f"投影被拒绝：{layout} / {floor} 自动标定未通过。")
            return
        if not self._begin_operation(f"正在配准 {layout} / {floor} 并生成投影……"):
            return
        opacity = float(self.opacity_var.get())
        threading.Thread(
            target=self._projection_worker,
            args=(layout, floor, opacity),
            name="route-demo-projection",
            daemon=True,
        ).start()

    def _projection_worker(self, layout: str, floor: str, opacity: float) -> None:
        try:
            time.sleep(0.06)
            capture_started = time.perf_counter()
            image = capture_region(self.region)
            screenshot = pil_to_bgr(image)
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            run_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{layout}_{floor}"
            result, payload = self.projection_service.project(
                screenshot, layout, floor, opacity, RUN_ROOT / run_id,
                capture_ms=capture_ms, capture_roi=self.region.as_list(),
            )
            if result.registration is None:
                raise RuntimeError("；".join(result.reasons) or "未执行配准")
            self.result_queue.put(("projection", image, screenshot, result.registration,
                                   result.projection, result.layer, payload))
        except Exception as exc:
            self.result_queue.put(("error", "路线投影", exc))

    def _poll_events(self) -> None:
        if self.hotkeys is not None:
            try:
                while True:
                    action = self.hotkeys.events.get_nowait()
                    if action == "recognize":
                        self.request_recognition()
                    elif action == "project":
                        self.request_projection()
            except queue.Empty:
                pass
        try:
            while True:
                event = self.result_queue.get_nowait()
                if event[0] == "recognition":
                    self._apply_recognition(*event[1:])
                elif event[0] == "projection":
                    self._apply_projection(*event[1:])
                else:
                    self._apply_error(event[1], event[2])
        except queue.Empty:
            pass
        if not self.closed:
            self.after(50, self._poll_events)

    def _apply_recognition(self, image, ranking, summary, capture_ms: float, inference_ms: float) -> None:
        self.operation_busy = False
        self._set_preview("capture", pil_to_bgr(image))
        self._update_ranking(ranking)
        self.last_recognition = summary
        if summary["accepted"]:
            self.last_layout = summary["result"]
            self.main_result_var.set(layout_display_name(self.last_layout))
            self.result_banner.configure(bg="#15803d")
            asset_state = "有路线资产" if self.last_layout in self.asset_layouts else "尚无路线资产"
            self.recognition_var.set(
                f"识别通过：{self.last_layout} · 置信度 {summary['confidence']:.1%} · "
                f"间隔 {summary['margin']:.1%} · 探索 {summary['explored_ratio']:.1%} · {asset_state}"
            )
            self._log(f"识别通过：{self.last_layout}，置信度 {summary['confidence']:.1%}")
        else:
            # 识别失败后清除旧布局，避免把上一局的路线投影到当前地图。
            self.last_layout = None
            self.main_result_var.set("暂无法判断")
            self.result_banner.configure(bg="#b45309")
            best = ranking[0]
            self.recognition_var.set(
                f"识别拒绝：最高 {best[0]} {best[1]:.1%} · 间隔 {summary['margin']:.1%} · "
                f"探索 {summary['explored_ratio']:.1%}"
            )
            self._log("识别未达到阈值；请继续除雾后重试。")
        self.metric_vars["timing"].set(f"截图 {capture_ms:.0f} ms；布局推理 {inference_ms:.0f} ms")
        self.status_var.set(
            f"识别完成。{self.recognize_hotkey_var.get()} 重新识别；"
            f"{self.project_hotkey_var.get()} 投影当前楼层。"
        )

    def _apply_projection(self, image, screenshot, registration, projection, layer, payload) -> None:
        self.operation_busy = False
        self._set_preview("capture", screenshot)
        if registration.alignment_preview is not None:
            self._set_preview("alignment", registration.alignment_preview)
        validation = registration.validation
        match = registration.match.result if registration.match is not None else None
        self.metric_vars["structure"].set(
            f"{validation.get('structure_precision', 0.0):.1%}"
            if validation
            else "无"
        )
        self.metric_vars["semantic"].set(
            f"{validation.get('semantic_precision', 0.0):.1%}"
            if validation
            else "无"
        )
        overlay_validation = payload.get("overlay_validation")
        self.metric_vars["route"].set(
            f"覆盖 {overlay_validation['structure_coverage']:.1%}；未覆盖距离 P95 {overlay_validation['distance_p95_px']:.1f} px"
            if overlay_validation and "structure_coverage" in overlay_validation
            else "按正式流程显示；不做路线结构复核"
        )
        if validation and validation.get("edge_median_error_px") is not None:
            self.metric_vars["edge"].set(
                f"中位 {validation['edge_median_error_px']:.2f} px；P95 {validation['edge_p95_error_px']:.2f} px"
            )
        else:
            self.metric_vars["edge"].set("无")
        self.metric_vars["scale"].set(
            f"{match.best_scale:.4f}；位置 {match.best_position}; 置信间隔 {match.confidence:.4f}"
            if match is not None and match.best_scale is not None
            else "无"
        )
        self.metric_vars["bbox"].set(str(registration.observation_bbox or "无"))
        timings = registration.timings
        self.metric_vars["timing"].set(
            f"截图 {payload['capture_ms']:.0f} ms；分割 {timings.get('segmentation', 0):.0f} ms；"
            f"粗匹配 {timings.get('coarse_match', 0):.0f} ms；精匹配 {timings.get('fine_match', 0):.0f} ms；"
            f"投影 {payload.get('projection_ms') or 0:.0f} ms"
        )
        self.metric_vars["paths"].set(payload["diagnostic_directory"])

        if payload["accepted"] and projection is not None and layer is not None:
            self._set_preview("projection", projection)
            self.preview_tabs.select(2)
            self.overlay.show(layer, self.region.left, self.region.top)
            self.main_result_var.set("投影成功")
            self.result_banner.configure(bg="#15803d")
            self.registration_var.set(
                f"配准通过：{payload['layout']} / {payload['floor']} · "
                f"后端 {payload['backend_info'].get('effective_backend', 'unknown')} · "
                f"结构 {validation.get('structure_precision', 0):.1%}"
            )
            self.overlay_var.set(
                f"覆盖层：显示中 · ROI {self.region.left},{self.region.top} · 透明度 {payload['opacity']:.0%}"
            )
            self.status_var.set("路线已显示在游戏地图区域；覆盖层不会拦截鼠标。")
            self._log(f"投影成功，诊断已保存：{payload['diagnostic_directory']}")
        else:
            self.overlay.hide()
            reasons = "；".join(
                registration.rejection_reasons
                if not registration.accepted
                else (overlay_validation or {}).get("rejection_reasons", [])
            ) or "未知原因"
            self.main_result_var.set("投影被拒绝")
            self.result_banner.configure(bg="#b91c1c")
            self.registration_var.set(f"质量检查拒绝：{reasons}")
            self.overlay_var.set("覆盖层：隐藏（防止错误路线）")
            self.status_var.set("路线未通过质量检查，已禁止显示。请查看配准叠图、投影预览和诊断信息。")
            self.preview_tabs.select(2 if projection is not None else (1 if registration.alignment_preview is not None else 0))
            if projection is not None:
                self._set_preview("projection", projection)
            self._log(f"投影被拒绝：{reasons}；诊断：{payload['diagnostic_directory']}")

    def _apply_error(self, operation: str, exc: Exception) -> None:
        self.operation_busy = False
        self.overlay.hide()
        self.main_result_var.set(f"{operation}失败")
        self.result_banner.configure(bg="#b91c1c")
        self.status_var.set(str(exc))
        self.overlay_var.set("覆盖层：隐藏")
        self._log(f"{operation}异常：{exc}")

    def _set_preview(self, key: str, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((820, 690), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.photo_refs[key] = photo
        self.preview_labels[key].configure(image=photo, text="")

    def _update_ranking(self, ranking: list[tuple[str, float]]) -> None:
        for item in self.ranking.get_children():
            self.ranking.delete(item)
        for label, probability in ranking[:5]:
            self.ranking.insert("", "end", values=(f"{layout_display_name(label)} ({label})", f"{probability:.2%}"))

    def _log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def hide_overlay(self, *, log: bool = True) -> None:
        if hasattr(self, "overlay"):
            self.overlay.hide()
        self.overlay_var.set("覆盖层：隐藏")
        if log:
            self._log("覆盖层已隐藏。")

    def close(self) -> None:
        self.closed = True
        self.demo_config.update(
            {
                "recognize_hotkey": self.recognize_hotkey_var.get(),
                "project_hotkey": self.project_hotkey_var.get(),
                "floor": FLOOR_LABELS[self.floor_var.get()],
                "opacity": float(self.opacity_var.get()),
            }
        )
        save_demo_config(self.demo_config, self.demo_config_path)
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.overlay.destroy()
        self.destroy()


def run_demo(
    model_path: Path = DEFAULT_MODEL_PATH,
    capture_config_path: Path = CAPTURE_CONFIG,
    demo_config_path: Path = DEMO_CONFIG,
    backend: str | None = None,
) -> None:
    enable_physical_pixel_coordinates()
    try:
        app = RouteProjectionDemo(model_path, capture_config_path, demo_config_path, backend)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("路线投影 Demo 启动失败", str(exc), parent=root)
        root.destroy()
        raise
    app.mainloop()
