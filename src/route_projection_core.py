from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from ctypes import wintypes

import cv2
import numpy as np

from src.app_catalog import (
    DEFAULT_SETTINGS_PATH,
    FLOOR_NAMES,
    FLOORS,
    HOTKEYS,
    MAP_KEYS,
    MapCatalog,
    key_to_vk,
    load_settings,
    save_settings,
)
from src.capture.region import CaptureRegion, enable_physical_pixel_coordinates
from src.capture.runtime import GlobalActionHotkeys, capture_region, pil_to_bgr
from src.floor_indicator import (
    FLOOR_INDICATOR_MARGIN, FLOOR_INDICATOR_ROI, FLOOR_INDICATOR_SEARCH_ROI,
    FloorDetection, classify_floor_indicator,
)
from src.layout_classifier import load_classifier
from src.projection_workflow import WorkflowResult, process_screenshot, save_workflow_result
from src.registration.runtime_registration import MAX_EDGE_MEDIAN_ERROR_PX
from src.registration.temporal_evidence import TemporalEvidence


ROOT = Path(__file__).resolve().parents[1]
AUTO = "自动判断"


def capture_bgr(roi: list[int]) -> np.ndarray:
    return pil_to_bgr(capture_region(CaptureRegion.from_value(roi)))


def floor_control_mouse_down(titles: list[str], reference_size: list[int]) -> bool:
    """Observe a click in the floor control, without capturing or classifying it."""
    if sys.platform != 'win32' or not (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000):
        return False
    try:
        client = foreground_game_client(titles)
        region = game_capture_region(client, list(FLOOR_INDICATOR_SEARCH_ROI), reference_size)
        point = wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return False
        return region.left <= point.x < region.left+region.width and region.top <= point.y < region.top+region.height
    except (OSError, ValueError, RuntimeError):
        return False


def game_capture_region(client: CaptureRegion, roi: list[int], reference_size: list[int]) -> CaptureRegion:
    """Map a client-local reference ROI to physical screen pixels."""
    reference_width, reference_height = reference_size
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("游戏参考分辨率必须大于 0")
    local = CaptureRegion.from_value(roi)
    if local.left < 0 or local.top < 0 or local.left + local.width > reference_width or local.top + local.height > reference_height:
        raise ValueError("地图区域必须完全位于游戏参考画面内")
    x0 = round(local.left * client.width / reference_width)
    y0 = round(local.top * client.height / reference_height)
    x1 = round((local.left + local.width) * client.width / reference_width)
    y1 = round((local.top + local.height) * client.height / reference_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("游戏窗口太小，地图截图区域无效")
    return CaptureRegion(client.left + x0, client.top + y0, x1 - x0, y1 - y0)


def foreground_game_client(titles: list[str]) -> CaptureRegion:
    """Locate the visible game client area; never substitute a desktop ROI."""
    if sys.platform != "win32":
        raise RuntimeError("仅支持在 Windows 游戏窗口中截图")
    if not titles or not game_window_is_foreground(titles):
        raise RuntimeError("游戏窗口不在前台，请切回游戏后按 F8 或地图键")
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if user32.IsIconic(hwnd):
        raise RuntimeError("游戏窗口已最小化，无法截取地图区域")
    user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.ClientToScreen.restype = wintypes.BOOL
    client_rect = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError()
    width = int(client_rect.right - client_rect.left)
    height = int(client_rect.bottom - client_rect.top)
    if width <= 0 or height <= 0:
        raise RuntimeError("游戏窗口已最小化或画面区域无效")
    return CaptureRegion(int(origin.x), int(origin.y), width, height)


def ensure_region_on_screen(region: CaptureRegion) -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    left = user32.GetSystemMetrics(76)
    top = user32.GetSystemMetrics(77)
    right = left + user32.GetSystemMetrics(78)
    bottom = top + user32.GetSystemMetrics(79)
    if (region.left < left or region.top < top or
            region.left + region.width > right or region.top + region.height > bottom):
        raise RuntimeError("地图截图区域超出可见屏幕，请移动或调整游戏窗口")


def capture_floor_indicator(client: CaptureRegion, reference_size: list[int], *, timings=None) -> tuple[FloorDetection, np.ndarray, list[int]]:
    region_started = time.perf_counter()
    x, y, width, height = FLOOR_INDICATOR_ROI
    search_x, search_y, search_width, search_height = FLOOR_INDICATOR_SEARCH_ROI
    margin = FLOOR_INDICATOR_MARGIN
    core = game_capture_region(client, [x, y, width, height], reference_size)
    search = game_capture_region(
        client,
        [search_x - margin, search_y - margin, search_width + 2 * margin, search_height + 2 * margin],
        reference_size,
    )
    ensure_region_on_screen(search)
    capture_started = time.perf_counter()
    if timings is not None:
        timings['floor_region_ms'] = (capture_started - region_started) * 1000
    image = capture_bgr(search.as_list())
    classify_started = time.perf_counter()
    if timings is not None:
        timings['floor_capture_ms'] = (classify_started - capture_started) * 1000
    detection = classify_floor_indicator(image, (core.width, core.height))
    if timings is not None:
        timings['floor_classify_ms'] = (time.perf_counter() - classify_started) * 1000
    return detection, image, search.as_list()


def map_key_down(key: str) -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(key_to_vk(key)) & 0x8000)


def escape_key_down() -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)


def own_window_is_foreground() -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    hwnd = user32.GetForegroundWindow()
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return process_id.value == os.getpid()


def foreground_window_title() -> str:
    if sys.platform != "win32":
        return ""
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowTextW.restype = ctypes.c_int
    buffer = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def game_window_is_foreground(titles: list[str]) -> bool:
    if not titles:
        return False
    current = foreground_window_title().casefold()
    return any(title.casefold() in current for title in titles)


def opacity_to_percent(opacity: float) -> int:
    return round(min(1.0, max(0.0, opacity)) * 100)


def percent_to_opacity(percent: float) -> float:
    return round(min(100.0, max(0.0, percent)) / 100.0, 4)


@dataclass
class MapToggleState:
    is_open: bool = False
    key_was_down: bool = False
    escape_was_down: bool = False

    def observe(self, key_down: bool, *, enabled: bool = True) -> str | None:
        pressed_now = key_down and not self.key_was_down
        self.key_was_down = key_down
        if not enabled or not pressed_now:
            return None
        self.is_open = not self.is_open
        return "opened" if self.is_open else "closed"

    def reset(self, key_down: bool = False) -> None:
        self.is_open = False
        self.key_was_down = key_down
        self.escape_was_down = False

    def observe_escape(self, key_down: bool, *, enabled: bool = True) -> bool:
        pressed_now = key_down and not self.escape_was_down
        self.escape_was_down = key_down
        if not enabled or not pressed_now or not self.is_open:
            return False
        self.is_open = False
        return True


def format_projection_metrics(result: WorkflowResult) -> str:
    validation = result.registration.validation if result.registration else {}
    method = getattr(result.registration, "timings", {}).get("method") if result.registration else None
    method_name = {"fixed-scale-template": "定尺度模板匹配候选", "component_tolerance": "component_tolerance 几何配准",
                   "temporal_reuse": "历史变换复核", "temporal_search": "时序快速搜索",
                   "soft_pose_reuse": "暂存姿态复核",
                   "temporal_local_zoom": "局部缩放搜索", "temporal_recenter_zoom": "地图重新居中搜索",
                   "temporal_insufficient_evidence": "证据不足", "insufficient_evidence": "证据不足",
                   "local_structure_index": "局部结构索引",
                   "global_search": "全局搜索", "global_sparse_unique": "稀疏结构唯一匹配",
                   "global_candidate_review": "候选结构复核"}.get(method, "尚未配准")
    overlay = result.overlay_validation or {}
    timing = result.timings_ms
    if validation.get('coordinate_space') == 'original_source_image':
        template = validation.get('template', {})
        preview = ('后台处理中' if getattr(result, 'projection_preview_pending', False)
                   else f"{timing.get('projection_composite', 0):.0f} ms")
        submit = f"{timing['overlay_submit']:.1f} ms" if 'overlay_submit' in timing else '未提交'
        delivery = (f" · 回传 {timing['result_delivery']:.1f} ms · 覆盖提交 {submit}"
                    if 'result_delivery' in timing else '')
        end_to_end = (f" · 按键检测至覆盖提交 {timing['key_to_overlay_submit']:.0f} ms"
                      if 'key_to_overlay_submit' in timing else '')
        return (f"完整原图配准：{'候选已生成' if result.accepted else '未生成候选'} · 定尺度模板匹配不分楼层\n"
                f"尺度 {template.get('scale', 0):.4f} · 原图位置 {template.get('location', '—')} · 请核对投影位置\n"
                f"耗时：截图 {timing.get('capture', 0):.0f} ms · 楼层识别已跳过 · "
                f"配准 {timing.get('registration', 0):.0f} ms · 投影图层 {timing.get('projection_layer', timing.get('projection', 0)):.0f} ms · "
                f"预览合成 {preview}{delivery}{end_to_end}")
    structure = validation.get("structure_precision")
    semantic = validation.get("semantic_precision")
    edge = validation.get("edge_median_error_px")
    component_search = validation.get('registration_algorithm') == 'component_tolerance'
    if component_search:
        raster = validation.get('raster', {})
        structure = raster.get('structure_precision')
        semantic = raster.get('semantic_precision')
    coverage = overlay.get("structure_coverage")
    disagreement = overlay.get("route_pose_disagreement") or {}
    disagreement_p95 = disagreement.get("p95_px")
    soft_warning = " · 缩放仍待远处结构确认" if validation.get("soft_pose_reuse") else ""
    route_corroborated = bool(validation.get("route_corroborated"))
    registration_state = ("路线复核通过（完整显示，缩放持续校准）" if getattr(result,'accepted',False) and validation.get('projection_scope')=='local' else
                          "路线复核通过" if route_corroborated else
                          "通过" if result.registration and result.registration.accepted else "未通过")
    structure_text = f"{structure:.1%}" if structure is not None else "未计算"
    semantic_text = f"{semantic:.1%}" if semantic is not None else "未计算"
    edge_text = ("未计算" if edge is None else
                 f"{edge:.1f}px（路线已复核）" if route_corroborated else
                 f"{edge:.1f}px（须 ≤ {MAX_EDGE_MEDIAN_ERROR_PX:g}px）")
    if validation.get('edge_median_screen_px') is not None:
        edge_text = f"{validation['edge_median_screen_px']:.1f} 屏幕px（可靠墙组）"
    elif edge is not None and result.registration is not None:
        matrix = getattr(result.registration,'screenshot_to_clean',None)
        if matrix is not None:
            factor = float(matrix[0,0])*float(result.registration.timings.get('template_registration_scale',1))
            if factor>0:
                edge_text = (f"{edge/factor:.1f} 屏幕px（路线已复核）" if route_corroborated else
                             f"{edge/factor:.1f} 屏幕px（须 ≤ {MAX_EDGE_MEDIAN_ERROR_PX/factor:.1f}）")
    coverage_text = (f"{coverage:.1%}" if coverage is not None else
                     "未计算（底图配准未通过）" if registration_state == "未通过" else
                     "未计算（尚未生成路线层）")
    if component_search:
        median = validation.get('component_acceptance', {}).get('checks', {}).get('trusted_edge_median', {}).get('value')
        edge_text = '未计算' if median is None else f'{median:.1f} 屏幕px（可信结构）'
        coverage_text = '已停用（按底图配准结果显示）'
    route_uncertainty = (f" · 全图候选偏移 P95 {disagreement_p95:.1f}px"
                         if disagreement_p95 is not None else "")
    if validation.get('registration_algorithm') == 'fixed_scale_template':
        registration_state = '候选已生成' if result.registration.accepted else '未生成候选'
        coverage_text = '已停用（模板匹配候选直接显示）'
        template = validation.get('template', {})
        soft_warning = f" · 尺度 {template.get('scale', 0):.4f} · 请核对投影位置"
    scale_sensitivity = (overlay.get("scale_profile_disagreement") or {}).get("p95_px")
    if scale_sensitivity is not None and scale_sensitivity > 10:
        route_uncertainty += f" · 缩放敏感度 {scale_sensitivity:.1f}px（非实测误差，请核对远处路线）"
    return (
        f"底图配准：{registration_state}（{method_name}）· 结构 {structure_text} · 语义 {semantic_text}\n"
        f"边缘中位误差：{edge_text} · 路线覆盖：{coverage_text}{route_uncertainty}{soft_warning}\n"
        f"耗时：截图 {timing.get('capture', 0):.0f} ms · 楼层 {timing.get('floor_detection', 0):.0f} ms · "
        f"识别 {timing.get('classification', 0):.0f} ms · "
        f"配准 {timing.get('registration', 0):.0f} ms · 投影 {timing.get('projection', 0):.0f} ms"
    )



