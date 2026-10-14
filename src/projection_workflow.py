from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from src.paths import DIAGNOSTIC_ROOT
from src.app_catalog import DifficultyProfile, MapCatalog, TEMPLATE_VERSION_NAMES
from src.layout_retrieval import create_layout_retriever, LayoutRetriever, LayoutRetrievalResult
from src.projection.route_overlay import composite_route_layer
from src.registration.result import OriginalRegistrationResult

RUN_ROOT = DIAGNOSTIC_ROOT / 'route_projection/app_runs'

@dataclass
class WorkflowResult:
    screenshot: np.ndarray
    decision: dict | None
    selected_layout: str | None
    variant_id: str | None
    variant_source: str | None
    registration: OriginalRegistrationResult | None
    overlay_validation: dict | None
    layer: np.ndarray | None
    projection: np.ndarray | None
    accepted: bool
    reasons: list[str]
    timings_ms: dict[str, float]
    profile_key: str
    floor: str
    template_path: str | None
    asset_path: str | None
    calibration_path: str | None
    attempts: list[dict] = field(default_factory=list)
    run_directory: str | None = None
    session_id: str | None = None
    projection_preview_pending: bool = False
    preview_error: str | None = None
    layout_recognition_backend: str = 'icon_structure'
    layout_retrieval: LayoutRetrievalResult | None = None
    recognition_window: list[int] | None = None
    gui_diagnostics: dict = field(default_factory=dict)
    capture_window_id: int | None = None

    def diagnostics(self) -> dict:
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile_key,
            "session_id": self.session_id,
            "projection_preview_pending": self.projection_preview_pending,
            "preview_error": self.preview_error,
            "floor": self.floor,
            "floor_detection": None,  # Historical diagnostics key; floor identification is not performed.
            "decision": self.decision,
            "layout_recognition_backend": self.layout_recognition_backend,
            "layout_retrieval": self.layout_retrieval.diagnostics() if self.layout_retrieval else None,
            "recognition_window": self.recognition_window,
            "selected_layout": self.selected_layout,
            "variant_id": self.variant_id,
            "variant_source": self.variant_source,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "timings_ms": self.timings_ms,
            "gui_diagnostics": self.gui_diagnostics,
            "template": self.template_path,
            "calibration": self.calibration_path,
            "asset": self.asset_path,
            "registration": self.registration.diagnostics() if self.registration else None,
            "overlay_validation": self.overlay_validation,
            "attempts": self.attempts,
            "run_directory": self.run_directory,
        }


def complete_projection_preview(result: WorkflowResult) -> WorkflowResult:
    """Complete a preview on a new result; never mutate a displayed result's arrays or timings."""
    if not result.projection_preview_pending:
        return replace(result, timings_ms=dict(result.timings_ms))
    timings = dict(result.timings_ms)
    started = time.perf_counter()
    try:
        projection, _ = composite_route_layer(result.screenshot, result.layer)
        error = None
    except Exception as exc:
        projection, error = None, str(exc)
    timings['projection_composite'] = (time.perf_counter()-started)*1000
    timings['projection_compute_total'] = timings.get('projection_layer', 0.)+timings['projection_composite']
    return replace(result, projection=projection, timings_ms=timings,
                   projection_preview_pending=False, preview_error=error)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"无法写入诊断图片：{path}")
    encoded.tofile(str(path))


def save_workflow_result(result: WorkflowResult, directory: Path | None = None) -> Path:
    if directory is None:
        layout = result.selected_layout or "unknown"
        directory = RUN_ROOT / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{layout}_{result.floor}"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result.run_directory = str(directory)
    _write_image(directory / "01_screenshot.png", result.screenshot)
    if result.projection is not None:
        _write_image(directory / "05_projection.png", result.projection)
    if result.layer is not None:
        _write_image(directory / "06_overlay_layer.png", result.layer)
    (directory / "result.json").write_text(
        json.dumps(result.diagnostics(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def process_screenshot(
    screenshot_bgr: np.ndarray,
    *,
    catalog: MapCatalog,
    profile: DifficultyProfile,
    opacity: float,
    recognize_only: bool = False,
    layout_override: str | None = None,
    registration_backend: str = 'template',
    defer_projection_preview: bool = False,
    layout_recognition_backend: str = 'icon_structure',
    layout_retriever: LayoutRetriever | None = None,
    recognition_reference_size: tuple[int, int] = (1180, 1050),
    template_variant_id: str = 'route_zhanshi_fast',
    recognition_cancelled=None,
) -> WorkflowResult:
    started = time.perf_counter()
    if screenshot_bgr.ndim != 3 or screenshot_bgr.shape[2] != 3:
        raise ValueError("截图必须为 BGR 三通道图片")
    timings: dict[str, float] = {}
    decision = None
    retrieval = None
    if layout_recognition_backend not in ('icon_structure', 'pure_structure'):
        raise ValueError('未知的布局识别方法')
    if registration_backend != 'template':
        raise ValueError('仅支持定尺度模板匹配')
    if recognize_only or layout_override is None:
        if not recognize_only:
            raise ValueError('请先识别并确认布局，再进行配准投影')
        retriever = layout_retriever or create_layout_retriever(layout_recognition_backend, profile.clean_root)
        try:
            retrieval = retriever.recognize(screenshot_bgr, recognition_reference_size,
                **({'cancelled': recognition_cancelled} if recognition_cancelled is not None else {}))
        finally:
            if layout_retriever is None and hasattr(retriever, 'close'):
                retriever.close()
        decision = retrieval.decision()
        timings['layout_retrieval'] = retrieval.timings_ms['total']
        timings['classification'] = timings['layout_retrieval']  # Historical diagnostics alias.
    selected_layout = layout_override or (decision["layout"] if decision else None)
    result = WorkflowResult(
        screenshot=screenshot_bgr,
        decision=decision,
        selected_layout=selected_layout,
        variant_id=None,
        variant_source=None,
        registration=None,
        overlay_validation=None,
        layer=None,
        projection=None,
        accepted=False,
        reasons=[],
        timings_ms=timings,
        profile_key=profile.key,
        floor='all',
        template_path=None,
        asset_path=None,
        calibration_path=None,
        layout_recognition_backend=layout_recognition_backend,
        layout_retrieval=retrieval,
    )
    if recognize_only:
        # Recognition succeeds when candidates are available for manual review.
        # Candidates are never automatically locked; the UI requires manual confirmation.
        result.selected_layout = decision["best_label"] if decision else None
        result.accepted = bool(result.selected_layout)
        if not result.accepted:
            result.reasons.extend(retrieval.reasons if retrieval else ["未返回可供选择的布局候选"])
        result.timings_ms["total"] = (time.perf_counter() - started) * 1000.0
        return result
    from src.registration.template_matching.direct import project_original
    result.floor = 'all'
    result.variant_id = template_variant_id
    result.variant_source = TEMPLATE_VERSION_NAMES.get(template_variant_id, template_variant_id) + ' · 三层完整原图'
    if not selected_layout:
        result.reasons = ['请先识别并确认布局，再进行 定尺度模板匹配 原图配准']
    else:
        registration, source, layer, composite, direct_timings = project_original(
            screenshot_bgr, profile.key, selected_layout, opacity, variant_id=template_variant_id,
            defer_preview=defer_projection_preview)
        result.registration = registration
        result.layer, result.projection = layer, composite
        result.asset_path = str(source) if source is not None else None
        result.accepted = registration.accepted
        result.projection_preview_pending = bool(defer_projection_preview and registration.accepted)
        result.reasons = list(registration.rejection_reasons)
        result.overlay_validation = {'accepted':result.accepted, 'enabled':False,
                                     'rejection_reasons':list(result.reasons),
                                     'scope':'original_source_image'}
        result.timings_ms.update(direct_timings)
        result.attempts = [{'layout':selected_layout, 'accepted':result.accepted,
                            'reasons':list(result.reasons)}]
    result.timings_ms['total'] = (time.perf_counter()-started)*1000
    return result
