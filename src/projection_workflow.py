from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.paths import CACHE_ROOT, LOG_ROOT, DIAGNOSTIC_ROOT
from src.app_catalog import DifficultyProfile, MapCatalog, prediction_decision
from src.layout_classifier import predict_image
from src.layout_retrieval import LayoutRetriever, LayoutRetrievalResult
from src.projection.route_overlay import (
    RouteOverlayAsset,
    composite_route_layer,
    load_route_overlay_asset,
    measure_route_pose_disagreement,
    warp_route_overlay,
    warp_route_structure,
)
from src.registration.runtime_registration import (
    PreparedRegistrationTemplate,
    RuntimeRegistrationResult,
    SoftPoseHint,
    prepare_registration_template,
    validate_route_overlay,
)
from src.registration.temporal_evidence import TemporalEvidence
from src.registration.multiscale_structure.runtime import register_screenshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = DIAGNOSTIC_ROOT / 'route_projection/app_runs'


def validate_display_asset(registration, layer, asset, *, catalog, profile, asset_cache):
    """A clean reference cannot corroborate itself; retain independent route evidence."""
    algorithm = registration.validation.get('registration_algorithm')
    if algorithm in {'component_tolerance', 'multiscale_structure', 'fixed_scale_template'}:
        # Reviewed single-frame registration validates screenshot against clean map. Route annotations are
        # display content, and do not veto (or rescue) registration.
        pose = registration.clean_to_screen
        valid = bool(registration.accepted and pose is not None
                     and np.asarray(pose).shape in ([(2, 3), (3, 3)] if algorithm == 'fixed_scale_template' else [(2, 3)])
                     and np.isfinite(pose).all())
        return {'accepted': valid, 'rejection_reasons': [] if valid else ['配准未通过或姿态无效'],
                'enabled': False, 'reason': f'{algorithm}_registration_only',
                'display_variant': asset.variant}
    evidence_asset = asset
    if asset.asset_kind == "clean":
        variants = [v for v in catalog.variants(profile, asset.layout)
                    if v.asset_kind != "clean" and v.floor_status.get(asset.floor) == "PASS"]
        if not variants:
            return {"accepted": False, "rejection_reasons": ["缺少用于独立复核的路线资产"]}
        variant = next((v for v in variants if v.variant_id == "route_lhp_solo"), variants[0])
        key = (profile.key, asset.layout, variant.variant_id, asset.floor)
        if key not in asset_cache:
            asset_cache[key] = load_route_overlay_asset(variant.calibration_path, asset.floor)
        evidence_asset = asset_cache[key]
        layer, _ = warp_route_overlay(evidence_asset, registration.clean_to_screen,
                                      (layer.shape[1], layer.shape[0]), 1.0)
    structure = warp_route_structure(evidence_asset, registration.clean_to_screen,
                                     (layer.shape[1], layer.shape[0]))
    validation = validate_route_overlay(registration, layer, structure)
    validation.update(display_variant=asset.variant, evidence_variant=evidence_asset.variant)
    if (validation["accepted"] and registration.validation.get("credible_geometry_confirmed")
            and not _route_strongly_confirms(validation)):
        # Printed route annotations can obscure a correctly aligned room. Keep
        # the display asset's ordinary checks, and seek independent pose support
        # in another calibrated route drawing (never the clean template itself).
        for variant in catalog.variants(profile, asset.layout):
            if (variant.asset_kind == "clean" or variant.variant_id == evidence_asset.variant
                    or variant.floor_status.get(asset.floor) != "PASS"):
                continue
            key = (profile.key, asset.layout, variant.variant_id, asset.floor)
            if key not in asset_cache:
                asset_cache[key] = load_route_overlay_asset(variant.calibration_path, asset.floor)
            corroborator = asset_cache[key]
            size = (layer.shape[1], layer.shape[0])
            alternative, _ = warp_route_overlay(corroborator, registration.clean_to_screen, size, 1.0)
            supporting = validate_route_overlay(registration, alternative,
                warp_route_structure(corroborator, registration.clean_to_screen, size))
            if supporting["accepted"] and _route_strongly_confirms(supporting):
                validation["independent_pose_corroboration"] = {**supporting, "variant": variant.variant_id}
                break
    return validation


def switch_result_version(result, *, catalog, profile, variant_id, opacity, asset_cache, prepared_cache):
    """Redraw an accepted capture at the same pose; never run recognition again."""
    if result.registration and result.registration.validation.get('coordinate_space') == 'original_source_image':
        raise ValueError('定尺度模板匹配 直接显示三层完整原图；切换到 CPU/GPU 后可使用分层路线版本')
    if not result.accepted or result.registration is None or result.registration.clean_to_screen is None:
        raise ValueError("当前没有可复用的有效配准结果")
    if result.profile_key != profile.key:
        raise ValueError("当前配准属于其他地图或难度")
    started = time.perf_counter()
    variant = next((v for v in catalog.variants(profile, result.selected_layout) if v.variant_id == variant_id), None)
    if variant is None:
        raise ValueError("所选地图版本不存在")
    key = (profile.key, result.selected_layout, variant_id, result.floor)
    if key not in asset_cache:
        asset_cache[key] = load_route_overlay_asset(variant.calibration_path, result.floor)
    asset = asset_cache[key]
    registration = result.registration
    size = (result.screenshot.shape[1], result.screenshot.shape[0])
    layer, _ = warp_route_overlay(asset, registration.clean_to_screen, size, opacity)
    validation = validate_display_asset(registration, layer, asset, catalog=catalog, profile=profile, asset_cache=asset_cache)
    if any(registration.validation.get(k) for k in ("route_corroborated", "scale_only_relaxed", "history_background_soft_override", "sparse_refinement_confirmed")):
        if not _route_strongly_confirms(validation):
            validation["accepted"] = False
            validation["rejection_reasons"].append("当前姿态需要独立路线强复核，所选版本证据不足")
    prepared = prepared_cache.get((profile.key, result.selected_layout, result.floor))
    if registration.validation.get("scale_only_relaxed") and prepared is not None:
        validation["route_pose_disagreement"] = measure_route_pose_disagreement(
            asset, registration.clean_to_screen, _plausible_route_poses(registration, prepared), size)
    _record_display_scope(validation, registration, layer)
    projection, _ = composite_route_layer(result.screenshot, layer)
    elapsed = (time.perf_counter() - started)*1000
    return replace(result, variant_id=variant_id, variant_source=variant.source_name,
                   asset_path=str(asset.asset_path), calibration_path=str(variant.calibration_path),
                   overlay_validation=validation, layer=layer if validation["accepted"] else None,
                   projection=projection if validation["accepted"] else None,
                   accepted=validation["accepted"], reasons=list(validation["rejection_reasons"]),
                   timings_ms={**result.timings_ms, "version_switch": elapsed}, run_directory=None,
                   attempts=[{"layout": result.selected_layout, "variant": variant_id,
                              "accepted": validation["accepted"], "method": "reuse_accepted_pose"}])


def _record_display_scope(validation, registration, layer):
    # Display the entire asset after route verification. Keep the registration's
    # evidence scope unchanged so uncertain scale cannot become a trusted prior.
    validation.update(projection_scope='full',
        evidence_scope=registration.validation.get('projection_scope','full'),
        rendered_pixels=int(np.count_nonzero(layer[:, :, 3])))


def _route_can_corroborate(registration: RuntimeRegistrationResult) -> bool:
    """Allow a rejected pose to reach the independent route-image check."""
    validation = registration.validation
    if validation.get('registration_algorithm') in {'component_tolerance', 'multiscale_structure', 'fixed_scale_template'}:
        return False
    return bool(
        registration.clean_to_screen is not None
        and registration.observation_mask is not None
        and validation.get("observed_structure_pixels", 0) >= 2500
        and validation.get("structure_precision", 0.0) >= 0.87
        and validation.get("semantic_precision", 0.0) >= 0.78
        and not validation.get("sparse_spatial_rejected")
        and not validation.get("sparse_scale_rejected")
        and not validation.get("sparse_motion_rejected")
        and not validation.get("mapped_fraction_rejected")
        and not validation.get("spatial_competitor_rejected")
        and not validation.get("regional_pose_rejected")
        and not validation.get("reliable_recovery_rejected")
        and (validation.get("history_conflict_rate") is None
             or validation["history_conflict_rate"] <= 0.20
             or validation.get("history_background_soft_override"))
    )


def _route_strongly_confirms(overlay_validation: dict) -> bool:
    supporting = overlay_validation.get("independent_pose_corroboration")
    if overlay_validation.get("accepted") and supporting and supporting.get("accepted"):
        if (supporting.get("observed_pixels", 0) >= 8000
                and supporting.get("structure_coverage", 0) >= .72
                and supporting.get("distance_p95_px", float("inf")) <= 7):
            return True
    return bool(
        overlay_validation.get("observed_pixels", 0) >= 8000
        and overlay_validation.get("structure_coverage", 0.0) >= 0.72
        and overlay_validation.get("distance_p95_px", float("inf")) <= 7.0
    )


def _plausible_route_poses(
    registration: RuntimeRegistrationResult,
    prepared: PreparedRegistrationTemplate,
) -> list[np.ndarray]:
    """Convert close match candidates to clean-map-to-screen poses."""
    if (registration.match is None or registration.observation_bbox is None
            or registration.clean_to_screen is None):
        return []
    candidates = registration.match.result.top_k_candidates[:12]
    if not candidates:
        return []
    best_score = max(item.score for item in candidates)
    x, y, _, _ = registration.observation_bbox
    poses = []
    for candidate in candidates:
        if candidate.score < best_score - 0.035:
            continue
        scale = float(candidate.scale)
        to_processed = np.float32([
            [scale, 0.0, float(candidate.top_left[0]) - scale * x],
            [0.0, scale, float(candidate.top_left[1]) - scale * y],
        ])
        to_clean = to_processed.copy()
        to_clean[:, 2] += np.float32(prepared.processed.crop_offset)
        to_clean /= np.float32(prepared.registration_scale)
        pose = cv2.invertAffineTransform(to_clean)
        if np.max(np.abs(pose - registration.clean_to_screen)) > 1e-3:
            poses.append(pose)
    return poses


def _scale_profile_route_poses(registration: RuntimeRegistrationResult) -> list[np.ndarray]:
    """Project sampled local scale ambiguity, without claiming an error bound."""
    regional = registration.validation.get("regional_evidence") or {}
    if (registration.screenshot_to_clean is None or registration.observation_bbox is None
            or regional.get("region_count", 0) < 4):
        return []
    ratios = regional.get("plausible_relative_scales", [])
    anchor = regional.get("scale_profile_anchor")
    if not ratios or anchor is None:
        return []
    x, y, _, _ = registration.observation_bbox
    screen_anchor = np.float64(anchor) + (x, y)
    original = np.asarray(registration.screenshot_to_clean, np.float64)
    clean_anchor = original[:, :2] @ screen_anchor + original[:, 2]
    poses = []
    for ratio in sorted({min(ratios), max(ratios)}):
        if abs(ratio-1.) < 1e-6:
            continue
        pose = original.copy()
        pose[:, :2] *= ratio
        pose[:, 2] = clean_anchor-pose[:, :2] @ screen_anchor
        poses.append(cv2.invertAffineTransform(pose))
    return poses


@dataclass
class WorkflowResult:
    screenshot: np.ndarray
    decision: dict | None
    selected_layout: str | None
    variant_id: str | None
    variant_source: str | None
    registration: RuntimeRegistrationResult | None
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
    floor_detection: dict | None = None
    floor_indicator_image: np.ndarray | None = None
    session_id: str | None = None
    projection_preview_pending: bool = False
    preview_error: str | None = None
    layout_recognition_backend: str = 'cnn'
    layout_retrieval: LayoutRetrievalResult | None = None
    recognition_window: list[int] | None = None

    def diagnostics(self) -> dict:
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile_key,
            "session_id": self.session_id,
            "projection_preview_pending": self.projection_preview_pending,
            "preview_error": self.preview_error,
            "floor": self.floor,
            "floor_detection": self.floor_detection,
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
    if result.floor_indicator_image is not None:
        _write_image(directory / "07_floor_indicator.png", result.floor_indicator_image)
    if result.registration:
        registration = result.registration
        if registration.observation_image is not None:
            _write_image(directory / "02_observation.png", registration.observation_image)
        if registration.observation_mask is not None:
            _write_image(directory / "03_observation_mask.png", registration.observation_mask)
        source = registration.search_segmentation
        if source is not None and source.uncertain_mask is not None:
            _write_image(directory / "08_fog_uncertain.png", source.uncertain_mask)
            x, y, w, h = registration.observation_bbox
            _write_image(directory / "09_search_mask.png", source.map_mask[y:y+h, x:x+w])
        if registration.alignment_preview is not None:
            _write_image(directory / "04_alignment.png", registration.alignment_preview)
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
    model_bundle: tuple[Any, dict, Any] | None,
    floor: str,
    layout_override: str | None,
    route_versions: dict,
    opacity: float,
    prepared_cache: dict[tuple[str, str, str], PreparedRegistrationTemplate],
    asset_cache: dict[tuple[str, str, str, str], RouteOverlayAsset],
    recognize_only: bool = False,
    registration_priors: dict[tuple[str, str, str], np.ndarray] | None = None,
    registration_soft_priors: dict[tuple[str, str, str], SoftPoseHint] | None = None,
    registration_evidence: dict[tuple[str, str, str], "TemporalEvidence"] | None = None,
    template_prepare_lock: Any | None = None,
    registration_backend: str = 'auto',
    defer_projection_preview: bool = False,
    template_cache_root: Path | None = None,
    layout_recognition_backend: str = 'cnn',
    layout_retriever: LayoutRetriever | None = None,
    recognition_reference_size: tuple[int, int] = (1180, 1050),
    template_variant_id: str = 'route_zhanshi_fast',
) -> WorkflowResult:
    started = time.perf_counter()
    template_cache_root = template_cache_root or CACHE_ROOT / 'route_projection_templates'
    if screenshot_bgr.ndim != 3 or screenshot_bgr.shape[2] != 3:
        raise ValueError("截图必须为 BGR 三通道图片")
    timings: dict[str, float] = {}
    decision = None
    retrieval = None
    if layout_recognition_backend not in ('cnn', 'icon_structure'):
        raise ValueError('未知的布局识别方法')
    if layout_recognition_backend == 'icon_structure' and (recognize_only or layout_override is None):
        if not recognize_only:
            raise ValueError('请先识别并确认布局，再进行配准投影')
        retrieval = (layout_retriever or LayoutRetriever(profile.clean_root)).recognize(
            screenshot_bgr, recognition_reference_size)
        decision = retrieval.decision()
        timings['classification'] = retrieval.timings_ms['total']
    elif layout_recognition_backend == 'cnn' and model_bundle is not None and (recognize_only or layout_override is None):
        model, checkpoint, device = model_bundle
        image = Image.fromarray(cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB))
        inference_started = time.perf_counter()
        ranking = predict_image(model, checkpoint, device, image)
        decision = prediction_decision(ranking, checkpoint)
        timings["classification"] = (time.perf_counter() - inference_started) * 1000.0
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
        floor=floor,
        template_path=None,
        asset_path=None,
        calibration_path=None,
        layout_recognition_backend=layout_recognition_backend,
        layout_retrieval=retrieval,
    )
    if recognize_only:
        # Recognition succeeds when candidates are available for manual review.
        # Keep decision['accepted'] as the separate automatic-lock confidence gate.
        result.selected_layout = decision["best_label"] if decision else None
        result.accepted = bool(result.selected_layout)
        if not result.accepted:
            result.reasons.extend(retrieval.reasons if retrieval else ["模型未返回可供选择的布局候选"])
        result.timings_ms["total"] = (time.perf_counter() - started) * 1000.0
        return result
    if registration_backend == 'template':
        from src.registration.template_matching.direct import project_original
        result.floor = 'all'
        from src.app_catalog import TEMPLATE_VERSION_NAMES
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
    available = set(catalog.layouts(profile))
    candidates = ([layout_override] if layout_override else
                  [item["layout"] for item in decision["ranking"][:4]] if decision else [])
    if not candidates:
        result.reasons.append("没有可用的布局候选，请检查分类模型")
    for candidate in candidates:
        result.selected_layout = candidate
        result.variant_id = None
        result.variant_source = None
        result.registration = None
        result.overlay_validation = None
        result.layer = None
        result.projection = None
        result.template_path = None
        result.asset_path = None
        result.calibration_path = None
        attempt = {"layout": candidate, "accepted": False, "reasons": []}
        result.attempts.append(attempt)
        if candidate not in available:
            attempt["reasons"].append("缺少干净底图")
            continue
        variant = catalog.selected_variant(profile, candidate, route_versions)
        if variant is None:
            attempt["reasons"].append("缺少路线版本")
            continue
        if variant.floor_status.get(floor) != "PASS":
            attempt["reasons"].append(f"标定状态：{variant.floor_status.get(floor)}")
            continue
        template = catalog.clean_template(profile, candidate, floor)
        prepared_key = (profile.key, candidate, floor)
        if prepared_key not in prepared_cache:
            if template_prepare_lock is None:
                prepared_cache[prepared_key] = prepare_registration_template(
                    candidate, floor, template,
                    template_cache_root / profile.map_id / profile.difficulty_id,
                )
            else:
                with template_prepare_lock:
                    if prepared_key not in prepared_cache:
                        prepared_cache[prepared_key] = prepare_registration_template(
                            candidate, floor, template,
                            template_cache_root
                            / profile.map_id / profile.difficulty_id,
                        )
        evidence = None
        if registration_evidence is not None:
            evidence = registration_evidence.get(prepared_key)
            if evidence is None or not evidence.matches(prepared_cache[prepared_key]):
                evidence = TemporalEvidence.empty(prepared_cache[prepared_key])
                registration_evidence[prepared_key] = evidence
        registration = register_screenshot(
            screenshot_bgr,
            prepared_cache[prepared_key],
            prior_screenshot_to_clean=(registration_priors or {}).get(prepared_key),
            soft_pose_hint=(registration_soft_priors or {}).get(prepared_key),
            temporal_evidence=evidence,
            # Default dispatch remains compatible with historical injected
            # registration callables; an explicit CPU choice is forwarded.
            **({'backend': registration_backend} if registration_backend != 'auto' else {}),
        )
        timings["registration"] = timings.get("registration", 0.0) + float(registration.timings.get("total", 0.0))
        result.registration = registration
        result.template_path = str(template.resolve())
        if not registration.accepted and not _route_can_corroborate(registration):
            attempt["registration"] = registration.diagnostics()
            attempt["reasons"].extend(registration.rejection_reasons)
            continue
        asset_key = (profile.key, candidate, variant.variant_id, floor)
        if asset_key not in asset_cache:
            asset_cache[asset_key] = load_route_overlay_asset(variant.calibration_path, floor)
        asset = asset_cache[asset_key]
        projection_started = time.perf_counter()
        layer, _matrix = warp_route_overlay(
            asset, registration.clean_to_screen, (screenshot_bgr.shape[1], screenshot_bgr.shape[0]), opacity
        )
        timings["projection_warp"] = (time.perf_counter() - projection_started) * 1000.0
        composite_started = time.perf_counter()
        projection, _alpha = composite_route_layer(screenshot_bgr, layer)
        timings["projection_composite"] = (time.perf_counter() - composite_started) * 1000.0
        validation_started = time.perf_counter()
        overlay_validation = validate_display_asset(registration, layer, asset,
            catalog=catalog, profile=profile, asset_cache=asset_cache)
        _record_display_scope(overlay_validation, registration, layer)
        timings["projection_validation"] = (time.perf_counter() - validation_started) * 1000.0
        if registration.validation.get("scale_only_relaxed"):
            disagreement = measure_route_pose_disagreement(
                asset, registration.clean_to_screen,
                _plausible_route_poses(registration, prepared_cache[prepared_key]),
                (screenshot_bgr.shape[1], screenshot_bgr.shape[0]),
            )
            overlay_validation["route_pose_disagreement"] = disagreement
        profile_poses = _scale_profile_route_poses(registration)
        if profile_poses:
            profile_disagreement = measure_route_pose_disagreement(
                asset, registration.clean_to_screen, profile_poses,
                (screenshot_bgr.shape[1], screenshot_bgr.shape[0]))
            profile_disagreement["kind"] = "sampled_local_scale_sensitivity_not_error_bound"
            overlay_validation["scale_profile_disagreement"] = profile_disagreement
        timings["projection"] = timings.get("projection", 0.0) + (time.perf_counter() - projection_started) * 1000.0
        if (registration.accepted and registration.validation.get("scale_only_relaxed")
                and not _route_strongly_confirms(overlay_validation)):
            registration.accepted = False
            registration.rejection_reasons.append("同位置缩放放宽后，路线结构复核证据不足")
        if (registration.accepted and registration.validation.get("sparse_refinement_confirmed")
                and not _route_strongly_confirms(overlay_validation)):
            registration.accepted = False
            registration.rejection_reasons.append("稀疏尺度细化后，路线结构复核证据不足")
        if (registration.accepted and registration.validation.get("history_background_soft_override")
                and not _route_strongly_confirms(overlay_validation)):
            registration.accepted = False
            registration.rejection_reasons.append("历史背景证据冲突，路线复核未达到高可信标准")
        if not registration.accepted:
            if _route_strongly_confirms(overlay_validation):
                registration.validation["route_corroborated"] = True
                registration.validation["pre_route_rejection_reasons"] = list(registration.rejection_reasons)
                registration.rejection_reasons.clear()
                registration.accepted = True
            else:
                attempt["reasons"].extend(registration.rejection_reasons)
                attempt["reasons"].append("路线图未能充分确认稀疏画面的配准位置")
        attempt["registration"] = registration.diagnostics()
        attempt["overlay_validation"] = overlay_validation
        attempt["accepted"] = bool(registration.accepted and overlay_validation["accepted"])
        attempt["reasons"].extend(overlay_validation["rejection_reasons"])
        result.layer = layer
        result.projection = projection
        result.overlay_validation = overlay_validation
        result.variant_id = variant.variant_id
        result.variant_source = variant.source_name
        result.calibration_path = str(variant.calibration_path.resolve())
        result.asset_path = str(asset.asset_path)
        if attempt["accepted"]:
            if evidence is not None and registration.validation.get('projection_scope')!='local':
                evidence.clear_pending()
            result.accepted = True
            result.reasons = []
            break
    if not result.accepted:
        result.reasons = [f"{item['layout']}：{'；'.join(item['reasons']) or '未通过结构校验'}" for item in result.attempts]
        if not result.reasons:
            result.reasons.append("没有可用的布局候选")
    result.timings_ms["total"] = (time.perf_counter() - started) * 1000.0
    return result
