from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.registration.runtime_registration import (
    _background_only_history_conflict,
    prepare_registration_template,
    register_screenshot,
)
from src.registration.temporal_evidence import TemporalEvidence
from src.projection_workflow import _route_can_corroborate, _route_strongly_confirms


ROOT = Path(__file__).resolve().parents[1]


def test_confirmed_structure_accumulates_and_rejects_conflicting_position(tmp_path) -> None:
    cv2.setNumThreads(4)
    screenshot = cv2.imread(str(ROOT / 'tests/fixtures/registration/game2.png'))
    assert screenshot is not None
    prepared = prepare_registration_template(
        "layout_05", "2nd",
        ROOT / 'assets/maps/a_bringer_of_doom/nightmare/clean/layout_05/layout_05_2nd.png',
        tmp_path / "template_cache",
    )
    baseline = register_screenshot(screenshot, prepared)
    assert baseline.accepted

    memory = TemporalEvidence.empty(prepared)
    assert memory.add_registration(baseline, prepared, source="capture-one")
    assert memory.updates[0].source == "capture-one"
    assert memory.updates[0].new_structure_pixels > 1000
    assert np.count_nonzero(memory.last_source_index) > 1000
    assert np.max(memory.confidence) > 0

    confirmed = register_screenshot(screenshot, prepared, baseline.screenshot_to_clean, memory)
    assert confirmed.accepted
    assert confirmed.timings["method"] == "temporal_reuse"
    assert confirmed.validation["history_conflict_rate"] == 0.0
    assert memory.add_registration(confirmed, prepared, source="capture-two")
    assert memory.predicted_center(np.array([1000.0, 550.0], np.float32)) is not None

    invalid = replace(baseline, validation={**baseline.validation, "route_corroborated": True})
    assert not memory.add_registration(invalid, prepared)
    relaxed = replace(baseline, validation={**baseline.validation, "scale_only_relaxed": True})
    assert not memory.add_registration(relaxed, prepared)
    assert len(memory.updates) == 2

    wrong = baseline.screenshot_to_clean.copy()
    wrong *= np.float32(prepared.registration_scale)
    wrong[:, 2] -= np.float32(prepared.processed.crop_offset)
    wrong[:, 2] += 80.0
    observation = baseline.observation_segmentation
    assert observation is not None and baseline.observation_bbox is not None
    compared, conflict = memory.conflict(
        corridor=observation.corridor_mask, room=observation.room_mask,
        background=observation.known_background_mask, transform=wrong,
        origin=baseline.observation_bbox[:2],
    )
    assert compared > 1000
    assert conflict is not None and conflict > 0.20


def test_only_background_dominated_conflict_can_request_route_review() -> None:
    validation = {
        "observed_structure_pixels": 18000,
        "structure_precision": 0.96,
        "semantic_precision": 0.89,
        "edge_median_error_px": 4.6,
    }
    details = {"compared_pixels": 8500, "conflict_rate": 0.22,
               "hard_conflict_pixels": 110}
    assert _background_only_history_conflict(validation, details)
    assert not _background_only_history_conflict(
        validation, {**details, "hard_conflict_pixels": 1500}
    )
    assert not _background_only_history_conflict(
        {**validation, "edge_median_error_px": 12.0}, details
    )

    candidate = SimpleNamespace(
        clean_to_screen=np.eye(2, 3), observation_mask=np.ones((2, 2), np.uint8),
        validation={**validation, "history_conflict_rate": 0.22,
                    "history_background_soft_override": True},
    )
    assert _route_can_corroborate(candidate)
    assert _route_strongly_confirms({"observed_pixels": 22000,
                                    "structure_coverage": 0.77,
                                    "distance_p95_px": 4.1})
    assert not _route_strongly_confirms({"observed_pixels": 22000,
                                        "structure_coverage": 0.714,
                                        "distance_p95_px": 4.1})
    candidate.validation.pop("history_background_soft_override")
    assert not _route_can_corroborate(candidate)
    candidate.validation["history_conflict_rate"] = 0.0
    candidate.validation["sparse_spatial_rejected"] = True
    assert not _route_can_corroborate(candidate)


def test_pending_candidates_require_new_structure_in_stationary_view(tmp_path) -> None:
    prepared = prepare_registration_template(
        "layout_05", "2nd",
        ROOT / 'assets/maps/a_bringer_of_doom/nightmare/clean/layout_05/layout_05_2nd.png',
        tmp_path / "template_cache",
    )
    memory = TemporalEvidence.empty(prepared)
    original = np.zeros((120, 160), np.uint8)
    original[20:70, 20:80] = 255
    first = np.float32([[1, 0, 5], [0, 1, 10]])
    other = np.float32([[1, 0, 105], [0, 1, 10]])
    memory.remember_candidates([first, first.copy(), other], original)
    assert memory.pending_poses is not None
    assert len(memory.pending_poses.transforms) == 2
    assert memory.pending_new_mask(original) is None

    revealed = original.copy()
    revealed[72:110, 20:80] = 255
    assert np.count_nonzero(memory.pending_new_mask(revealed)) >= 1200
    moved = np.zeros_like(original)
    moved[20:70, 95:155] = 255
    assert memory.pending_new_mask(moved) is None
    assert memory.pending_poses is None
