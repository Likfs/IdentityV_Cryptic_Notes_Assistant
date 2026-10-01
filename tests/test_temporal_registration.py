from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.registration.runtime_registration import (
    _same_location_scale_consensus, prepare_registration_template, register_screenshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_strong_same_location_zoom_candidates_can_pass_sparse_review() -> None:
    first = SimpleNamespace(top_left=(726, 163), scale=0.9196, score=0.7368)
    zoomed = SimpleNamespace(top_left=(718, 157), scale=0.9596, score=0.7257)
    reviewed = [(first, None, None, False), (zoomed, None, None, False)]
    validation = {
        "observed_structure_pixels": 22748, "mapped_structure_fraction": 1.0, "structure_precision": 0.9773,
        "semantic_precision": 0.9105, "edge_median_error_px": 3.7,
        "edge_p95_error_px": 25.7,
    }
    direction = {
        "directional_comparable_pixels": 196, "directional_precision": 0.939,
        "topology_observed_pixels": 387, "topology_precision": 0.411,
    }
    accepted, spread = _same_location_scale_consensus(
        reviewed, (243, 285), validation, direction, 26822)
    assert accepted and 0.04 < spread < 0.08

    distant = SimpleNamespace(top_left=(790, 157), scale=0.9596, score=0.7257)
    assert not _same_location_scale_consensus(
        [reviewed[0], (distant, None, None, False)],
        (243, 285), validation, direction, 26822)[0]
    assert not _same_location_scale_consensus(
        reviewed, (243, 285), {**validation, "semantic_precision": 0.89},
        direction, 26822)[0]


def test_trusted_sparse_view_stays_local_and_ambiguous_view_is_rejected(tmp_path) -> None:
    cv2.setNumThreads(4)
    screenshot = cv2.imread(str(PROJECT_ROOT / 'tests/fixtures/registration/game2.png'))
    assert screenshot is not None
    prepared = prepare_registration_template(
        "layout_05", "2nd",
        PROJECT_ROOT / 'assets/maps/a_bringer_of_doom/nightmare/clean/layout_05/layout_05_2nd.png',
        tmp_path / "template_cache",
    )
    baseline = register_screenshot(screenshot, prepared)
    assert baseline.accepted and baseline.screenshot_to_clean is not None

    reduced = np.empty_like(screenshot)
    reduced[:] = screenshot[100, 100]
    reduced[475:650, 945:1040] = screenshot[475:650, 945:1040]
    without_history = register_screenshot(reduced, prepared)
    assert not without_history.accepted
    assert (without_history.validation.get("sparse_spatial_rejected")
            or without_history.validation.get("sparse_scale_rejected"))
    assert without_history.timings["method"] == "global_search"
    assert without_history.screenshot_to_clean is not None  # rejected pose retained for diagnostics
    tracked = register_screenshot(reduced, prepared, baseline.screenshot_to_clean)
    assert tracked.accepted
    assert tracked.timings["method"] == "temporal_reuse"
    assert np.allclose(tracked.screenshot_to_clean, baseline.screenshot_to_clean, atol=0.01)

    shifted = np.empty_like(screenshot)
    shifted[:] = screenshot[100, 100]
    shifted[475:650, 953:1048] = screenshot[475:650, 945:1040]
    relocated = register_screenshot(shifted, prepared, baseline.screenshot_to_clean)
    assert relocated.accepted
    assert relocated.timings["method"] == "temporal_search"
    assert relocated.screenshot_to_clean is not None
    old_point = baseline.screenshot_to_clean @ np.array([992.0, 550.0, 1.0])
    new_point = relocated.screenshot_to_clean @ np.array([1000.0, 550.0, 1.0])
    assert np.linalg.norm(old_point - new_point) < 4.0

    ambiguous = np.empty_like(screenshot)
    ambiguous[:] = screenshot[100, 100]
    ambiguous[530:580, 960:1000] = screenshot[530:580, 960:1000]
    rejected = register_screenshot(ambiguous, prepared, baseline.screenshot_to_clean)
    assert not rejected.accepted
    assert any("稀疏画面" in reason for reason in rejected.rejection_reasons)
