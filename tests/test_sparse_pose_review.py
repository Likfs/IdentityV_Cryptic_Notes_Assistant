from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from test_first_sparse_registration import make_scene
from src.registration.runtime_registration import _validation, prepare_registration_template, register_screenshot
from src.registration_core.sparse_pose_review import (
    group_poses, observation_anchor, refine_anchor_scale, review_sparse_pose,
)
from src.projection_workflow import _route_can_corroborate


def test_source_coverage_is_invariant_to_zoom_but_detects_clipping():
    template, observation, _ = make_scene()
    for scale in (.3, .58, .8, 1.4):
        quality = _validation(np.float32([[scale, 0, 10], [0, scale, 10]]), observation, template)
        assert quality["mapped_structure_fraction"] == 1.0
    # Independent source-pixel oracle, with scale > 1 to catch area inflation.
    pose = np.float32([[1.6, 0, -140], [0, 1.6, -80]])
    yy, xx = np.nonzero(observation.map_mask)
    xy = np.column_stack((xx, yy)) @ pose[:, :2].T + pose[:, 2]
    expected = ((xy[:, 0] >= 0) & (xy[:, 0] < 460) & (xy[:, 1] >= 0) & (xy[:, 1] < 420)).mean()
    actual = _validation(pose, observation, template)["mapped_structure_fraction"]
    assert 0 < actual < 1 and actual == expected
    assert _validation(np.float32([[1, 0, -1000], [0, 1, -1000]]), observation, template)["mapped_structure_fraction"] == 0


def test_grouping_uses_visible_anchor_not_left_corner_and_keeps_distinct_positions():
    center = np.float32([300, 200])
    a = np.float32([[.5, 0, 50], [0, .5, 30]])
    b = np.float32([[.9, 0, -70], [0, .9, -50]])  # same visible anchor
    c = a.copy()
    c[0, 2] += 100
    assert group_poses([a, b, c], center) == [[0, 1], [2]]


def test_wall_scale_profile_recovers_known_zoom_without_moving_anchor():
    template, observation, truth = make_scene()
    center = observation_anchor(observation)
    anchor = truth[:, :2] @ center + truth[:, 2]
    initial = truth.copy()
    initial[:, :2] *= .965
    initial[:, 2] = anchor-initial[:, :2]@center
    estimated, details = refine_anchor_scale(template, observation, initial,
        scale_min=.7, scale_max=1.4, viewport=(500, 600), origin=(180, 110))
    assert np.allclose(estimated[:, :2]@center+estimated[:, 2], anchor, atol=1e-4)
    assert abs(estimated[0, 0]-truth[0, 0]) < .01
    assert details["status"] == "confirmed"
    assert details["viewport_disagreement_p95_px"] <= 8


def test_single_wall_never_confirms_scale():
    template, observation, truth = make_scene()
    observation.map_mask[:] = 0
    observation.map_mask[60:64, 20:170] = 255
    _, details = refine_anchor_scale(template, observation, truth,
        scale_min=.7, scale_max=1.4, viewport=(500, 600), origin=(0, 0))
    assert details["status"] == "insufficient_walls"


def test_route_overlap_cannot_override_unresolved_scale():
    registration = SimpleNamespace(clean_to_screen=np.eye(2, 3), observation_mask=np.ones((100, 100)),
        validation={"observed_structure_pixels": 10000, "structure_precision": 1.,
                    "semantic_precision": 1., "sparse_scale_rejected": True})
    assert not _route_can_corroborate(registration)


def test_duplicate_geometry_remains_spatially_ambiguous_after_equal_refinement():
    template, observation, truth = make_scene()
    # Duplicate the entire floor beside itself, preserving all local evidence.
    for name in ("image", "room_mask", "corridor_mask", "map_mask", "edge_mask", "stair_mask", "distance_transform"):
        value = getattr(template, name)
        setattr(template, name, np.concatenate((value, value), axis=1))
    other = truth.copy()
    other[0, 2] += 460
    _, details = review_sparse_pose(template, observation, truth, [other],
        scale_min=.7, scale_max=1.4, viewport=(500, 600), origin=(0, 0))
    assert details["position_status"] == "ambiguous"
    assert details["competitive_spatial_groups"] >= 1
    assert details["scale"]["status"] == "not_evaluated"


def test_reported_layout14_capture_separates_location_from_scale(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / 'runtime/diagnostics/route_projection/app_runs/20260926_153922_481583_layout_14_2nd/01_screenshot.png'
    if not path.is_file():
        pytest.skip("User's local diagnostic capture is not distributed")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    prepared = prepare_registration_template("layout_14", "2nd",
        root / "assets/maps/a_bringer_of_doom/nightmare/clean/layout_14/layout_14_2nd.png", tmp_path)
    result = register_screenshot(image, prepared)
    assert result.validation["mapped_structure_fraction"] == 1
    assert not result.validation.get("mapped_fraction_rejected")
    assert result.validation["sparse_margin_status"] == "not_evaluated_low_score"
    assert result.validation["sparse_pose_review"]["position_status"] == "confirmed"
    assert "regional_evidence" in result.validation  # available even for rejected poses
    assert not any("多个可能位置" in r or "少量可见结构" in r for r in result.rejection_reasons)
