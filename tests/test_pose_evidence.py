from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from src.registration.runtime_registration import _validation
from src.registration.temporal_evidence import PendingPoseHypotheses, TemporalEvidence, estimate_pending_translation
from src.registration_core.color_segmenter import SegmentationResult
from src.registration_core.pose_evidence import build_regional_evidence, regional_support, analyze_pose
from src.projection.route_overlay import composite_route_layer
from src.registration_core.masked_matcher import _support_correlation, _trimmed_mean
from test_first_sparse_registration import make_scene
from src.projection_workflow import _scale_profile_route_poses


def test_zero_border_correlation_preserves_every_response_coordinate():
    rng = np.random.default_rng(21)
    for dtype in (np.uint8, np.float32):
        template = rng.integers(0, 20, (180, 210)).astype(dtype)
        patch = np.zeros((90, 110), dtype)
        patch[41:65, 16:53] = rng.integers(0, 2, (24, 37))
        expected = cv2.matchTemplate(template, patch, cv2.TM_CCORR)
        actual = _support_correlation(template, patch)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected, atol=.01)
    values = rng.normal(size=900).astype(np.float32)
    assert np.isclose(_trimmed_mean(values, .15), np.sort(values)[:765].mean(), atol=1e-6)


def test_validation_crop_and_batched_warp_preserve_full_map_metrics():
    template, observed, _ = make_scene()
    for scale, x, y in ((1.1, 70, 65), (.87, -40, 20), (1.03, 430, 360),
                         (.713, 51.7, -19.3), (1., -1000, -1000)):
        pose = np.float32([[scale, 0, x], [0, scale, y]])
        result = _validation(pose, observed, template)
        size = (template.image.shape[1], template.image.shape[0])
        corridor, room, edge = [cv2.warpAffine(mask, pose, size, flags=cv2.INTER_NEAREST)
                                for mask in (observed.corridor_mask, observed.room_mask, observed.edge_mask)]
        mapped = corridor | room
        count = max(1, cv2.countNonZero(mapped))
        assert result["observed_structure_pixels"] == count
        assert result["structure_precision"] == cv2.countNonZero(mapped & template.map_mask)/count
        assert result["semantic_precision"] == (cv2.countNonZero(room & template.room_mask)
                                                + cv2.countNonZero(corridor & template.corridor_mask))/count
        distances = template.distance_transform[edge > 0]
        assert result["edge_median_error_px"] == (float(np.median(distances)) if len(distances) else None)


def test_new_evidence_crop_uses_full_screenshot_origin():
    template, observed, truth = make_scene()
    # Embed the identical observation at a nonzero screenshot coordinate.
    origin = (140, 95)
    full_masks = []
    for mask in (observed.corridor_mask, observed.room_mask, observed.map_mask, observed.edge_mask):
        full = np.zeros((400, 440), np.uint8)
        full[95:325, 140:330] = mask
        full_masks.append(full)
    full = SegmentationResult(*full_masks, np.zeros((400, 440), np.uint8), full_masks[2])
    global_pose = truth.copy()
    global_pose[:, 2] -= truth[:, :2] @ np.float32(origin)
    assert _validation(global_pose, observed, template, origin=origin) == _validation(global_pose, full, template)
    assert _validation(global_pose, observed, template)["semantic_precision"] < .5


def test_balanced_regions_reject_displacement_and_keep_fog_neutral():
    template, observed, truth = make_scene()
    evidence = build_regional_evidence(observed)
    wrong = truth.copy()
    wrong[:, 2] += (70, 90)
    correct, displaced = regional_support(template, evidence, [truth, wrong])
    assert correct["supported_regions"] >= 4
    assert correct["balanced_score"] > displaced["balanced_score"]+.25
    assert displaced["contradictory_regions"] >= 2
    # Background pixels never enter positive-evidence scoring.
    changed = replace(observed, known_background_mask=np.full_like(observed.map_mask, 255))
    assert regional_support(template, build_regional_evidence(changed), [truth])[0] == correct


def test_pose_analysis_separates_repeated_location_from_same_anchor_scale():
    template, observed, truth = make_scene()
    evidence = build_regional_evidence(observed)
    # Deliberately repeat the entire template at a second known location.
    double = replace(template, image=np.tile(template.image, (1, 2, 1)))
    for name in ("corridor_mask", "room_mask", "map_mask", "edge_mask", "stair_mask", "distance_transform"):
        setattr(double, name, np.tile(getattr(template, name), (1, 2)))
    repeated = truth.copy()
    repeated[0, 2] += template.image.shape[1]
    zoom = truth.copy()
    zoom[:, :2] *= 1.005
    zoom[:, 2] = truth[:, :2] @ evidence.center + truth[:, 2] - zoom[:, :2] @ evidence.center
    analysis = analyze_pose(double, evidence, truth, [repeated, zoom])
    assert analysis["spatial_competitors"] == 1
    assert analysis["same_anchor_competitors"] == 1
    assert analysis["uncertainty_kind"] == "sampled_local_support_not_error"


def test_projection_roi_preserves_pixels_and_empty_layer():
    rng = np.random.default_rng(8)
    screenshot = rng.integers(0, 256, (350, 480, 3), dtype=np.uint8)
    layer = np.zeros((350, 480, 4), np.uint8)
    layer[90:220, 120:360] = rng.integers(0, 256, (130, 240, 4), dtype=np.uint8)
    alpha = layer[:, :, 3].astype(np.float32)/255
    old = ((screenshot.astype(np.float32)/255*(1-alpha[:, :, None])
            + layer[:, :, :3].astype(np.float32)/255*alpha[:, :, None])*255).clip(0, 255).astype(np.uint8)
    actual, returned_alpha = composite_route_layer(screenshot, layer)
    assert np.array_equal(actual, old)
    assert np.array_equal(returned_alpha, layer[:, :, 3])
    assert np.array_equal(composite_route_layer(screenshot, np.zeros_like(layer))[0], screenshot)


def test_scale_sensitivity_keeps_visible_anchor_and_moves_distant_points():
    pose = np.float64([[1.2, 0, 50], [0, 1.2, 75]])
    result = SimpleNamespace(screenshot_to_clean=pose, observation_bbox=(100, 200, 300, 200),
                             validation={"regional_evidence": {"region_count": 6,
                                "scale_profile_anchor": [150, 100], "plausible_relative_scales": [.98, 1., 1.02]}})
    alternatives = _scale_profile_route_poses(result)
    screen_anchor = np.float64([250, 300])
    clean_anchor = pose[:, :2] @ screen_anchor+pose[:, 2]
    assert len(alternatives) == 2
    for alternative in alternatives:
        assert np.allclose(alternative[:, :2] @ clean_anchor+alternative[:, 2], screen_anchor)
        far_clean = pose[:, :2] @ (screen_anchor + [1000, 0])+pose[:, 2]
        assert np.linalg.norm(alternative[:, :2] @ far_clean+alternative[:, 2]-screen_anchor-[1000, 0]) > 15


def test_pending_pan_transports_candidates_but_does_not_confirm_them():
    old = np.zeros((400, 500), np.uint8)
    cv2.rectangle(old, (75, 65), (145, 140), 255, -1)
    cv2.rectangle(old, (185, 155), (250, 245), 255, -1)
    cv2.rectangle(old, (140, 100), (230, 112), 255, -1)
    moved = cv2.warpAffine(old, np.float32([[1, 0, 90], [0, 1, 45]]), (500, 400))
    cv2.rectangle(moved, (360, 250), (405, 300), 255, -1)
    shift = estimate_pending_translation(old, moved)
    assert shift is not None and np.linalg.norm(np.subtract(shift, (90, 45))) <= 1
    shape = old.shape
    memory = TemporalEvidence(shape, "test", *(np.zeros(shape, np.uint8) for _ in range(3)),
                              np.zeros(shape, np.uint32), np.zeros(shape, np.uint8))
    pose = np.float32([[1.2, 0, 30], [0, 1.2, 40]])
    other = pose.copy()
    other[0, 2] += 200
    memory.pending_poses = PendingPoseHypotheses([pose, other], old)
    new = memory.pending_new_mask(moved, allow_translation=True)
    assert new is not None and np.count_nonzero(new) >= 1200
    assert memory.updates == [] and not np.any(memory.structure_mask)
    transported = memory.pending_poses.transforms[0]
    assert np.allclose(transported[:, 2], pose[:, 2]-pose[:, :2] @ np.float32([90, 45]), atol=1)
    memory.pending_poses = PendingPoseHypotheses([pose, other], old)
    assert memory.pending_new_mask(moved) is None
    assert memory.pending_poses is None


def test_pan_refuses_single_wall_or_unrelated_view():
    wall = np.zeros((350, 450), np.uint8)
    wall[70:80, 20:340] = 255
    assert estimate_pending_translation(wall, np.roll(wall, 50, axis=0)) is None
    template, observed, _ = make_scene()
    assert estimate_pending_translation(template.map_mask, np.zeros_like(template.map_mask)) is None


def test_fine_search_scale_boundary_falls_back_to_full_grid(monkeypatch):
    from src.registration_core import cpu_pyramid_matcher as matcher
    from src.registration_core.masked_matcher import LocalizationResult, MatchCandidate
    template, observed, _ = make_scene()
    calls = []
    def fake_match(*, cfg, **kwargs):
        calls.append(cfg)
        is_coarse = not cfg["pre_match_search_windows_enabled"]
        radius = cfg.get("allowed_player_template_radius", 0)
        scale = cfg["scale_values"][0] if radius in (24., 48.) else 1.1
        position = (35, 32) if is_coarse else (70, 65)
        candidate = MatchCandidate(position, scale, .99, .99, .99, .99, 0,
                                   semantic_consistency=.99, edge_precision=.99)
        return LocalizationResult("test", position, scale, .99, False, [candidate], None, None)
    monkeypatch.setattr(matcher, "match_single_frame", fake_match)
    monkeypatch.setattr(matcher, "regional_support", lambda *a: [
        {"region_count": 8, "supported_regions": 8, "contradictory_regions": 0}])
    pyramid, _ = matcher.build_cpu_pyramid_template(template, factor=.5)
    result = matcher.match_single_frame_cpu_pyramid(
        template=template, pyramid_template=pyramid, observation=observed, pyramid_factor=.5,
        cfg={"scale_min": 1., "scale_max": 1.2, "staged_fine_search_enabled": True})
    trace = result.timing.sparse_refinement["fine_search_tiers"]
    assert len(trace) == 3 and trace[-1]["full_fallback"]
    assert calls[-1]["allowed_player_template_radius"] == 100.
    assert len(calls[-1]["scale_values"]) == 11


def test_audit_counts_missing_accepted_pose_as_invalid_and_groups_sessions():
    from tools.evaluation.audit_pose_quality import accepted_wrong, real_review_queue
    assert accepted_wrong({"accepted": True, "p95_screen_error_px": None})
    assert accepted_wrong({"accepted": True, "p95_screen_error_px": float("nan")})
    replay = {"rows": [{"source": str(i), "layout": "layout_01", "floor": "b1", "accepted": False,
                        "session_id": "one-game", "sha256": "s", "template_sha256": "t"} for i in range(3)]}
    review = real_review_queue(replay)
    assert review["human_labelled_cases"] == 0
    assert len({r["split"] for r in review["rows"]}) == 1
    assert all(not r["human_verified"] for r in review["rows"])


def test_regional_competitor_blocks_known_30px_false_acceptance(tmp_path):
    from src.app_catalog import MapCatalog
    from src.registration.runtime_registration import prepare_registration_template, register_screenshot
    from src.projection_workflow import _route_can_corroborate
    catalog = MapCatalog()
    profile = catalog.profile("a_bringer_of_doom", "nightmare")
    prepared = prepare_registration_template("layout_13", "b1",
        catalog.clean_template(profile, "layout_13", "b1"), tmp_path)
    template = prepared.processed
    canvas = np.full_like(template.image, 15)
    canvas[template.corridor_mask > 0] = (150, 142, 126)
    canvas[template.room_mask > 0] = (65, 103, 142)
    truth = np.float32([[1.0274015665054321, 0, 78.7654037475586],
                        [0, 1.0274015665054321, -596.2371826171875]])
    processed_to_screen = truth.copy()
    processed_to_screen[:, :2] /= prepared.registration_scale
    processed_to_screen[:, 2] += processed_to_screen[:, :2] @ np.float32(template.crop_offset)
    complete = cv2.warpAffine(canvas, processed_to_screen, (1000, 850), borderValue=(15, 15, 15))
    screenshot = np.full_like(complete, 15)
    screenshot[335:515, 380:620] = complete[335:515, 380:620]
    rejected = register_screenshot(screenshot, prepared)
    assert not rejected.accepted and rejected.validation.get("regional_pose_rejected")
    assert not _route_can_corroborate(rejected)
    prepared.config["regional_pose_guard_enabled"] = False
    baseline = register_screenshot(screenshot, prepared)
    assert baseline.accepted
    from tools.evaluation.benchmark_first_registration import pose_delta
    assert pose_delta(baseline.clean_to_screen, truth, screenshot.shape) > 25
