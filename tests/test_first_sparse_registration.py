from pathlib import Path
import gc
import weakref

import cv2
import numpy as np

from src.registration_core.color_segmenter import SegmentationResult
from src.registration_core.geometric_pair_index import GeometricPairIndex
from src.registration_core.masked_matcher import (
    _safe_match, _pad_template_for_matching, _template_directional_distance_fields,
    match_single_frame, review_pose_candidates,
)
from src.registration_core.sparse_pose_refiner import refine_sparse_poses
from src.registration_core.template_preprocessor import ProcessedTemplate
from src.app_catalog import MapCatalog
from src.registration.runtime_registration import prepare_registration_template, register_screenshot


def make_scene():
    room = np.zeros((420, 460), np.uint8)
    corridor = np.zeros_like(room)
    room[90:170, 90:190] = 255
    room[105:145, 175:225] = 255
    room[245:295, 160:235] = 255
    corridor[160:260, 135:149] = 255
    corridor[225:240, 135:210] = 255
    combined = cv2.bitwise_or(room, corridor)
    edge = cv2.Canny(combined, 40, 120)
    template = ProcessedTemplate(
        "point_test", np.zeros((*room.shape, 3), np.uint8), corridor, room, combined,
        edge, np.zeros_like(room), cv2.distanceTransform(255-edge, cv2.DIST_L2, 3),
        (17, 23), str(Path("synthetic.png")),
        corridor.astype(np.float32)/255, room.astype(np.float32)/255)
    truth = np.float32([[1.1, 0, 70], [0, 1.1, 65]])
    inverse = cv2.invertAffineTransform(truth)
    obs_room = cv2.warpAffine(room, inverse, (190, 230), flags=cv2.INTER_NEAREST)
    obs_corridor = cv2.warpAffine(corridor, inverse, (190, 230), flags=cv2.INTER_NEAREST)
    obs_map = cv2.bitwise_or(obs_room, obs_corridor)
    observed = SegmentationResult(obs_corridor, obs_room, obs_map, cv2.Canny(obs_map, 40, 120),
                                  np.zeros_like(obs_map), obs_map)
    return template, observed, truth


def test_binary_shortcut_preserves_masked_correlation_including_empty_overlap():
    rng = np.random.default_rng(31)
    template = np.uint8(rng.random((91, 103)) > .6) * 255
    template[:40, :40] = 0
    patch = np.uint8(rng.random((23, 21)) > .5) * 255
    baseline = _safe_match(template, patch, patch)
    current = _safe_match(template, patch, patch, binary_fast=True)
    expected = np.nan_to_num(baseline, nan=0, posinf=0, neginf=0)
    assert np.allclose(current, expected, atol=2e-5)
    assert current[0, 0] == 0


def test_template_cache_is_owned_and_released_with_source_asset():
    template, _, _ = make_scene()
    padded = _pad_template_for_matching(template, 24)
    _template_directional_distance_fields(padded)
    assert _pad_template_for_matching(template, 24) is padded
    reference = weakref.ref(padded)
    del padded, template
    gc.collect()
    assert reference() is None
    # A new template with identical metadata but different pixels owns its cache.
    another, _, _ = make_scene()
    another.room_mask[:] = 0
    assert not np.any(_pad_template_for_matching(another, 24).room_mask)


def test_cropped_search_preserves_scale_position_and_nonzero_crop_offset():
    template, observed, _ = make_scene()
    anchor = (observed.map_mask.shape[1]/2, observed.map_mask.shape[0]/2)
    cfg = {"scale_values": [1.06, 1.1, 1.14], "top_k": 6, "candidate_pool_size": 6,
           "candidate_border_padding": 24, "geometric_review_enabled": True,
           "geometric_review_candidate_count": 8, "pre_match_search_windows_enabled": True,
           "allowed_player_template_points": [[70 + 1.1*anchor[0] + 17, 65 + 1.1*anchor[1] + 23]],
           "allowed_player_template_radius": 16}
    old = match_single_frame(template=template, observation=observed, cfg=cfg, player_roi=anchor)
    new = match_single_frame(template=template, observation=observed,
                             cfg={**cfg, "first_sparse_optimized_enabled": True}, player_roi=anchor)
    assert old.best_position == new.best_position
    assert old.best_scale == new.best_scale
    assert [(c.top_left, c.scale) for c in old.top_k_candidates] == [(c.top_left, c.scale) for c in new.top_k_candidates]
    assert np.allclose([c.score for c in old.top_k_candidates], [c.score for c in new.top_k_candidates], atol=1e-5)


def test_point_refinement_recovers_translation_and_scale_on_partial_geometry():
    template, observation, truth = make_scene()
    poses, diagnostics = refine_sparse_poses(template, observation, [((77, 60), 1.13)],
                                             scale_min=.9, scale_max=1.3)
    assert poses and diagnostics["iterations"] <= 8
    pose = poses[0]
    estimated = np.float32([[pose.scale, 0, pose.position[0]], [0, pose.scale, pose.position[1]]])
    points = np.float32([[0, 0, 1], [190, 0, 1], [0, 230, 1], [190, 230, 1]])
    assert np.max(np.linalg.norm(points @ estimated.T - points @ truth.T, axis=1)) < 4
    # Exact candidate review still uses the template score definition.
    reviewed = review_pose_candidates(template, observation, [(pose.position, pose.scale)], {})
    assert reviewed and reviewed[0].semantic_consistency > .9


def test_geometry_voting_preserves_repeated_positions_instead_of_claiming_uniqueness():
    features = np.zeros((180, 210, 12), np.float32)
    observed = np.zeros((70, 65, 12), np.float32)
    landmarks = [(12, 12, 2), (40, 15, 3), (18, 48, 6), (45, 52, 8)]
    for x, y, channel in landmarks:
        observed[y-1:y+2, x-1:x+2, channel] = 1
        for tx, ty in ((20, 20), (120, 100)):
            features[y+ty-1:y+ty+2, x+tx-1:x+tx+2, channel] = 1
    index = GeometricPairIndex.build(features, 1.0)
    candidates = index.query(observed, 1.0, scale_min=.95, scale_max=1.05)
    assert any(np.linalg.norm(np.subtract(item["position"], (20, 20))) < 2 for item in candidates)
    assert any(np.linalg.norm(np.subtract(item["position"], (120, 100))) < 2 for item in candidates)


def test_conflicting_sparse_proposal_preserves_correct_wide_search(tmp_path):
    """Regression: narrowing around a plausible wall displaced this map by 80px."""
    catalog = MapCatalog()
    profile = catalog.profile("a_bringer_of_doom", "nightmare")
    prepared = prepare_registration_template(
        "layout_11", "b1", catalog.clean_template(profile, "layout_11", "b1"), tmp_path)
    template = prepared.processed
    canvas = np.full_like(template.image, 15)
    canvas[template.corridor_mask > 0] = (150, 142, 126)
    canvas[template.room_mask > 0] = (65, 103, 142)
    truth = np.float32([[.9116103649139404, 0, 237.01605224609375],
                        [0, .9116103649139404, -790.8101806640625]])
    processed_to_screen = truth.copy()
    processed_to_screen[:, :2] /= prepared.registration_scale
    processed_to_screen[:, 2] += processed_to_screen[:, :2] @ np.float32(template.crop_offset)
    complete = cv2.warpAffine(canvas, processed_to_screen, (1000, 850), borderValue=(15, 15, 15))
    screenshot = np.full_like(complete, 15)
    screenshot[265:585, 290:710] = complete[265:585, 290:710]
    result = register_screenshot(screenshot, prepared)
    assert result.accepted
    xx, yy = np.meshgrid(np.linspace(0, 999, 15), np.linspace(0, 849, 15))
    screen = np.float32(np.column_stack((xx.ravel(), yy.ravel()))).reshape(-1, 1, 2)
    clean = cv2.transform(screen, cv2.invertAffineTransform(truth))
    projected = cv2.transform(clean, result.clean_to_screen)
    assert np.percentile(np.linalg.norm((projected-screen).reshape(-1, 2), axis=1), 95) < 10
