"""Known geometry, ambiguous rectangles, fog and independent frame evidence."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_first_sparse_registration import make_scene
from src.registration_core.credible_geometry import screenshot_evidence, review_geometry
from src.registration.temporal_evidence import TemporalEvidence
from src.registration.runtime_registration import prepare_registration_template, register_screenshot
from src.projection_workflow import _route_can_corroborate


def colored_scene():
    template, observation, truth = make_scene()
    observation.corridor_mask[observation.room_mask > 0] = 0
    image = np.full((*observation.map_mask.shape, 3), 15, np.uint8)
    image[observation.corridor_mask > 0] = (150, 142, 126)
    image[observation.room_mask > 0] = (65, 103, 142)
    return template, observation, truth, image


def test_material_color_is_not_an_icon_and_dark_fog_is_not_negative_evidence():
    _, observation, _, image = colored_scene()
    observation.known_background_mask = 255-observation.map_mask
    evidence = screenshot_evidence(image, observation)
    assert min(np.bincount(evidence.directions)) >= 3
    assert np.count_nonzero(evidence.trusted_masks[1]) > 5000
    assert not np.any(evidence.background_mask[:10])
    # Fake corridor label on a fogged room cannot enter trusted room/corridor history.
    image[45:60, 50:70] = (35, 50, 65)
    observation.corridor_mask[45:60, 50:70] = 255
    observation.room_mask[45:60, 50:70] = 0
    evidence = screenshot_evidence(image, observation)
    assert not np.any(evidence.trusted_masks[0][48:57, 53:67])


def test_joint_refinement_recovers_known_pose_and_respects_global_scale_bounds():
    template, observation, truth, image = colored_scene()
    evidence = screenshot_evidence(image, observation)
    seed = truth.copy()
    seed[:, :2] *= .98
    seed[:, 2] += (3, -2)
    pose, details = review_geometry(template, evidence, [seed], scale_min=1.0, scale_max=1.2,
                                    viewport=(300, 300), origin=(0, 0))
    assert 1.0 <= pose[0, 0] <= 1.2
    assert abs(pose[0, 0]-truth[0, 0]) < .015
    assert np.linalg.norm(pose[:, 2]-truth[:, 2]) < 2
    assert details['leave_one_wall_out_p95_px'] < 8
    assert details['wall_median_screen_px'] < 1


def test_duplicate_layout_is_not_unique_after_equal_refinement():
    template, observation, truth, image = colored_scene()
    for name in ('image', 'room_mask', 'corridor_mask', 'map_mask', 'edge_mask', 'stair_mask', 'distance_transform'):
        value = getattr(template, name)
        setattr(template, name, np.concatenate((value, value), axis=1))
    other = truth.copy()
    other[0, 2] += 460
    _, details = review_geometry(template, screenshot_evidence(image, observation), [other, truth],
        scale_min=.7, scale_max=1.4, viewport=(300, 300), origin=(0, 0))
    assert not details['accepted']
    assert details['position_status'] == 'ambiguous'
    assert details['competitive_branches'] >= 1


def test_single_rectangle_cannot_establish_absolute_position():
    template, observation, _, image = colored_scene()
    observation.room_mask[:] = 0
    observation.room_mask[20:120, 30:130] = 255
    observation.corridor_mask[:] = 0
    observation.map_mask[:] = observation.room_mask
    image[:] = 15
    image[observation.map_mask > 0] = (65, 103, 142)
    _, details = review_geometry(template, screenshot_evidence(image, observation),
        [np.float32([[1, 0, 60], [0, 1, 70]])], scale_min=.7, scale_max=1.4,
        viewport=(300, 300), origin=(0, 0))
    assert not details['accepted']
    assert not details.get('distinctive_structure', False)


def test_pending_repeated_frame_and_zoom_never_create_new_evidence():
    template, _, _ = make_scene()
    old = template.map_mask
    shape = old.shape
    memory = TemporalEvidence(shape, 'test', *(np.zeros(shape, np.uint8) for _ in range(3)),
                              np.zeros(shape, np.uint32), np.zeros(shape, np.uint8))
    pose = np.float32([[1, 0, 0], [0, 1, 0]])
    memory.remember_candidates([pose], old)
    stamp = memory.pending_poses.created_at
    assert memory.pending_new_mask(old, allow_translation=True) is None
    memory.remember_candidates([pose], old)
    assert memory.pending_poses.created_at == stamp
    zoom = cv2.getRotationMatrix2D((160, 180), 0, 1.12).astype(np.float32)
    changed = cv2.warpAffine(old, zoom, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)
    assert memory.pending_new_mask(changed, allow_translation=True) is None
    assert memory.pending_poses is None
    assert not memory.updates and not np.any(memory.structure_mask)
    memory.remember_candidates([pose], old)
    memory.pending_poses.created_at -= 121
    assert memory.pending_new_mask(old, allow_translation=True) is None
    assert memory.pending_poses is None


@pytest.mark.parametrize('name,floor,expected', [
    ('20260926_154255_737467_layout_14_1st', '1st', False),
    ('20260926_154751_759727_layout_14_b1', 'b1', True),
])
def test_reported_fog_capture_uses_shared_verifier_and_only_trusted_history(tmp_path, name, floor, expected):
    root = Path(__file__).resolve().parents[1]
    path = root/'runtime/diagnostics/route_projection/app_runs'/name/'01_screenshot.png'
    if not path.is_file():
        pytest.skip('Local user diagnostic is not distributed')
    image = cv2.imdecode(np.fromfile(path, np.uint8), 1)
    prepared = prepare_registration_template('layout_14', floor,
        root/f'assets/maps/a_bringer_of_doom/nightmare/clean/layout_14/layout_14_{floor}.png', tmp_path)
    # This fixture verifies the v4 verifier/history contract. Recovery has its
    # own real-case assertions in test_registration_recovery.py.
    prepared.config['registration_recovery_enabled'] = False
    result = register_screenshot(image, prepared)
    assert result.accepted == expected, result.rejection_reasons
    assert result.validation['credible_geometry']['position_status'] == 'confirmed'
    if expected:
        assert result.screenshot_to_clean[0, 0]*prepared.registration_scale > 1.5
        assert result.validation['credible_geometry_confirmed']
        assert np.count_nonzero(result.observation_segmentation.map_mask) < np.count_nonzero(result.observation_mask)
        memory = TemporalEvidence.empty(prepared)
        assert memory.add_registration(result, prepared)
    else:
        assert result.validation['sparse_scale_rejected']
        assert not _route_can_corroborate(result)
        assert not any('多个可能位置' in r for r in result.rejection_reasons)
