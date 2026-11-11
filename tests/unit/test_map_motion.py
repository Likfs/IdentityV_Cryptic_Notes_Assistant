import numpy as np
import pytest

from src.map_motion import MapMotion, zoom_factor


def test_additive_zoom_clamps_last_half_step_and_reverses_by_a_full_step():
    motion = MapMotion(np.eye(3), (100, 80))
    for step in range(9):
        motion.wheel(120)
        assert motion.zoom == pytest.approx(1 + (step + 1) * 2 / 7)
    assert motion.zoom == pytest.approx(25 / 7)
    motion.wheel(120)
    assert motion.zoom == pytest.approx(26 / 7)
    for _ in range(5):
        motion.wheel(120)
    motion.wheel(-120)
    assert motion.zoom == pytest.approx(24 / 7)
    motion.wheel(-120 * 20)
    assert motion.zoom == 1
    motion.wheel(-120)
    motion.wheel(120)
    assert motion.zoom == pytest.approx(9 / 7)


def test_fractional_wheel_accumulates_and_discards_outward_input_at_limits():
    motion = MapMotion(np.eye(3), (0, 0))
    motion.wheel(-60)
    motion.wheel(60)
    assert motion.zoom == 1
    motion.wheel(60)
    assert motion.zoom == pytest.approx(9 / 7)
    motion.wheel(1200)
    motion.wheel(60)
    motion.wheel(-120)
    assert motion.zoom == pytest.approx(24 / 7)
    motion.wheel(-60)
    motion.wheel(60)
    assert motion.zoom == pytest.approx(24 / 7)


def test_pan_zoom_order_moves_the_entire_pose_around_fixed_center_without_mutating_registration():
    pose = np.array([[.5, 0, 20], [0, .5, -10], [0, 0, 1.]])
    motion = MapMotion(pose, (100, 80))
    motion.pan(30, -20)
    motion.wheel(240)
    expected_origin = np.array([100, 80]) + 11 / 7 * (np.array([50, -30]) - [100, 80])
    np.testing.assert_allclose(motion.origin, expected_origin)
    assert motion.scale == pytest.approx(.5 * 11 / 7)
    motion.pan(-15, 5)
    motion.wheel(-240)
    np.testing.assert_allclose(motion.origin, np.array([50, -30]) + np.array([-15, 5]) * 7 / 11)
    assert motion.scale == pytest.approx(.5)
    np.testing.assert_array_equal(pose, [[.5, 0, 20], [0, .5, -10], [0, 0, 1]])


@pytest.mark.parametrize('pose', [np.diag([0, 0, 1]), np.diag([1, 2, 1]),
                                 [[1, .1, 0], [0, 1, 0], [0, 0, 1]],
                                 np.full((3, 3), np.nan), np.eye(2)])
def test_invalid_registration_cannot_start_following(pose):
    with pytest.raises(ValueError, match='等比例'):
        MapMotion(pose, (0, 0))


def test_reinitialized_pose_keeps_current_zoom_and_next_wheel_uses_its_actual_ratio():
    motion = MapMotion([[2, 0, 50], [0, 2, -30], [0, 0, 1]], (100, 80), zoom_units=26)
    motion.wheel(-120)
    assert motion.zoom == pytest.approx(24 / 7)
    assert motion.scale == pytest.approx(2 * 24 / 26)
    np.testing.assert_allclose(motion.origin, np.array([100, 80]) + 24 / 26 * (np.array([50, -30]) - [100, 80]))


@pytest.mark.parametrize('units', [6, 27, 7.5, True, float('nan')])
def test_unknown_or_out_of_range_zoom_cannot_be_used_for_registration(units):
    with pytest.raises(ValueError, match='地图倍率'):
        zoom_factor(units)
