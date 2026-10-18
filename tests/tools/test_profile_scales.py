"""Offline transfer must preserve measured ratios and reject unsupported floors."""
import numpy as np
import pytest

from tools.assets.calibration.derive_profile_scales import solve_scales, match_link


def edge(a, b, ratio):
    return dict(a=a, b=b, ratio=ratio, inliers=50, span_a=[1200, 1800],
                span_b=[1200, 1800], residual_p95_original_pixels=.5)


def test_scale_transfer_preserves_different_source_resolutions():
    scales, links, rejected, error = solve_scales(
        ['anchor', 'first', 'second'],
        [edge('anchor', 'first', 1.1), edge('first', 'second', .8),
         edge('anchor', 'second', .88)], 'anchor', 3.178)
    assert scales == pytest.approx(dict(anchor=3.178, first=3.4958, second=2.79664))
    assert len(links) == 3 and not rejected and error < 1e-10


def test_unconnected_or_weak_single_relation_cannot_publish():
    with pytest.raises(ValueError, match='未连接'):
        solve_scales(['anchor', 'a', 'missing'], [edge('anchor', 'a', 1)], 'anchor', 3.178)
    weak = edge('anchor', 'a', 1)
    weak['inliers'] = 8
    with pytest.raises(ValueError, match='单一局部关系'):
        solve_scales(['anchor', 'a'], [weak], 'anchor', 3.178)


def test_shared_geometry_fit_measures_scale_without_assuming_equal_images():
    rng = np.random.default_rng(12)
    points = rng.uniform(0, 1800, (45, 2)).astype(np.float32)
    desc = rng.uniform(0, 255, (45, 128)).astype(np.float32)
    a = dict(points=points, desc=desc, angles=np.zeros(45))
    b = dict(points=points*1.125+[250, -170], desc=desc, angles=np.zeros(45))
    result = match_link(a, b)
    assert result['ratio'] == pytest.approx(1.125, abs=1e-6)
    assert result['inliers'] == 45
    assert result['residual_p95_original_pixels'] < .001
