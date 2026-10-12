import numpy as np
import pytest
from tools.common.route_calibration import fit_fixed_scale


def test_fixed_scale_fit_recovers_scale_translation_and_reports_residuals():
    points = np.float64([[0, 0], [100, 0], [0, 100], [100, 100]])
    scale, diagnostics = fit_fixed_scale(points, points*.73+[27, 405])
    assert scale == pytest.approx(.73)
    assert diagnostics['translation'] == pytest.approx([27, 405])
    assert diagnostics['residual_rms_original_pixels'] < 1e-12
    with pytest.raises(ValueError, match='空间范围不足'):
        fit_fixed_scale(np.zeros((3, 2)), np.ones((3, 2)))
