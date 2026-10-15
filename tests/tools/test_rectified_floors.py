import cv2
import numpy as np
import pytest

from tools.assets.calibration.rectify_single_floor_routes import rectify
from src.registration.template_matching import matcher


def test_anisotropic_route_becomes_a_fixed_scale_match(monkeypatch):
    clean = np.zeros((280, 320, 4), np.uint8)
    rng = np.random.default_rng(321)
    for _ in range(28):
        x, y = rng.integers([8, 8], [275, 235])
        cv2.rectangle(clean, (int(x), int(y)), (int(x)+22, int(y)+28), (150, 150, 150, 255), -1)
    native = cv2.resize(clean, (160, 224), interpolation=cv2.INTER_NEAREST)
    layer, matrix, translation = rectify(native, np.diag([2., 1.25, 1.]), (320, 280))
    assert layer.shape == (536, 576, 4)
    assert not layer[:128, :, 3].any()
    np.testing.assert_allclose(matrix, translation @ np.diag([2., 1.25, 1.]))
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', 1.)
    shot = clean[45:215, 40:260, :3].copy()
    match = matcher._match_template(shot, 'rectified', 1., map_image=layer[:, :, :3],
                                    image_fingerprint='rectified-test', map_threshold=48)
    np.testing.assert_allclose(match['location'], [168, 173], atol=2)
    assert match['score'] > 10


def test_invalid_geometry_is_not_silently_rectified():
    with pytest.raises(ValueError, match='矩阵无效'):
        rectify(np.zeros((20, 30, 4), np.uint8), np.zeros((3, 3)), (30, 20))
