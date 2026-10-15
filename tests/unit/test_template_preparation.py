import numpy as np
import pytest
from src.registration.template_matching import matcher, fog_crop


def test_request_shares_only_equivalent_processing_and_keeps_per_map_size_check(monkeypatch):
    game = np.random.default_rng(12).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    prepared = matcher.ScreenshotPreparation(game)
    first = prepared.get(1.49, .25)
    assert prepared.get(1.4900000000000002, .25) is first
    assert prepared.get(1.6, .25) is not first
    assert prepared.get(1.49, .5) is not first
    monkeypatch.setattr(matcher, 'TEMPLATE_GAME_THRESH', 128)
    assert prepared.get(1.49, .25) is not first
    assert matcher.ScreenshotPreparation(game.copy()).get(1.49, .25) is not first
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    with pytest.raises(ValueError, match='可见区域大于'):
        matcher._match_template(game, 'small', 1.49, map_image=game[:4,:4],
                                image_fingerprint='small', prepared=prepared)


@pytest.mark.parametrize('shape', [(1,1),(1,90),(90,1),(16,23),(128,200)])
def test_binary_crop_preserves_legacy_boundaries(shape):
    rng = np.random.default_rng(42)
    samples = [np.zeros(shape,np.uint8), np.full(shape,255,np.uint8)]
    for density in [.001,.01,.1,.5,.99]:
        samples.append((rng.random(shape)<density).astype(np.uint8)*255)
    for sample in samples:
        h,w = sample.shape
        left = fog_crop.scan(sample,'left') or 0
        right = fog_crop.scan(sample,'right') or w-1
        top = fog_crop.scan(sample,'top') or 0
        bottom = fog_crop.scan(sample,'bottom') or h-1
        actual = fog_crop.crop_binary(sample)
        assert (actual['left'],actual['top']) == (left,top)
        np.testing.assert_array_equal(actual['image'],sample[top:bottom,left:right])


def test_binary_crop_keeps_variance_threshold_for_long_sparse_rows():
    for width in (589,590,1000,2000):
        image=np.zeros((5,width),np.uint8)
        image[1:4,1]=255
        old=fog_crop.crop(image)
        new=fog_crop.crop_binary(image)
        assert (new['left'],new['top'])==(old['left'],old['top'])
        np.testing.assert_array_equal(new['image'],old['image'])
