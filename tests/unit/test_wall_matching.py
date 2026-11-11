"""Directional correlation geometry, ambiguity inputs and bounded feature reuse."""
import cv2
import numpy as np
import pytest
from src.registration.template_matching import walls, matcher, floors


@pytest.mark.parametrize('source_shape,query_shape', [((28,35),(12,15)), ((17,19),(24,29))])
def test_sparse_query_crop_preserves_full_plane_including_negative_origins(source_shape, query_shape):
    rng = np.random.default_rng(504)
    fields = [rng.random(source_shape, dtype=np.float32) for _ in range(4)]
    query = [(rng.random(query_shape)>.8).astype(np.float32) for _ in range(4)]
    for ch in query:
        ch[:3] = 0
        ch[:, :2] = 0
    hs, ws = source_shape
    hq, wq = query_shape
    bounds = (-wq+1, -hq+1, ws-1, hs-1)
    actual, count = walls.correlate(query, fields, bounds)
    expected = np.zeros_like(actual)
    for ch, field in zip(query, fields):
        target = np.pad(field, ((hq-1,hq-1),(wq-1,wq-1)))
        expected += cv2.matchTemplate(target, ch, cv2.TM_CCORR)
    expected /= count
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert cv2.minMaxLoc(actual)[3] == cv2.minMaxLoc(expected)[3]


@pytest.mark.parametrize('invalid', ['query', 'source'])
def test_missing_continuous_walls_is_an_explicit_failure(invalid):
    zero = [np.zeros((20,20),np.float32) for _ in range(4)]
    nonzero = [np.ones((20,20),np.float32) for _ in range(4)]
    with pytest.raises(ValueError, match='连续墙线'):
        walls.correlate(zero if invalid == 'query' else nonzero,
                        zero if invalid == 'source' else nonzero, (0,0,0,0))


def test_fields_are_immutable_fingerprinted_and_not_retained_over_budget(monkeypatch):
    image = np.zeros((100,120,3),np.uint8)
    cv2.rectangle(image,(15,20),(80,75),(140,140,140),2)
    monkeypatch.setattr(matcher,'_map_cache',{})
    first = walls.prepare_map('source',1.,map_image=image,image_fingerprint='first')
    assert first is walls.prepare_map('source',1.,map_image=image,image_fingerprint='first')
    assert all(not field.flags.writeable for field in first)
    changed = walls.prepare_map('source',1.,map_image=image,image_fingerprint='changed')
    assert changed is not first
    monkeypatch.setattr(walls,'FIELD_CACHE_BYTES',0)
    uncached = walls.prepare_map('other',1.,map_image=image,image_fingerprint='other')
    assert all(not field.flags.writeable for field in uncached)
    assert not any(key[1]=='other' for key in matcher._map_cache)
    monkeypatch.setattr(floors,'_images',{})
    floors.trim_uncached_templates()
    assert not matcher._map_cache


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_scale_cannot_start_wall_search(scale):
    with pytest.raises(ValueError, match='固定尺度'):
        walls.match(np.full((30,40,3),100,np.uint8),'unused',scale,ds=.25)
