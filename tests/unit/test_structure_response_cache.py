"""Fixed response reuse must preserve float32 values and search semantics."""
import importlib
import cv2
import numpy as np
import pytest
from src.recognition_control import RecognitionCancelled

search = importlib.import_module('src.structure_layout.search')


def target():
    mask = np.ones((117, 149), np.uint8)
    mask[20:90, 25] = 0
    mask[60, 25:120] = 0
    distances = [cv2.distanceTransform(mask, cv2.DIST_L2, 5),
                 cv2.distanceTransform(mask.T.copy(), cv2.DIST_L2, 5)]
    return dict(dist=distances, scales=(1.5, .6176283703635228), coarse_resize=2/3)


def test_cached_responses_are_exact_readonly_and_release_replaced_distances(monkeypatch):
    t = target()
    expected = {s: [np.exp(-.5*(d/(search.SIGMA*s*t['coarse_resize']))**2)
                    for d in t['dist']] for s in t['scales']}
    assert search.prepare_response_cache([t])
    assert 'dist' not in t
    monkeypatch.setattr(search.np, 'exp', lambda *a, **k: pytest.fail('Response recomputed'))
    for s in t['scales']:
        maps = search.response_maps(t, s)
        assert maps is search.response_maps(t, s)
        for actual, wanted in zip(maps, expected[s]):
            np.testing.assert_array_equal(actual, wanted)
            assert not actual.flags.writeable


def test_cache_budget_falls_back_without_partial_changes():
    targets = [target(), target()]
    original = [t['dist'] for t in targets]
    assert not search.prepare_response_cache(targets, max_bytes=1)
    for t, distances in zip(targets, original):
        assert t['dist'] is distances and 'response_maps' not in t
        for actual, d in zip(search.response_maps(t, 1.5), distances):
            np.testing.assert_array_equal(actual, np.exp(-.5*(d/(search.SIGMA*1.5*(2/3)))**2))


def test_cancelled_cache_build_does_not_touch_target():
    t = target()
    with pytest.raises(RecognitionCancelled):
        search.prepare_response_cache([t], cancelled=lambda: True)
    assert 'dist' in t and 'response_maps' not in t
