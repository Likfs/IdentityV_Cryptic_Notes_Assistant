"""Check immutable mask reuse and complete response equivalence, not just maxima."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import cv2
import numpy as np
import pytest

from src.registration.template_matching import matcher


def image_inputs(seed=2):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (157, 193, 3), dtype=np.uint8)
    shot = image[30:81, 40:107].copy()
    return image, shot


def test_float_masks_reused_readonly_and_keyed_by_source_threshold_region_and_scale():
    image, _ = image_inputs()
    options = dict(map_image=image, image_fingerprint='first', include_masks=True)
    first = matcher.prepare_map('map', .5, **options)
    second = matcher.prepare_map('map', .5, **options)
    assert second[3] is first[3] and second[4] is first[4]
    np.testing.assert_array_equal(first[3], (first[0] > 0).astype(np.float32))
    np.testing.assert_array_equal(first[4], (first[0] == 0).astype(np.float32))
    with pytest.raises(ValueError):
        first[3][0, 0] = 9
    for changes in ({'image_fingerprint':'second'}, {'map_threshold':150},
                    {'search_region':[10, 12, 120, 100]}):
        other = matcher.prepare_map('map', .5, **(options | changes))
        assert other[3] is not first[3]
    assert matcher.prepare_map('map', .25, **options)[3] is not first[3]
    matcher._map_cache.clear()
    assert matcher.prepare_map('map', .5, **options)[3] is not first[3]


def test_mask_budget_falls_back_without_changing_pixels(monkeypatch):
    image, _ = image_inputs()
    options = dict(map_image=image, image_fingerprint='first', include_masks=True)
    monkeypatch.setattr(matcher, 'FLOAT_MASK_CACHE_BYTES', image.shape[0]*image.shape[1]*8)
    first = matcher.prepare_map('first', .5, **options)
    retained = first[3].nbytes + first[4].nbytes
    monkeypatch.setattr(matcher, 'FLOAT_MASK_CACHE_BYTES', retained + 1)
    second = matcher.prepare_map('second', .5, **options)
    again = matcher.prepare_map('second', .5, **options)
    assert second[3] is not again[3]
    assert sum(v[3].nbytes + v[4].nbytes for v in matcher._map_cache.values() if len(v) == 5) == retained
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a,b)
    # Eviction releases the cached masks together with their binary map.
    assert len(matcher._map_cache) == 2
    matcher._map_cache.clear()
    assert not matcher._map_cache


def legacy_response(map_white, map_black, prepared):
    white, black, wc, bc = prepared[:4]
    ww = np.clip(cv2.matchTemplate(map_white, white, cv2.TM_CCORR), 0., wc)
    bb = np.clip(cv2.matchTemplate(map_black, black, cv2.TM_CCORR), 0., bc)
    wb = np.maximum(wc - ww, 0.0)
    mw = np.maximum(ww + np.maximum(bc - bb, 0.0), 0.0)
    mb = wb + bb
    wn = np.sqrt(wc * mw)
    bn = np.sqrt(bc * mb)
    wn[wn <= 0] = 1.
    bn[bn <= 0] = 1.
    return matcher.TEMPLATE_WW*(ww/wn) + matcher.TEMPLATE_BB*(bb/bn) + matcher.TEMPLATE_WB*(wb/np.sqrt(wc*mb + 1e-12))


@pytest.mark.parametrize('ds,scale', [(1.,1.),(.5,1.49),(.25,1.5),(.1,3.178)])
@pytest.mark.parametrize('kind', ['random','white','sparse'])
def test_complete_response_and_inputs_unchanged(monkeypatch, ds, scale, kind):
    image, shot = image_inputs()
    if kind == 'white':
        image[:] = 255
        shot[:] = 255
    if kind == 'sparse':
        image[:] = 0
        image[::7, :] = 230
        image[:, ::9] = 230
        shot = image[30:81, 40:107].copy()
    # Keep enough source area for the largest fixed scale.
    image = cv2.resize(image, (420,380), interpolation=cv2.INTER_NEAREST)
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', ds)
    prepared = matcher.ScreenshotPreparation(shot)
    query = prepared.get(scale,ds)
    options = dict(map_image=image, image_fingerprint='fixture')
    bw, left, top = matcher.prepare_map('map', ds, **options)
    expected = legacy_response((bw>0).astype(np.float32), (bw==0).astype(np.float32), query)
    originals = [a.copy() for a in (image,shot,query[0],query[1])]
    before = cv2.minMaxLoc
    responses=[]
    def check(response):
        responses.append(response.copy())
        return before(response)
    monkeypatch.setattr(cv2,'minMaxLoc',check)
    result = matcher._match_template(shot,'map',scale,prepared=prepared,**options)
    np.testing.assert_array_equal(responses[0],expected)
    _, score, _, loc = before(expected)
    assert result['score'] == score
    assert result['location'] == (int((loc[0]+left-query[4])/ds),int((loc[1]+top-query[5])/ds))
    for a,b in zip((image,shot,query[0],query[1]), originals):
        np.testing.assert_array_equal(a,b)


def test_concurrent_matches_share_only_immutable_inputs(monkeypatch):
    image, shot = image_inputs()
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .5)
    options=dict(map_image=image,image_fingerprint='fixture')
    prepared=matcher.ScreenshotPreparation(shot)
    prepared.get(1.,.5)
    cached=matcher.prepare_map('map',.5,include_masks=True,**options)
    expected=matcher._match_template(shot,'map',1.,prepared=prepared,**options)
    copies=[a.copy() for a in cached[3:]]
    barrier=Barrier(2)
    def run(_):
        barrier.wait(timeout=10)
        return matcher._match_template(shot,'map',1.,prepared=prepared,**options)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run,range(2))) == [expected,expected]
    for a,b in zip(cached[3:],copies):
        np.testing.assert_array_equal(a,b)
