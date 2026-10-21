from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.registration.template_matching import matcher, direct


@pytest.mark.parametrize('origin', [(100, 40), (-40, 40), (40, -35), (40, 105)])
def test_expanded_search_recovers_known_pose_on_each_edge_without_changing_source(origin, monkeypatch):
    rng = np.random.default_rng(64)
    binary = (rng.random((220, 240)) < .45).astype(np.uint8) * 200
    image = np.repeat(binary[:, :, None], 3, axis=2)
    height, width = 140, 180
    x, y = origin
    yy, xx = np.mgrid[:height, :width]
    inside = (xx+x >= 0) & (xx+x < 240) & (yy+y >= 0) & (yy+y < 220)
    shot = (rng.random((height, width)) < .1).astype(np.uint8) * 200
    shot[inside] = binary[(yy+y)[inside], (xx+x)[inside]]
    shot = np.repeat(shot[:, :, None], 3, axis=2)
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', 1.)
    options = dict(map_image=image, image_fingerprint='known-geometry')
    cached = matcher.prepare_map('map', 1., include_masks=True, **options)
    saved = [array.copy() for array in cached[3:]]
    keys = set(matcher._map_cache)
    before_image, before_shot = image.copy(), shot.copy()
    normal = matcher._match_template(shot, 'map', 1., **options)
    result = matcher._match_template(shot, 'map', 1., expand_search=True, **options)
    assert result['location'] == origin and result['scale'] == 1.
    assert result['score'] > 0 and not result['search']['boundary_edges']
    assert result['search']['expanded']
    # The existing exclusive fog crop gives a 179 x 139 query here.
    assert result['search']['padding_xy'] == [89., 69.]
    assert set(matcher._map_cache) == keys
    for array, original in zip(cached[3:], saved):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(shot, before_shot)
    assert matcher._match_template(shot, 'map', 1., **options) == normal


def test_expansion_is_bounded_and_cannot_match_tiny_source_to_large_query(monkeypatch):
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', 1.)
    game = np.full((120, 120, 3), 200, np.uint8)
    image = np.full((20, 20, 3), 200, np.uint8)
    with pytest.raises(matcher.TemplateSizeError, match='可见区域大于'):
        matcher._match_template(game, 'small', 1., expand_search=True,
                                map_image=image, image_fingerprint='small')


@pytest.mark.parametrize('scenario,retry', [('interior', False), ('boundary', True),
    ('too_large', True), ('invalid', False), ('mixed_failure', False)])
def test_retry_recompares_configured_floors_and_preserves_interior_fast_path(monkeypatch, scenario, retry):
    image = np.full((80, 80, 3), 200, np.uint8)
    rgba = np.dstack((image, np.full((80, 80), 255, np.uint8)))
    snapshot = SimpleNamespace(entries={'2nd': {}, '1st': {}}, load=lambda floor:
        (dict(saved_scale=1. if floor == '2nd' else 2.), Path(floor), {}, image, rgba))
    calls = []

    def match(_game, path, scale, *, expand_search, **_options):
        calls.append((path, scale, expand_search))
        if not expand_search:
            if scenario == 'too_large' or (scenario == 'mixed_failure' and path == '2nd'):
                raise matcher.TemplateSizeError('固定尺度下截图可见区域大于原图搜索区域')
            if scenario in ('invalid', 'mixed_failure'):
                raise ValueError('invalid pixels')
        score = (3. if path == '2nd' else 2.) if not expand_search else (4. if path == '2nd' else 5.)
        return dict(score=score, location=(10, 20), search=dict(expanded=expand_search,
            boundary_edges=['right'] if scenario == 'boundary' and not expand_search else []))

    monkeypatch.setattr(matcher, '_match_template', match)
    attempts, best, _ = direct._match_floor_candidates(snapshot, np.full((20, 20, 3), 200, np.uint8), 1.5)
    assert sorted(calls) == sorted([(floor, scale, expanded)
        for floor, scale in [('2nd', 1.5), ('1st', 3.)]
        for expanded in ([False, True] if retry else [False])])
    assert [attempt['candidate_floor'] for attempt in attempts] == ['2nd', '1st']
    if retry:
        assert best[3] == '1st' and best[1] == 3. and best[2] == 2.
        assert all(attempt['search']['expanded'] for attempt in attempts)
        assert all(attempt['retry_trigger'] == ('best_at_boundary' if scenario == 'boundary'
            else 'all_candidates_too_large') for attempt in attempts)
        assert all('initial_attempt' in attempt for attempt in attempts)
    elif scenario == 'interior':
        assert best[3] == '2nd' and all('initial_attempt' not in attempt for attempt in attempts)
    else:
        assert best is None


def test_retry_keeps_load_failure_diagnostic_serializable(monkeypatch):
    import json
    image = np.full((80, 80, 3), 200, np.uint8)
    rgba = np.dstack((image, np.full((80, 80), 255, np.uint8)))

    def load(floor):
        if floor == '2nd':
            raise ValueError('source unavailable')
        return dict(saved_scale=1.), Path(floor), {}, image, rgba

    snapshot = SimpleNamespace(entries={'2nd': {}, '1st': {}}, load=load)
    monkeypatch.setattr(matcher, '_match_template', lambda *a, expand_search, **kw:
        dict(score=2., location=(0, 0), search=dict(expanded=expand_search,
            boundary_edges=[] if expand_search else ['left'])))
    attempts, best, _ = direct._match_floor_candidates(snapshot, image[:20, :20], 1.)
    assert best[3] == '1st'
    assert attempts[0]['initial_attempt']['reason'] == 'source unavailable'
    assert json.loads(json.dumps(attempts)) == attempts
