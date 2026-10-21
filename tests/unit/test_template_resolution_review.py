from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.registration.template_matching import direct, matcher


def source_snapshot(floors=('2nd', '1st', 'b1')):
    image = np.full((100, 100, 3), 200, np.uint8)
    rgba = np.dstack((image, np.full((100, 100), 255, np.uint8)))
    return SimpleNamespace(entries=dict.fromkeys(floors), load=lambda floor:
        (dict(saved_scale=1.49), Path(floor), {}, image, rgba))


@pytest.mark.parametrize('gap,trigger', [(0., True), (.999, True), (1., False), (1.001, False)])
def test_review_threshold_is_strict_and_uses_only_review_scores(monkeypatch, gap, trigger):
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    calls = []

    def match(game, path, scale, *, ds, expand_search, **options):
        assert scale == 1.49 and not expand_search
        calls.append((path, ds))
        scores = {'2nd': 13., '1st': 13.-gap, 'b1': 5.} if ds == .25 else {
            '2nd': 9., '1st': 10., 'b1': 11.}
        return dict(score=scores[path], scale=scale, location=(10, 20) if ds == .25 else (12, 22),
                    search=dict(expanded=False, boundary_edges=[]))

    monkeypatch.setattr(matcher, '_match_template', match)
    attempts, best, _ = direct._match_floor_candidates(source_snapshot(), np.full((30, 30, 3), 200, np.uint8), 1.)
    assert len(calls) == (6 if trigger else 3)
    assert matcher.TEMPLATE_DS == .25
    assert best[3] == ('b1' if trigger else '2nd')
    assert best[0]['ds'] == (.5 if trigger else .25)
    review = best[0]['resolution_review']
    assert review['triggered'] is trigger and review['initial_gap'] == pytest.approx(gap)
    if trigger:
        assert best[0]['score'] == 11.  # Every review score is below the coarse 13.
        assert best[0]['location'] == (12, 22) and best[1] == best[2] == 1.49
        assert review['initial_floor'] == '2nd' and review['final_floor'] == 'b1'
        assert [a['candidate_floor'] for a in attempts] == ['2nd', '1st', 'b1']
        assert all(a['ds'] == .5 for a in attempts)


@pytest.mark.parametrize('ds,floors', [(.5, ('2nd', '1st')), (1., ('2nd', '1st')), (.25, ('1st',))])
def test_single_candidate_and_high_resolution_do_not_trigger_or_downgrade(monkeypatch, ds, floors):
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', ds)
    calls = []
    def match(*args, ds, **options):
        calls.append(ds)
        return dict(score=12., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    _, best, _ = direct._match_floor_candidates(source_snapshot(floors), np.full((30, 30, 3), 200, np.uint8), 1.)
    assert calls == [ds]*len(floors)
    assert not best[0]['resolution_review']['triggered'] and best[0]['ds'] == ds


def test_failed_review_keeps_coarse_pose_and_serializable_failure_diagnostics(monkeypatch):
    import json
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    def match(*args, ds, **options):
        if ds == .5:
            raise ValueError('review unavailable')
        return dict(score=12., location=(8, 12))
    monkeypatch.setattr(matcher, '_match_template', match)
    attempts, best, _ = direct._match_floor_candidates(source_snapshot(), np.full((30, 30, 3), 200, np.uint8), 1.)
    assert best[3] == '2nd' and best[0]['ds'] == .25
    assert best[0]['location'] == (8, 12)
    review = best[0]['resolution_review']
    assert review['outcome'] == 'failed' and review['review_gap'] is None
    assert all(a['reason'] == 'review unavailable' for a in review['review_candidates'])
    assert json.loads(json.dumps(dict(attempts=attempts, review=review)))['review'] == review


def test_review_obeys_boundary_retry_and_retains_both_stages(monkeypatch):
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    calls = []
    def match(game, path, scale, *, ds, expand_search, **options):
        calls.append((path, ds, expand_search))
        score = (8. if path == '2nd' else 7.5) if ds == .25 else (9. if path == '2nd' else 11.)
        return dict(score=score, location=(0, 0), search=dict(expanded=expand_search,
            boundary_edges=[] if expand_search else ['left']))
    monkeypatch.setattr(matcher, '_match_template', match)
    attempts, best, _ = direct._match_floor_candidates(source_snapshot(('2nd', '1st')),
        np.full((30, 30, 3), 200, np.uint8), 1.)
    assert len(calls) == 8 and best[3] == '1st'
    review = best[0]['resolution_review']
    assert all(a['retry_trigger'] == 'best_at_boundary' for a in review['initial_candidates'])
    assert all(a['initial_attempt']['ds'] == .25 for a in review['initial_candidates'])
    assert all(a['initial_attempt']['ds'] == .5 for a in attempts)
    assert review['review_gap'] == 2.


def test_matcher_explicit_resolution_does_not_mutate_default_or_cached_pixels(monkeypatch):
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    image = np.zeros((90, 100, 3), np.uint8)
    image[20:65, 25:75] = 200
    game = image[15:70, 20:80].copy()
    options = dict(map_image=image, image_fingerprint='resolution-invariant')
    coarse = matcher._match_template(game, 'map', 1., **options)
    refined = matcher._match_template(game, 'map', 1., ds=.5, **options)
    assert matcher.TEMPLATE_DS == .25
    assert matcher._match_template(game, 'map', 1., **options) == coarse
    assert refined['scale'] == coarse['scale'] == 1.
    assert {key[2] for key in matcher._map_cache if key[0] == 'template'} == {.25, .5}
