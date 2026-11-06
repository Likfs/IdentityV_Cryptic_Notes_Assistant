from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.floor_definitions import FLOORS
from src.registration.template_matching import direct, floors, matcher


@pytest.mark.parametrize('review_fails', [False, True])
def test_seven_floor_batches_preserve_ties_boundary_retry_and_review(monkeypatch, review_fails):
    image = np.full((100,100,3), 180, np.uint8)
    rgba = np.dstack((image,np.full((100,100),255,np.uint8)))
    snapshot = SimpleNamespace(entries=dict.fromkeys(FLOORS), _session=None, load=lambda floor:
        (dict(saved_scale=1.5), Path(floor), {}, image, rgba))
    protected = []
    monkeypatch.setattr(floors, 'retain_working_set', lambda records, **kwargs: protected.append(len(records)))
    monkeypatch.setattr(matcher, 'TEMPLATE_DS', .25)
    calls = []
    def match(game, path, scale, *, ds, expand_search, **kwargs):
        calls.append((path,ds,expand_search))
        if ds==.5 and review_fails:
            raise ValueError('review unavailable')
        return dict(score=10.+(2. if ds==.5 and path=='b4' else 0.), location=(4,8),
                    search=dict(boundary_edges=[] if expand_search else ['left']))
    monkeypatch.setattr(matcher,'_match_template',match)
    attempts,best,_=direct._match_floor_candidates(snapshot,np.full((30,30,3),180,np.uint8),1.,bounded=True)
    assert [a['candidate_floor'] for a in attempts] == list(FLOORS)
    assert best[3] == ('3rd' if review_fails else 'b4')
    assert best[6:] == (None,None), 'Completed candidates must not retain decoded floor arrays'
    assert all(count<=2 for count in protected)
    assert best[0]['resolution_review']['initial_floor'] == '3rd'
    assert best[0]['resolution_review']['outcome'] == ('failed' if review_fails else 'reviewed')
    assert len(calls) == (21 if review_fails else 28)
    assert all(a['initial_attempt']['ds']==(.25 if review_fails else .5) for a in attempts)
