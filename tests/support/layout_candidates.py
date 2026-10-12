from __future__ import annotations
from src import layout_retrieval as retrieval


def candidates():
    return retrieval.LayoutRetrievalResult(candidates=[
        retrieval.LayoutCandidate(1, 'layout_22', 'layout_22/layout_22_2nd.png', 'a',
                                  .95, .9, .8, .98, '图标位置通过', {}),
        retrieval.LayoutCandidate(2, 'layout_27', 'layout_27/layout_27_2nd.png', 'b',
                                  .2, .3, .1, .1, '入口图标未可靠定位', {}),
    ], timings_ms={'total': 10.})
