"""Algorithm fixtures shared with GUI workflow tests; no Qt import here."""
import shutil

import pytest

from src.paths import PROJECT_ROOT
from src.layout_retrieval import LayoutRetriever


@pytest.fixture(scope='session')
def published_icon_scene(tmp_path_factory):
    from tests.support.structure_scene import publish_structure
    root = tmp_path_factory.mktemp('retrieval_scene')
    for layout in ('layout_05', 'layout_11', 'layout_15'):
        target = root / 'clean' / layout / f'{layout}_2nd.png'
        target.parent.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT/'tests/fixtures/retrieval/icon_sources'/f'{layout}_2nd.png', target)
    # Labelled source crops are enlarged from 600 to 1000 pixels: source/query=0.6.
    clean,_ = publish_structure(root, root/'clean', scale=.6, floors=('2nd',))
    return root, LayoutRetriever(clean, cache_root=root/'cache')


@pytest.fixture
def retrieval_scene(published_icon_scene,monkeypatch):
    from src import runtime_resources
    root,retriever = published_icon_scene
    # Matcher readers use their explicit package root, so GUI projection fixtures stay isolated.
    monkeypatch.setattr(runtime_resources,'PROJECT_ROOT',root)
    return retriever
