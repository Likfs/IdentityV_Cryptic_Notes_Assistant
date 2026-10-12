"""Algorithm fixtures shared with GUI workflow tests; no Qt import here."""
import shutil

import pytest

from src.app_catalog import MapCatalog
from src.layout_retrieval import LayoutRetriever


@pytest.fixture(scope='session')
def retrieval_scene(tmp_path_factory):
    root = tmp_path_factory.mktemp('retrieval_scene')
    profile = next(iter(MapCatalog().profiles.values()))
    for layout in ('layout_05', 'layout_11', 'layout_15'):
        target = root / 'clean' / layout / f'{layout}_2nd.png'
        target.parent.mkdir(parents=True)
        shutil.copy2(profile.clean_root / layout / target.name, target)
    return LayoutRetriever(root / 'clean', cache_root=root / 'cache')
