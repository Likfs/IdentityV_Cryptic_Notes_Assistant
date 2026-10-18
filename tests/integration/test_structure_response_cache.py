"""Cached normalized snapshots remain atomic, reusable and cancellable."""
from threading import Event
import importlib
import pytest
from src.recognition_control import RecognitionCancelled
from src.structure_layout import retriever as module
from tests.support.normalized_index import make_profile
from tools.assets.imports import build_recognition_index as builder


def test_normalized_response_cache_reuse_and_cancelled_rebuild(tmp_path, monkeypatch):
    clean = make_profile(tmp_path)
    monkeypatch.setattr(builder, 'project_relative', lambda p: 'assets/test/clean')
    builder.build(clean, clean.parent/'recognition')
    retriever = module.PureStructureRetriever(clean, floors=('1st', '2nd'))
    search = importlib.import_module('src.structure_layout.search')
    cancel = Event()
    try:
        fingerprint = retriever.prepare()
        snapshot = retriever._targets
        assert all('response_maps' in t and 'dist' not in t for t in snapshot)
        actual = search.response_maps
        def interrupted(*args):
            cancel.set()
            return actual(*args)
        monkeypatch.setattr(search, 'response_maps', interrupted)
        assert retriever.prepare() == fingerprint
        assert retriever._targets is snapshot and not cancel.is_set()
        monkeypatch.setitem(module.RESPONSE_CACHE_POLICY, 'version', 'test_policy_change')
        with pytest.raises(RecognitionCancelled):
            retriever.prepare(cancelled=cancel.is_set)
        assert retriever._targets is snapshot
        assert retriever._fingerprint == fingerprint
        monkeypatch.setattr(search, 'response_maps', actual)
        rebuilt = retriever.prepare()
        assert rebuilt != fingerprint
        assert retriever._targets is not snapshot
        assert all('response_maps' in t for t in retriever._targets)
        monkeypatch.setitem(module.RESPONSE_CACHE_POLICY, 'max_bytes', 1)
        assert retriever.prepare() != rebuilt
        assert all('dist' in t and 'response_maps' not in t for t in retriever._targets)
    finally:
        retriever.close()
    assert retriever._targets == [] and not retriever._fingerprint
