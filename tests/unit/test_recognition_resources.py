"""Prepared indexes survive UI sessions and have one shared owner."""
from types import SimpleNamespace

import pytest

from src import recognition_resources as resources
from src.recognition_control import RecognitionCancelled, check_cancelled


def test_prepared_index_reuses_same_instance_without_reopening(tmp_path, monkeypatch):
    calls = []
    closed = []
    def prepare(**kwargs):
        calls.append(kwargs)
        kwargs['progress'](1, 1)
        return 'verified'
    retriever = SimpleNamespace(prepare=prepare, close=lambda:closed.append(True))
    monkeypatch.setattr(resources, 'create_layout_retriever', lambda *a, **kw:retriever)
    profile = SimpleNamespace(key='map/hard', clean_root=tmp_path/'clean', floors=('b1',))
    owner = resources.RecognitionResources(tmp_path/'cache')
    progress = []
    assert not owner.is_ready(profile.key, 'pure_structure')
    assert owner.prepare(profile, 'pure_structure', progress=lambda *value:progress.append(value)) == 'verified'
    assert owner.get(profile, 'pure_structure') is retriever
    assert owner.prepare(profile, 'pure_structure') == 'verified'
    assert len(calls) == 1 and progress == [(1, 1)]
    owner.close()
    owner.close()
    assert closed == [True] and not owner.is_ready(profile.key, 'pure_structure')


def test_cancelled_preparation_never_marks_ready_and_can_resume(tmp_path, monkeypatch):
    obsolete = [False]
    attempts = []
    def prepare(*, cancelled, progress):
        attempts.append(True)
        if len(attempts) == 1:
            obsolete[0] = True
        check_cancelled(cancelled)
        return 'ready'
    monkeypatch.setattr(resources, 'create_layout_retriever', lambda *a, **kw:SimpleNamespace(prepare=prepare))
    profile = SimpleNamespace(key='map/hard', clean_root=tmp_path/'clean', floors=('b1',))
    owner = resources.RecognitionResources(tmp_path/'cache')
    with pytest.raises(RecognitionCancelled):
        owner.prepare(profile, 'pure_structure', cancelled=lambda:obsolete[0])
    assert not owner.is_ready(profile.key, 'pure_structure')
    obsolete[0] = False
    assert owner.prepare(profile, 'pure_structure') == 'ready'
    assert len(attempts) == 2
    owner.close()
