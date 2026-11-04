"""Lossless numeric feature files and safe ownership boundaries."""
import numpy as np
import pytest

from src.structure_layout.feature_store import FeatureStore
from src.recognition_control import RecognitionCancelled


def test_round_trip_preserves_dtype_shape_values_and_read_only_mapping(tmp_path):
    original=dict(full=[np.arange(20,dtype=np.float32).reshape(4,5)],
                  rawdist=[np.arange(20,dtype=np.uint8).reshape(4,5)],
                  points=[np.empty((0,2),np.float32)],layout='layout_01',scales=[1.5])
    store=FeatureStore(tmp_path)
    try:
        key=store.ensure({'source':'test','version':1},lambda:original)
        with store.mapped(key) as target:
            for field in ('full','rawdist','points'):
                np.testing.assert_array_equal(target[field][0],original[field][0])
                assert target[field][0].dtype==original[field][0].dtype
                assert not target[field][0].flags.writeable
            assert target['layout']=='layout_01' and target['scales']==[1.5]
        assert store.built==1
    finally:
        store.close()
    second=FeatureStore(tmp_path)
    try:
        second.ensure({'source':'test','version':1},lambda:pytest.fail('Rebuilt unchanged features'))
        assert second.built==0
    finally:
        second.close()


def test_cancelled_write_leaves_no_partial_file_and_prune_respects_other_sessions(tmp_path):
    first=FeatureStore(tmp_path)
    second=FeatureStore(tmp_path)
    try:
        key=first.ensure({'source':'one'},lambda:dict(low=np.ones((3,4),np.float32)))
        path=first.files[key].path
        second.ensure({'source':'two'},lambda:dict(low=np.ones((3,4),np.float32)))
        unrelated=tmp_path/'user.txt';unrelated.write_text('keep')
        lookalike=tmp_path/('0'*16+'.'+'0'*64+'.feat');lookalike.write_text('not a feature package')
        second.prune()
        assert path.is_file() and unrelated.is_file()
        first.close();second.prune()
        assert not path.exists() and unrelated.is_file() and lookalike.is_file()
        with pytest.raises(RecognitionCancelled):
            second.ensure({'source':'cancel'},lambda:pytest.fail('Cancelled build started'),cancelled=lambda:True)
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        first.close();second.close()


def test_algorithm_identity_works_with_frozen_loader_without_source_files(monkeypatch):
    from types import SimpleNamespace
    from src.structure_layout import retriever
    codes={name:compile('value = '+str(index),'<frozen>', 'exec') for index,name in enumerate(
        ('features','local','search','retriever','correlation','feature_store'))}
    def spec(fullname):
        return SimpleNamespace(loader=SimpleNamespace(get_code=lambda name:codes[name.rsplit('.',1)[-1]]))
    monkeypatch.setattr(retriever.importlib.util,'find_spec',spec)
    from pathlib import Path
    monkeypatch.setattr(Path,'read_bytes',lambda *_:pytest.fail('Frozen fingerprint tried to read source files'))
    first=retriever.algorithm_code_fingerprints()
    assert len(first)==6 and first==retriever.algorithm_code_fingerprints()
    codes['features']=compile('value = 99','<frozen>','exec')
    second=retriever.algorithm_code_fingerprints()
    assert second['features'] != first['features']
    assert {k:v for k,v in second.items() if k!='features'}=={k:v for k,v in first.items() if k!='features'}
