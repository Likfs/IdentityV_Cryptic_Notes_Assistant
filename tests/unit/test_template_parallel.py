import hashlib
import os
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from src.registration.template_matching import direct, matcher, resources


def test_batch_verification_is_bounded_and_joins_before_cache_publish(tmp_path):
    items = []
    for index in range(12):
        path = tmp_path/f'{index}.png'
        path.write_bytes(str(index).encode())
        items.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    barrier = Barrier(4)
    threads = set()
    lock = Lock()
    def resolve(name):
        with lock:
            threads.add(get_ident())
        # Every group of four must run concurrently, without reacquiring the main lock.
        barrier.wait(timeout=10)
        return tmp_path/name
    with resources.resource_lock:
        resources.verify_fingerprints(items, resolve)
    assert len(threads) == 4
    assert len(resources.verified_fingerprints) == len(items)


def test_failed_batch_joins_workers_without_publishing_partial_fingerprints(tmp_path):
    items = []
    finished = []
    for index in range(4):
        path = tmp_path/f'{index}.png'
        path.write_bytes(b'pixels')
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if index else 'wrong'
        items.append((path.name, digest))
    barrier = Barrier(4)
    def resolve(name):
        barrier.wait(timeout=10)
        finished.append(name)
        return tmp_path/name
    with pytest.raises(ValueError, match='资源已改变'):
        resources.verify_fingerprints(items, resolve)
    assert len(finished) == 4
    assert not resources.verified_fingerprints
    # A subsequent valid request can acquire the lock and publish a complete cache.
    items[0] = (items[0][0], items[1][1])
    resources.verify_fingerprints(items, lambda name: tmp_path/name)
    assert len(resources.verified_fingerprints) == 4


def test_parallel_batch_still_reads_same_stat_json_content(tmp_path):
    path = tmp_path/'scale.json'
    path.write_bytes(b'{"scale":1}')
    stamp = path.stat()
    items = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest())]
    resources.verify_fingerprints(items, lambda name: tmp_path/name)
    path.write_bytes(b'{"scale":2}')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='资源已改变'):
        resources.verify_fingerprints(items, lambda name: tmp_path/name)


def test_parallel_floors_keep_input_order_ties_and_coordinator_only_preparation(monkeypatch):
    owner = get_ident()
    shot = np.zeros((40, 44, 3), np.uint8)
    shot[8:28, 9:31] = 200
    image = np.zeros((120, 140, 3), np.uint8)
    image[24:64, 30:74] = shot
    rgba = np.dstack((image, np.full(image.shape[:2], 255, np.uint8)))
    def load(floor):
        assert get_ident() == owner
        return dict(saved_scale=1.5, source_image=floor), Path(floor), {}, image, rgba
    snapshot = SimpleNamespace(entries={'2nd': {}, '1st': {}, 'b1': {}}, load=load)
    gate = Barrier(2)
    second_finished = Event()
    working = set()
    before = image.copy()
    original = matcher._match_template
    prepare = matcher.prepare_map
    get = matcher.ScreenshotPreparation.get
    def checked_prepare(*args, **kwargs):
        if get_ident() != owner:
            # A matching worker may look up a template, but must not populate the cache.
            keys = set(matcher._map_cache)
            result = prepare(*args, **kwargs)
            assert set(matcher._map_cache) == keys
            return result
        return prepare(*args, **kwargs)
    def checked_get(self, *args, **kwargs):
        size = len(self._groups)
        result = get(self, *args, **kwargs)
        if get_ident() != owner:
            assert len(self._groups) == size
        return result
    def match(game, path, scale, **kwargs):
        working.add(get_ident())
        if path != 'b1':
            gate.wait(timeout=10)
        if path == '2nd':
            assert second_finished.wait(10)
        result = original(game, path, scale, **kwargs)
        if path == '1st':
            second_finished.set()
        return result
    monkeypatch.setattr(matcher, '_match_template', match)
    monkeypatch.setattr(matcher, 'prepare_map', checked_prepare)
    monkeypatch.setattr(matcher.ScreenshotPreparation, 'get', checked_get)
    with resources.resource_lock:
        attempts, best, _ = direct._match_floor_candidates(snapshot, shot, 1.)
    assert len(working) == 2
    assert [a['candidate_floor'] for a in attempts] == ['2nd', '1st', 'b1']
    assert len({a['score'] for a in attempts}) == 1
    assert best[3] == '2nd'
    np.testing.assert_array_equal(image, before)


def test_unexpected_worker_failure_waits_for_other_floor_before_return(monkeypatch):
    shot = np.full((20, 20, 3), 200, np.uint8)
    image = np.full((80, 80, 3), 200, np.uint8)
    rgba = np.dstack((image, np.full((80, 80), 255, np.uint8)))
    snapshot = SimpleNamespace(entries={'2nd': {}, '1st': {}},
        load=lambda floor: (dict(saved_scale=1.), Path(floor), {}, image, rgba))
    gate = Barrier(2)
    completed = Event()
    def match(game, path, scale, **kwargs):
        gate.wait(timeout=10)
        if path == '2nd':
            raise RuntimeError('worker failure')
        completed.set()
        return dict(score=1., location=(0, 0))
    monkeypatch.setattr(matcher, '_match_template', match)
    with resources.resource_lock, pytest.raises(RuntimeError, match='worker failure'):
        direct._match_floor_candidates(snapshot, shot, 1.)
    assert completed.is_set()
