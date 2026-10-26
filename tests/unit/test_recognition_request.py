from threading import Event

import pytest
from src.recognition_control import RecognitionRequest


@pytest.mark.parametrize('change_at,deliver_first,valid', [
    (9., True, False), (9., False, False), (10., True, False),
    (11., True, True), (11., False, True),
])
def test_capture_boundary_uses_input_occurrence_not_delivery_order(change_at, deliver_first, valid):
    request = RecognitionRequest('work', 2, 'session', (), Event())
    if deliver_first:
        request.map_changed(change_at)
    request.captured(10.)
    if not deliver_first:
        request.map_changed(change_at)
    assert request.snapshot_ready is valid
    assert request.capture_invalid is not valid
    assert request.cancel.is_set() is not valid


def test_repeated_map_toggles_cannot_revive_cancelled_request():
    request = RecognitionRequest('work', 2, 'session', (), Event())
    request.captured(10.)
    for stamp in (11., 12., 13.):
        request.map_changed(stamp)
    assert request.snapshot_ready
    request.cancel.set()
    request.map_changed(14.)
    assert not request.snapshot_ready
