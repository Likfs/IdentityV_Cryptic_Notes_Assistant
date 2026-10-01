from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.registration.runtime_registration import runtime_registration_config
from src.registration_core.color_segmenter import segment_map, add_known_background_evidence
from src.registration_core.fog_evidence import corridor_fog_mask, filter_corridor_fog


def scene(width=12):
    image = np.empty((280, 340, 3), np.uint8)
    bg = np.float32([68, 55, 44]); room = np.float32([90, 97, 110])
    image[:] = bg
    image[40:240, 30:65] = (105, 88, 79)
    image[110:220, 100:240] = room
    # A gradual room/fog mixture crosses the corridor color interval.
    for y in range(60, 95):
        alpha = .15 + .35*(y-60)/34
        image[y, 100:240] = bg+alpha*(room-bg)
    # A true narrow passage with sharp walls, even at this ambiguous color.
    image[45:245, 285:285+width] = (76, 68, 65)
    return image


@pytest.mark.parametrize('width', [4, 8, 12, 18])
def test_smooth_fog_is_suppressed_but_crisp_narrow_passage_stays(width):
    image = scene(width)
    cfg = runtime_registration_config()
    original = segment_map(image, cfg, infer_known_background=False, include_stairs=False)
    mask = corridor_fog_mask(image, original)
    assert np.count_nonzero(mask[60:95, 105:235]) > 1000
    assert not np.any(mask[45:245, 285:285+width])
    assert not np.any(mask[110:220, 100:240])  # real room
    assert not np.any(mask[40:240, 30:65])    # clear corridor


def test_uncertain_fog_stays_unknown_after_background_inference():
    image = scene(); cfg = runtime_registration_config()
    original = segment_map(image, cfg, infer_known_background=False, include_stairs=False)
    filtered = filter_corridor_fog(image, original, cfg)
    fog = filtered.uncertain_mask > 0
    assert np.any(fog)
    assert not np.any(filtered.map_mask[fog])
    assert not np.any(filtered.edge_mask[fog])
    refreshed = add_known_background_evidence(image, filtered, cfg)
    assert not np.any(refreshed.valid_mask[fog])
    assert not np.any(refreshed.known_background_mask[fog])
    assert np.all(refreshed.unknown_mask[fog] == 255)
    assert np.array_equal(original.map_mask, segment_map(image, cfg,
        infer_known_background=False, include_stairs=False).map_mask)


def test_no_bright_reference_keeps_uniform_dim_corridor():
    image = np.full((120, 160, 3), (68, 55, 44), np.uint8)
    image[40:90, 20:140] = (76, 68, 65)
    cfg = runtime_registration_config()
    original = segment_map(image, cfg, infer_known_background=False, include_stairs=False)
    assert filter_corridor_fog(image, original, cfg) is original


def test_reported_fog_strip_is_detected_without_manual_coordinates():
    path = Path(__file__).resolve().parents[1] / (
        'runtime/diagnostics/route_projection/app_runs/20260926_184200_218934_layout_10_b1/01_screenshot.png')
    if not path.exists():
        pytest.skip('Local diagnostic is not distributed')
    image = cv2.imdecode(np.fromfile(path, np.uint8), 1)
    cfg = runtime_registration_config()
    original = segment_map(image, cfg, infer_known_background=False, include_stairs=False)
    filtered = filter_corridor_fog(image, original, cfg)
    strip = original.map_mask[397:440, 541:686] & original.corridor_mask[397:440, 541:686]
    removed = filtered.uncertain_mask[397:440, 541:686] & strip
    assert cv2.countNonZero(removed) / cv2.countNonZero(strip) > .85


def test_failed_filtered_search_can_reuse_original_segmentation_once(monkeypatch):
    from src.registration import recovery
    @dataclass
    class Prepared:
        config: dict
    prepared = Prepared({'corridor_fog_filter_enabled': True})
    raw = object(); search = object()
    filtered = SimpleNamespace(accepted=False, validation={'fog_filter':{'enabled':True,'uncertain_pixels':100}},
        source_segmentation=raw, search_segmentation=search, rejection_reasons=['filtered search failed'],
        timings={'total':10, 'fog_filter':2})
    restored = SimpleNamespace(accepted=True, validation={}, rejection_reasons=[], timings={'total':8})
    calls = []
    def once(image, template, **kwargs):
        calls.append(kwargs)
        assert not template.config['corridor_fog_filter_enabled']
        assert kwargs['_source_segmentation'] is raw
        return restored
    monkeypatch.setattr(recovery, 'recover', lambda _image, _template, result, _once: result)
    result = recovery.recover_unfiltered_search(None, prepared, filtered, once)
    assert result is restored and len(calls) == 1
    assert prepared.config['corridor_fog_filter_enabled']
    assert result.search_segmentation is search
    assert result.validation['fog_filter']['raw_fallback_selected']
    assert result.timings['total'] >= filtered.timings['total']


@pytest.mark.parametrize('accepted,local,updates', [(True,False,0),(False,True,0),(False,False,1)])
def test_success_or_trusted_history_never_triggers_raw_retry(accepted, local, updates):
    from src.registration.recovery import recover_unfiltered_search
    result = SimpleNamespace(accepted=accepted, validation={
        'fog_filter':{'enabled':True,'uncertain_pixels':100}, 'local_projection_available':local})
    actual = recover_unfiltered_search(None, None, result,
        lambda *a, **k: pytest.fail('must not retry'), memory=SimpleNamespace(updates=updates))
    assert actual is result
