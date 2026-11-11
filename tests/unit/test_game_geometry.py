import math

import numpy as np
import pytest

from src.capture.region import CaptureRegion
from src.game_geometry import map_viewport, map_center, floor_slot_regions, floor_sample_region, recognition_size
from src.game_geometry import map_match_region
from src.registration.template_matching.screen import factor_for_size
from src.route_projection_core import split_projection_frame, floor_control_regions


def test_reported_fixed_height_layout_uses_right_anchor_and_gl_viewport():
    from src.game_geometry import resolve_geometry
    client = CaptureRegion(0, 0, 2560, 1334)
    geometry = resolve_geometry(client)
    assert geometry.policy == 'fixed_height'
    assert geometry.visible_size == (1440, 750)
    assert geometry.scale == pytest.approx(1.778666615, abs=1e-7)
    # Cocos rounds visible width upward and centers the resulting GL viewport.
    assert geometry.viewport_origin == pytest.approx((-.64, 0.))
    assert map_viewport(client).as_list() == [844, 149, 1374, 1036]
    assert map_center(client) == pytest.approx((1530.792, 667.))
    assert floor_slot_regions(client)[0].as_list() == [2205, 172, 119, 150]
    assert factor_for_size(2560, 1334) == pytest.approx(1920000/1779556)


@pytest.mark.parametrize('width,height,expected', [
    (1004, 755, 'fixed_width'), (2560, 1440, 'fixed_width'),
    (2560, 1334, 'fixed_height'), (3440, 1440, 'fixed_height'),
    (1334, 750, 'fixed_height'),
    (35570, 20000, 'fixed_width'), (35572, 20000, 'fixed_height'),
])
def test_policy_matches_reported_strong_smaller_tolerance(width, height, expected):
    from src.game_geometry import resolve_geometry
    assert resolve_geometry(CaptureRegion(0, 0, width, height)).policy == expected


@pytest.mark.parametrize('width,height,scale,visible_width,viewport_x', [
    (2560, 1440, 2560/1334, 1334, 0), (1920, 1080, 1920/1334, 1334, 0),
    (1004, 755, 1004/1334, 1334, 0), (1698, 980, 1698/1334, 1334, 0),
    (3440, 1440, 1.92, 1792, -.32), (2560, 1334, 1334/750, 1440, -.64),
])
def test_viewport_and_button_edges_at_both_policies(width, height, scale, visible_width, viewport_x):
    client = CaptureRegion(-100, 45, width, height)
    right_anchor = client.left+viewport_x+visible_width*scale
    view = map_viewport(client)
    assert view.left == math.floor(right_anchor-965*scale)
    assert view.left+view.width == math.ceil(right_anchor-193*scale)
    assert view.top == 45+max(0, math.floor(height/2-291*scale))
    assert view.top+view.height == 45+min(height, math.ceil(height/2+291*scale))
    assert map_center(client) == pytest.approx((right_anchor-579*scale, 45+height/2))
    slots = floor_slot_regions(client)
    for rect, (a, b) in zip(slots, [(194, 278), (134, 212), (74, 152), (14, 98)]):
        assert rect.left == math.floor(right_anchor-199.44*scale)
        assert rect.left+rect.width == math.ceil(right_anchor-133.44*scale)
        assert rect.top == 45+math.floor(height/2-b*scale)
        assert rect.top+rect.height == 45+math.ceil(height/2-a*scale)
    factor = (2560/1334)/scale
    assert factor_for_size(width, height) == pytest.approx(factor)
    normalized = recognition_size(client, view)
    assert normalized[0]/view.width == pytest.approx(factor, abs=.002)
    assert normalized[1]/view.height == pytest.approx(factor, abs=.002)


def test_height_only_change_translates_ui_without_resizing_it():
    short, tall = CaptureRegion(0, 0, 1004, 755), CaptureRegion(0, 0, 1004, 955)
    for a, b in zip((map_viewport(short), *floor_slot_regions(short)),
                    (map_viewport(tall), *floor_slot_regions(tall))):
        assert b.as_list() == [a.left, a.top+100, a.width, a.height]


@pytest.mark.parametrize('width,height', [(2560,1440), (1004,755), (1698,980), (3440,1440), (2560,1334)])
def test_match_crop_avoids_left_ui_and_floor_buttons_without_moving_viewport(width, height):
    client = CaptureRegion(-100, 40, width, height)
    view = map_viewport(client)
    crop = map_match_region(client, view)
    from src.game_geometry import resolve_geometry
    geometry = resolve_geometry(client)
    assert crop.left >= geometry.point(834)[0] > geometry.point(840)[0]
    assert crop.left+crop.width <= geometry.point(206)[0] < geometry.point(199.44)[0]
    assert crop.top == view.top and crop.height == view.height
    assert view.left < crop.left < crop.left+crop.width < view.left+view.width
    assert map_center(client) == pytest.approx(geometry.point(579))


def test_single_floor_hidden_control_needs_no_template_or_fake_confidence(monkeypatch):
    from src import route_projection_core as core
    monkeypatch.setattr(core, 'detect_floor', lambda _:pytest.fail('Hidden single-floor UI was matched'))
    client = CaptureRegion(0, 0, 1004, 755)
    control = floor_sample_region(client)
    frame = np.full((control.height, control.width, 3), 30, np.uint8)
    image, result = split_projection_frame(frame, control, control, control, client=client, single_floor='1st')
    assert result.floor == '1st' and result.buttons == ()
    assert result.diagnostics()['selection_source'] == 'single_floor_layout'
    assert result.score == result.margin == 0 and not result.candidates
    assert floor_control_regions(client, result.buttons) == []
    np.testing.assert_array_equal(image, frame)


def test_multi_floor_hidden_control_still_rejects():
    client = CaptureRegion(0, 0, 1004, 755)
    control = floor_sample_region(client)
    frame = np.full((control.height, control.width, 3), 30, np.uint8)
    _, result = split_projection_frame(frame, control, control, control, client=client)
    assert result.floor is None and result.buttons == () and result.reason


@pytest.mark.parametrize('client_size', [(2560, 1440), (1004, 755), (3440, 1440)])
@pytest.mark.parametrize('selected,expected', [(0, '2nd'), (1, '1st')])
def test_two_floor_controls_use_known_sprites_and_fixed_slots(client_size, selected, expected):
    import cv2
    from tests.support.floor_ui import template_control
    client = CaptureRegion(0, 0, *client_size)
    region = floor_sample_region(client)
    image = np.full((510, 140, 3), 30, np.uint8)
    image[20:250, 30:118] = template_control(4+selected)[135:365, 30:118]
    native = cv2.resize(image, (region.width, region.height), interpolation=cv2.INTER_AREA)
    _, result = split_projection_frame(native, region, region, region, client=client,
                                       available_floors=('2nd', '1st'))
    assert result.floor == expected, result.diagnostics()
    assert result.button_slots == (0, 1)
    assert len(result.candidates) == 2 and result.candidates[0]['valid']
    assert result.diagnostics()['thresholds'] == dict(strip_score=.8, row_min=.45, row_mean=.65, highlight_gap=15, margin=.08)


def test_two_floor_layout_does_not_guess_when_unprovided_basement_is_highlighted():
    from tests.support.floor_ui import template_control
    client = CaptureRegion(0, 0, 2560, 1440)
    region = floor_sample_region(client)
    _, result = split_projection_frame(template_control(6), region, region, region, client=client,
                                       available_floors=('2nd', '1st'))
    assert result.floor is None and result.reason
