import hashlib
import json

import cv2
import numpy as np
import pytest

from src.floor_indicator import detect_floor, DEFAULT_BUTTONS
from src.route_projection_core import (floor_control_regions, projection_capture_regions,
                                       split_projection_frame)
from src.capture.region import CaptureRegion
from src.capture.mouse_pan import PanInput
from src import route_projection_core as core
from tests.support.floor_ui import ROOT, recorded_control, template_control, replay_control


ACTUAL_CASES = json.loads((ROOT/'tests/fixtures/floor_ui/cases.json').read_text(encoding='utf-8'))['regressions']


def test_normal_size_uses_fast_path_without_repeated_matching(monkeypatch):
    calls = []
    original = cv2.matchTemplate
    def match(*args, **kwargs):
        if args[0].shape != args[1].shape:
            calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'matchTemplate', match)
    result = detect_floor(recorded_control())
    assert result.floor == '1st'
    assert result.diagnostics()['search_path'] == 'size_hint'
    # Count spatial searches, excluding single-position digit comparisons.
    # This detects repeated size search without machine-dependent wall timing.
    assert len(calls) == 7


def test_size_mismatch_falls_back_without_relaxing_acceptance():
    image = np.full((510, 140, 3), 30, np.uint8)
    image[25:449, 30:118] = cv2.resize(
        template_control(2)[25:485, 30:118], (88, 424), interpolation=cv2.INTER_AREA)
    result = detect_floor(image)
    assert result.floor == 'b3'
    assert result.diagnostics()['search_path'] == 'size_search'
    assert all(result.candidates[0]['checks'].values())
    assert result.diagnostics()['thresholds'] == dict(
        strip_score=.80, row_min=.45, row_mean=.65, highlight_gap=15, margin=.08)


@pytest.mark.parametrize('case', ACTUAL_CASES, ids=lambda c:c['id'])
def test_actual_window_sizes_and_shifted_floor_control(case):
    path = ROOT/'tests/fixtures/floor_ui'/case['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    result = detect_floor(cv2.imread(str(path)))
    assert result.floor == case['selected_floor'], result.diagnostics()
    assert result.diagnostics()['checks'] == dict(best_candidate_valid=True, margin=True)
    assert len(result.buttons) == (3 if case['selected_floor'] == '1st' else 4)


@pytest.mark.parametrize('case', ACTUAL_CASES, ids=lambda c:c['id'])
def test_actual_controls_share_click_and_drag_regions(case):
    image = cv2.imread(str(ROOT/'tests/fixtures/floor_ui'/case['image']))
    client = CaptureRegion.from_value(case['capture_context']['game_client_screen'])
    from src.game_geometry import floor_sample_region
    region = floor_sample_region(client)
    _, detection = split_projection_frame(replay_control(case), region, region, region, client=client)
    crop = CaptureRegion.from_value(case['capture_context']['floor_ui_roi_screen'])
    buttons = floor_control_regions(client, detection.buttons)
    assert len(buttons) == len(case['button_centers_in_crop'])
    hook = PanInput()
    hook.arm(1, 42, client.as_list(), [rect.as_list() for rect in buttons])
    for rect, (x, y) in zip(buttons, case['button_centers_in_crop']):
        point = (crop.left+x, crop.top+y)
        assert rect.left <= point[0] < rect.left+rect.width
        assert rect.top <= point[1] < rect.top+rect.height
        hook.feed('down', point, 42)
        hook.feed('move', (point[0]+5, point[1]), 42)
        hook.feed('up', point, 42)
    assert not hook.drain() and not hook.dragging


@pytest.mark.parametrize('case', ACTUAL_CASES, ids=lambda c:c['id'])
def test_real_crops_in_same_frame_preserve_map_pixels(case):
    client = CaptureRegion.from_value(case['capture_context']['game_client_screen'])
    map_region = CaptureRegion.from_value(case['capture_context']['map_roi_screen'])
    capture, control = projection_capture_regions(client, map_region)
    image = replay_control(case)
    source = np.random.default_rng(33).integers(0, 256, (map_region.height, map_region.width, 3), np.uint8)
    frame = np.zeros((capture.height, capture.width, 3), np.uint8)
    x, y = map_region.left-capture.left, map_region.top-capture.top
    frame[y:y+map_region.height, x:x+map_region.width] = source
    x, y = control.left-capture.left, control.top-capture.top
    frame[y:y+control.height, x:x+control.width] = image
    actual, detection = split_projection_frame(frame, capture, map_region, control, client=client)
    np.testing.assert_array_equal(actual, source)
    assert detection.floor == case['selected_floor']


@pytest.mark.parametrize('client_size', [(1004, 755), (1698, 980), (2560, 1600), (3440, 1440)])
@pytest.mark.parametrize('index,floor', list(enumerate(('b1', 'b2', 'b3', 'b4', '2nd', '1st', 'b1'))))
def test_all_states_at_different_client_aspects(client_size, index, floor):
    from src.floor_indicator import SEARCH_ROI, REFERENCE_SIZE
    client = CaptureRegion(100, 50, *client_size)
    region = core.game_capture_region(client, list(SEARCH_ROI), REFERENCE_SIZE)
    atlas = cv2.imread(str(ROOT/'assets/app_ui/floor_controls.png'))
    height = 460 if index < 4 else 345
    sx, sy = client_size[0]/2560, client_size[1]/1440
    ui_scale = min(sx, sy)
    width, height_scaled = round(88*ui_scale/sx), round(height*ui_scale/sy)
    control = np.full((510, 140, 3), 30, np.uint8)
    control[25:25+height_scaled, 30:30+width] = cv2.resize(
        atlas[460-height:, index*88:(index+1)*88], (width, height_scaled), interpolation=cv2.INTER_AREA)
    native = cv2.resize(control, (region.width, region.height), interpolation=cv2.INTER_AREA)
    detection = detect_floor(native)
    assert detection.floor == floor, detection.diagnostics()
    for actual, expected in zip(detection.buttons, [(2200, 205+i*height_scaled/(height/115)) for i in range(height//115)]):
        assert actual[0] == pytest.approx(expected[0], abs=3)
        assert actual[1] == pytest.approx(expected[1], abs=3)


@pytest.mark.parametrize('kind', ['wrong_order', 'missing_number', 'noise'])
def test_size_adaptation_does_not_accept_invalid_labels(kind):
    image = template_control(1)
    if kind == 'wrong_order':
        image[255:370], image[370:485] = image[370:485].copy(), image[255:370].copy()
    elif kind == 'missing_number':
        image[25+3*115+22:25+3*115+90, 48:100] = 30
    else:
        image = np.random.default_rng(7).integers(0, 256, image.shape, dtype=np.uint8)
    assert detect_floor(image).floor is None


@pytest.mark.parametrize('kind,floor', [('underground', 'b1'), ('mixed', '1st')])
@pytest.mark.parametrize('factor', [.5, .75, 1., 1.25])
def test_recorded_controls_identify_selected_floor_across_sizes(kind, floor, factor):
    cases = json.loads((ROOT/'tests/fixtures/floor_ui/cases.json').read_text(encoding='utf-8'))
    path = ROOT/'tests/fixtures/floor_ui'/f'{kind}.png'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cases[kind]['sha256']
    image = cv2.resize(recorded_control(kind), None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    detection = detect_floor(image)
    assert detection.floor == floor, detection
    assert len(detection.buttons) == (4 if kind == 'underground' else 3)
    assert detection.buttons[-1][1]+detection.buttons[-1][3] == pytest.approx(665, abs=5)


@pytest.mark.parametrize('index,floor', list(enumerate(('b1', 'b2', 'b3', 'b4', '2nd', '1st', 'b1'))))
def test_all_seven_states_use_their_label_not_just_row_index(index, floor):
    image = template_control(index)
    image = cv2.warpAffine(image, np.float32([[1, 0, -8], [0, 1, 7]]), (140, 510))
    detection = detect_floor(image)
    assert detection.floor == floor, detection
    assert detection.buttons[0][:2] == (2192, 212 if index < 4 else 322)


@pytest.mark.parametrize('value', [0, 30, 100, 255])
def test_blank_ui_never_falls_back_to_a_floor(value):
    result = detect_floor(np.full((510, 140, 3), value, np.uint8))
    assert result.floor is None and result.buttons == () and result.reason


def test_two_highlights_are_rejected():
    image = template_control(0)
    other = template_control(1)
    image[140:255] = other[140:255]
    assert detect_floor(image).floor is None


def test_same_capture_preserves_map_origin_and_pixels():
    client = CaptureRegion(100, 50, 2560, 1440)
    map_region = CaptureRegion(1001, 235, 1180, 1050)
    capture, control = projection_capture_regions(client, map_region)
    assert capture.as_list() == [1001, 230, 1409, 1055]
    source = np.random.default_rng(24).integers(0, 255, (1050, 1180, 3), np.uint8)
    # The helper uses screen coordinates; move the recorded control with the window.
    frame = np.zeros((capture.height, capture.width, 3), np.uint8)
    frame[5:, :1180] = source
    frame[:510, 1269:1409] = recorded_control()
    screenshot, detection = split_projection_frame(frame, capture, map_region, control)
    np.testing.assert_array_equal(screenshot, source)
    assert screenshot.shape == (1050, 1180, 3) and detection.floor == '1st'
    with pytest.raises(ValueError, match='尺寸'):
        split_projection_frame(frame[:-1], capture, map_region, control)


def test_click_and_drag_exclusions_share_scaled_individual_buttons():
    detection = detect_floor(recorded_control())
    client = CaptureRegion(100, 50, 1920, 1080)
    buttons = floor_control_regions(client, detection.buttons)
    assert len(buttons) == 3
    hook = PanInput()
    hook.arm(1, 42, client.as_list(), [button.as_list() for button in buttons])
    for button in buttons:
        point = (button.left+button.width//2, button.top+button.height//2)
        hook.feed('down', point, 42)
        hook.feed('move', (point[0]+10, point[1]), 42)
        hook.feed('up', point, 42)
    assert not hook.drain() and not hook.dragging
    # The empty slot above the three-floor strip is ordinary map space.
    hook.feed('down', (1780, 245), 42)
    assert hook.dragging and hook.drain()[0][1] == 'down'
    assert len(floor_control_regions(client)) == len(DEFAULT_BUTTONS) == 4


def test_game_floor_click_hits_vertical_buttons_not_the_old_horizontal_region(monkeypatch):
    detection = detect_floor(recorded_control('underground'))
    client = CaptureRegion(100, 50, 1920, 1080)
    monkeypatch.setattr(core, 'foreground_game_client', lambda _:client)
    monkeypatch.setattr(core.ctypes.windll.user32, 'GetAsyncKeyState', lambda _:0x8000)
    point = [0, 0]
    def cursor(pointer):
        pointer._obj.x, pointer._obj.y = point
        return True
    monkeypatch.setattr(core.ctypes.windll.user32, 'GetCursorPos', cursor)
    for rect in floor_control_regions(client, detection.buttons):
        point[:] = [rect.left + rect.width // 2, rect.top + rect.height // 2]
        assert core.floor_control_mouse_down(['game'], detection.buttons)
    point[:] = [1600, 110]
    assert not core.floor_control_mouse_down(['game'], detection.buttons)


@pytest.mark.parametrize('valid', [True, False])
def test_diagnostics_keep_all_candidates_and_immutable_input_at_actual_size(valid):
    image = cv2.resize(recorded_control('underground'), (104, 383), interpolation=cv2.INTER_AREA)
    if not valid:
        image[:] = 100
    original = image.copy()
    detection = detect_floor(image)
    assert (detection.floor is not None) == valid
    diagnostics = detection.diagnostics()
    assert diagnostics['thresholds'] == dict(strip_score=.8, row_min=.45, row_mean=.65, highlight_gap=15, margin=.08)
    assert diagnostics['input_size'] == [104, 383] and diagnostics['normalized_size'] == [140, 510]
    assert len(diagnostics['candidates']) == 7
    assert diagnostics['candidates'][0]['score'] == detection.score
    for candidate in diagnostics['candidates']:
        assert len(candidate['row_scores']) == len(candidate['labels'])
        assert candidate['score'] == pytest.approx((candidate['strip_score'] + candidate['row_mean']) / 2)
        assert candidate['valid'] == all(candidate['checks'].values())
        assert candidate['failed_checks'] == [key for key, passed in candidate['checks'].items() if not passed]
        assert len(candidate['background_medians']) == len(candidate['labels'])
    json.dumps(diagnostics, allow_nan=False)
    image[:] = 0
    np.testing.assert_array_equal(detection.control_image, original)
    assert not detection.control_image.flags.writeable and not detection.normalized_image.flags.writeable


def test_detection_error_still_returns_the_original_ui_crop(monkeypatch):
    from tests.support.floor_ui import projection_frame
    client = CaptureRegion(0, 0, 2560, 1440)
    map_region = CaptureRegion(100, 100, 40, 40)
    capture, control = projection_capture_regions(client, map_region)
    frame = projection_frame(capture.as_list(), np.full((40, 40, 3), 80, np.uint8), (100, 100))
    def failed_templates():
        raise ValueError('测试楼层模板尺寸错误')
    monkeypatch.setattr(core, 'detect_floor', lambda _:failed_templates())
    screenshot, detection = split_projection_frame(frame, capture, map_region, control)
    assert screenshot.shape == (40, 40, 3)
    assert detection.floor is None and '模板尺寸' in detection.error
    np.testing.assert_array_equal(detection.control_image, recorded_control())
    assert not detection.control_image.flags.writeable
