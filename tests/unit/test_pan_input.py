from src.capture.mouse_pan import PanInput, wheel_delta


def test_mailbox_preserves_short_drag_and_final_release_coalesces_only_moves():
    stream = PanInput()
    stream.arm(3, 42, [0, 0, 200, 100], [])
    stream.feed('down', (20, 20), 42)
    for x in range(30, 80):
        stream.feed('move', (x, 20), 42)
    stream.feed('up', (95, 30), 42)
    assert stream.drain() == [(3, 'down', (20, 20), 0), (3, 'move', (79, 20), 0), (3, 'up', (95, 30), 0)]


def test_outside_and_control_clicks_do_not_drag_but_started_drag_can_leave_roi():
    stream = PanInput()
    stream.arm(1, 42, [100, 100, 200, 100], [[110, 110, 20, 20]])
    for point in [(0, 0), (115, 115)]:
        stream.feed('down', point, 42)
        stream.feed('move', (180, 140), 42)
        stream.feed('up', (180, 140), 42)
    assert stream.drain() == []
    stream.feed('down', (180, 140), 42)
    stream.feed('up', (500, 400), 42)
    assert [e[1] for e in stream.drain()] == ['down', 'up']


def test_focus_loss_cancels_even_if_focus_returns_before_gui_tick():
    stream = PanInput()
    stream.arm(1, 42, [0, 0, 200, 100], [])
    stream.feed('down', (20, 20), 42)
    stream.feed('move', (30, 20), 99)
    stream.feed('up', (40, 20), 42)
    assert stream.drain() == [(1, 'cancel', (30, 20), 0)]


def test_mailbox_overflow_cancels_instead_of_losing_button_edges():
    stream = PanInput()
    stream.arm(1, 42, [0, 0, 200, 100], [])
    for _ in range(200):
        stream.feed('down', (20, 20), 42)
        stream.feed('up', (30, 20), 42)
    assert [e[1] for e in stream.drain()] == ['cancel']


def test_new_session_discards_previous_mouse_events():
    stream = PanInput()
    stream.arm(1, 42, [0, 0, 200, 100], [])
    stream.feed('down', (20, 20), 42)
    stream.arm(2, 42, [0, 0, 200, 100], [])
    stream.feed('up', (80, 20), 42)
    assert stream.drain() == []


def test_wheel_preserves_amount_order_and_active_drag_with_client_wide_bounds():
    stream = PanInput()
    stream.arm(1, 42, [100, 100, 200, 100], [], wheel_roi=[0, 0, 1000, 800])
    stream.feed('down', (120, 120), 42)
    stream.feed('move', (130, 120), 42)
    stream.feed('wheel', (130, 120), 42, 240)
    stream.feed('move', (140, 120), 42)
    stream.feed('up', (140, 120), 42)
    stream.feed('wheel', (500, 400), 42, -60)
    stream.feed('wheel', (1100, 400), 42, 120)
    events = stream.drain()
    assert [event[1] for event in events] == ['down', 'move', 'wheel', 'move', 'up', 'wheel']
    assert [event[3] for event in events if event[1] == 'wheel'] == [240, -60]
    stream.feed('wheel', (500, 400), 42, -120)
    assert stream.drain() == [(1, 'wheel', (500, 400), -120)]


def test_native_wheel_delta_is_signed_and_does_not_include_low_word():
    for amount in (120, -120, 240, -240, 60, -60):
        assert wheel_delta(((amount & 0xffff) << 16) | 0x1234) == amount
