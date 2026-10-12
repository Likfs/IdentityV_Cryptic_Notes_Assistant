import pytest
from src import route_projection_app as app
from tests.support.key_events import record, install


def test_held_map_key_never_schedules_open_capture(window, source, monkeypatch):
    install(window, source)
    window.settings.update(registration_backend='template', map_capture_delay_ms=150)
    window.map_state.reset()
    scheduled = []
    clock = [100.]
    monkeypatch.setattr(app.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append((delay, callback)))
    record(source, True)
    window._handle_map_key()
    # Holding longer than the configured capture delay must still do nothing.
    clock[0] = 102.
    record(source, True)
    window._handle_map_key()
    assert not scheduled and not window.projection_delay_pending and not window.map_state.is_open
    record(source, False)
    window._handle_map_key()
    assert window.map_state.is_open and window.projection_delay_pending
    assert window.projection_triggered_at == 102.
    assert len(scheduled) == 1 and scheduled[0][0] == 150


def test_queued_release_delay_uses_release_time_not_press_or_gui_time(window, source, monkeypatch):
    install(window, source, template=True)
    window.map_state.reset()
    window.settings['map_capture_delay_ms'] = 500
    clock = [100.]
    monkeypatch.setattr(app.time, 'perf_counter', lambda: clock[0])
    scheduled = []
    monkeypatch.setattr(window, 'after', lambda delay, callback: scheduled.append((delay, callback)))
    record(source, True)
    clock[0] = 102.
    record(source, False)
    clock[0] = 102.125
    window._handle_map_key()
    assert window.projection_triggered_at == 102.
    assert len(scheduled) == 1 and scheduled[0][0] == 375


@pytest.mark.parametrize('delay', [0, 150, 500])
def test_registration_request_waits_until_release_deadline(window, source, monkeypatch, delay):
    install(window, source, template=True)
    window.map_state.reset()
    window.settings.update(map_capture_delay_ms=delay, auto_project=True)
    clock = [100.]
    requests, scheduled = [], []
    monkeypatch.setattr(app.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(window, 'after', lambda ms, callback: scheduled.append((ms, callback)))
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: requests.append(clock[0]))
    record(source, True)
    window._handle_map_key()
    clock[0] = 102.
    window._handle_map_key()
    assert not scheduled and not requests
    record(source, False)
    window._handle_map_key()
    assert len(scheduled) == 1
    if delay:
        clock[0] = 102. + delay/1000 - .025
        scheduled.pop(0)[1]()  # Even an early Qt timer may not capture early.
        assert not requests and scheduled
    clock[0] = 102. + delay/1000
    scheduled.pop(0)[1]()
    assert requests == [clock[0]]


@pytest.mark.parametrize('cancel', ['outside', 'other_window', 'escape', 'settings', 'reset'])
def test_cancelled_press_cannot_open_on_later_release(window, source, monkeypatch, cancel):
    from dataclasses import replace
    install(window, source, template=True)
    window.map_state.reset()
    scheduled = []
    monkeypatch.setattr(window, 'after', lambda *args: scheduled.append(args))
    record(source, True)
    window._handle_map_key()
    if cancel == 'escape':
        record(source, True, vk=0x1B)
        window._handle_map_key()
    elif cancel == 'settings':
        source.configure('G', ['IdentityV'], suspended=True)
    elif cancel == 'reset':
        window.reset_map_state()
    record(source, False, title='Other window' if cancel == 'outside' else 'IdentityV')
    if cancel == 'other_window':
        event = source.events.get_nowait()
        source.events.put(replace(event, foreground_hwnd=456))
    window._handle_map_key()
    assert not window.map_state.is_open and not window.projection_delay_pending and not scheduled


def test_close_release_does_not_reopen_or_schedule_capture(window, source, monkeypatch):
    install(window, source, template=True)
    scheduled = []
    monkeypatch.setattr(window, 'after', lambda *args: scheduled.append(args))
    assert window.map_state.is_open
    record(source, True)
    window._handle_map_key()
    assert not window.map_state.is_open
    generation = window.map_generation
    record(source, False)
    window._handle_map_key()
    assert not window.map_state.is_open and not scheduled and window.map_generation == generation
