import pytest
import ctypes
from ctypes import wintypes
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def isolated_monitor_binding(monkeypatch):
    from src.capture import monitors
    monkeypatch.setattr(monitors, '_api', None)


def test_monitor_binding_initializes_once_for_concurrent_first_queries(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event
    from src.capture import monitors

    user = ctypes.WinDLL('user32')
    loaded = []
    start = Barrier(8)
    def load(name):
        assert name == 'user32'
        loaded.append(name)
        Event().wait(.02)  # Allow competing first queries to reach initialization.
        return user
    monkeypatch.setattr(ctypes, 'WinDLL', load)
    def query(_index):
        start.wait(timeout=3)
        api = monitors.monitor_api()
        handle = api.MonitorFromPoint(wintypes.POINT(50, 50), 2)
        info = monitors.MonitorInfo()
        info.cbSize = ctypes.sizeof(monitors.MonitorInfo)
        assert handle and api.GetMonitorInfoW(handle, ctypes.byref(info))
        assert info.rcMonitor.right > info.rcMonitor.left
        return api
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(query, range(8)))
    assert loaded == ['user32']
    assert all(api is results[0] for api in results)
    assert results[0].GetMonitorInfoW.argtypes[1] is ctypes.POINTER(monitors.MonitorInfo)


@pytest.mark.parametrize('interruption', ['coordinate_query', 'registration_query'])
def test_monitor_queries_keep_types_when_calls_interleave(monkeypatch, interruption):
    """Use native ctypes checks: Python function fakes cannot detect pointer conflicts."""
    from src import route_projection_qt_ui as ui
    from src.registration.template_matching import screen
    from src.capture import region

    user = ctypes.WinDLL('user32')
    point = user.MonitorFromPoint
    point.argtypes = (wintypes.POINT, wintypes.DWORD)
    point.restype = wintypes.HANDLE
    enumerate_monitors = user.EnumDisplayMonitors
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                  ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    enumerate_monitors.argtypes = (wintypes.HDC, ctypes.POINTER(wintypes.RECT), callback, wintypes.LPARAM)
    enumerate_monitors.restype = wintypes.BOOL
    # Both the previous windll callers and the shared lazy binding reach this DLL.
    monkeypatch.setattr(ctypes.windll, 'user32', user)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda _name: user)
    monkeypatch.setattr(region, 'enable_physical_pixel_coordinates', lambda: None)
    qt_screen = SimpleNamespace(devicePixelRatio=lambda: 1.,
                                geometry=lambda: SimpleNamespace(x=lambda: 0, y=lambda: 0))
    monkeypatch.setattr(ui, 'QGuiApplication', SimpleNamespace(
        screens=lambda: [], primaryScreen=lambda: qt_screen))
    interrupted = []
    if interruption == 'coordinate_query':
        def interleaved_point(value, flags):
            handle = point(value, flags)
            assert screen.screen_info()['ok']
            interrupted.append(True)
            return handle
        monkeypatch.setattr(user, 'MonitorFromPoint', interleaved_point)
    else:
        def interleaved_enum(dc, clip, visit, parameter):
            assert ui.physical_to_qt_geometry(0, 0, 100, 100)[2:] == (100, 100, 1.)
            interrupted.append(True)
            return enumerate_monitors(dc, clip, visit, parameter)
        monkeypatch.setattr(user, 'EnumDisplayMonitors', interleaved_enum)
    for _ in range(10):
        if interruption == 'coordinate_query':
            assert ui.physical_to_qt_geometry(0, 0, 100, 100)[2:] == (100, 100, 1.)
        else:
            result = screen.screen_info()
            assert result['ok'] and result['width'] > 0 and result['height'] > 0
    assert len(interrupted) == 10


def test_screen_queries_reuse_ctypes_types_and_preserve_monitor_selection(monkeypatch):
    from src.registration.template_matching import screen
    from src.capture import region

    def monitor_info(handle, pointer):
        info = pointer._obj
        width, height = (2560, 1440) if handle == 11 else (1920, 1080)
        info.rcMonitor.right, info.rcMonitor.bottom = width, height
        info.dwFlags = int(handle == 11)
        info.szDevice = f'DISPLAY{handle}'
        return True

    def enumerate_monitors(_dc, _clip, callback, parameter):
        assert callback(11, 0, None, parameter)
        assert callback(22, 0, None, parameter)
        return True

    user = SimpleNamespace(GetMonitorInfoW=monitor_info, EnumDisplayMonitors=enumerate_monitors,
                           MonitorFromPoint=lambda *_: 11)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda _name: user)
    monkeypatch.setattr(region, 'enable_physical_pixel_coordinates', lambda: None)
    assert screen.screen_info() == dict(ok=True, screen_name='DISPLAY11', width=2560,
                                       height=1440, primary=True, factor=1.)
    count = len(ctypes._pointer_type_cache)
    for _ in range(200):
        selected = screen.screen_info('DISPLAY22')
        assert selected == dict(ok=True, screen_name='DISPLAY22', width=1920,
                                height=1080, primary=False, factor=pytest.approx(4/3))
    assert len(ctypes._pointer_type_cache) == count
    assert screen.screen_info('missing') == dict(ok=False, screen_name='missing',
                                                 width=0, height=0, factor=1.)


def test_coordinate_queries_reuse_ctypes_types_and_preserve_dpi_mapping(monkeypatch):
    from src import route_projection_qt_ui as ui

    def monitor_from_point(_point, _flags):
        return 22

    def monitor_info(_handle, pointer):
        info = pointer._obj
        info.rcMonitor.left = 1920
        info.szDevice = r'\\.\DISPLAY2'
        return True

    user = SimpleNamespace(MonitorFromPoint=monitor_from_point, GetMonitorInfoW=monitor_info,
                           EnumDisplayMonitors=lambda *_: False)
    screen = SimpleNamespace(name=lambda: 'DISPLAY2', devicePixelRatio=lambda: 1.5,
                             geometry=lambda: SimpleNamespace(x=lambda: 1280, y=lambda: 40))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda _name: user)
    monkeypatch.setattr(ui, 'QGuiApplication', SimpleNamespace(
        screens=lambda: [screen], primaryScreen=lambda: screen))
    assert ui.physical_to_qt_geometry(2070, 300, 1180, 1050) == (1380, 240, 787, 700, 1.5)
    count = len(ctypes._pointer_type_cache)
    for _ in range(200):
        assert ui.physical_to_qt_geometry(2070, 300, 1180, 1050) == (1380, 240, 787, 700, 1.5)
    assert len(ctypes._pointer_type_cache) == count
    assert ui.physical_to_qt_geometry(1920, 0, 0, 0) == (1280, 40, 1, 1, 1.5)


def test_screen_factor_matches_original_convention():
    from src.registration.template_matching.screen import factor_for_size
    assert factor_for_size(2560,1440)==1
    assert factor_for_size(1920,1080)==pytest.approx(4/3)
    assert factor_for_size(3840,2160)==pytest.approx(2/3)
    assert factor_for_size(0,0)==1
