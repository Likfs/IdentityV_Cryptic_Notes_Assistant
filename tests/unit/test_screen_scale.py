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


@pytest.mark.parametrize('interruption', ['coordinate_query', 'monitor_query'])
def test_monitor_queries_keep_types_when_calls_interleave(monkeypatch, interruption):
    """Use native ctypes checks: Python function fakes cannot detect pointer conflicts."""
    from src import route_projection_qt_ui as ui
    from src.capture.monitors import monitor_api, MonitorInfo, MonitorEnumProc
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
    def query_monitors():
        sizes = []
        api = monitor_api()
        @MonitorEnumProc
        def visit(handle, _dc, _rect, _parameter):
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(info)
            assert api.GetMonitorInfoW(handle, ctypes.byref(info))
            sizes.append((info.rcMonitor.right-info.rcMonitor.left,
                          info.rcMonitor.bottom-info.rcMonitor.top))
            return True
        assert api.EnumDisplayMonitors(None, None, visit, 0)
        assert sizes and all(width > 0 and height > 0 for width, height in sizes)
    if interruption == 'coordinate_query':
        def interleaved_point(value, flags):
            handle = point(value, flags)
            query_monitors()
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
            query_monitors()
    assert len(interrupted) == 10


def test_capture_scale_uses_supplied_client_without_querying_desktop(monkeypatch):
    from src.registration.template_matching import screen
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *_: pytest.fail('Scale queried desktop dimensions'))
    assert screen.capture_info((2560, 1440))['factor'] == 1.
    count = len(ctypes._pointer_type_cache)
    for _ in range(200):
        expected = dict(
            source='capture_game_client', width=1920, height=1080,
            reference_size=[2560, 1440], factor=pytest.approx(4/3))
        actual = screen.capture_info((1920, 1080))
        assert {key:actual[key] for key in expected} == expected
        assert actual['geometry_model'] == 'fixed_width'
        assert actual['visible_design_size'] == [1334, 751]
    assert len(ctypes._pointer_type_cache) == count


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


def test_capture_factor_matches_reference_convention():
    from src.registration.template_matching.screen import factor_for_size
    assert factor_for_size(2560,1440)==1
    assert factor_for_size(1920,1080)==pytest.approx(4/3)
    assert factor_for_size(3840,2160)==pytest.approx(2/3)
    assert factor_for_size(3072,1920)==pytest.approx(5/6)
    assert factor_for_size(3440,1440)==pytest.approx(1000/1000.5)
    assert factor_for_size(2560,1334)==pytest.approx(1920000/1779556)


@pytest.mark.parametrize('size', [None, (), (1920,), (1920,1080,1), (0,1080),
    (1920,-1), (True,1080), (1920.,1080), ('1920',1080), (float('inf'),1080)])
def test_invalid_capture_size_cannot_fall_back_to_desktop(size):
    from src.registration.template_matching.screen import capture_info
    with pytest.raises(ValueError):
        capture_info(size)
