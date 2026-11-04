import pytest
import ctypes
from types import SimpleNamespace


def test_screen_queries_reuse_ctypes_types_and_preserve_monitor_selection(monkeypatch):
    from src.registration.template_matching import screen
    from src.capture import region

    def monitor_info(handle, pointer):
        info = pointer._obj
        width, height = (2560, 1440) if handle == 11 else (1920, 1080)
        info.monitor.right, info.monitor.bottom = width, height
        info.flags = int(handle == 11)
        info.device = f'DISPLAY{handle}'
        return True

    def enumerate_monitors(_dc, _clip, callback, parameter):
        assert callback(11, 0, None, parameter)
        assert callback(22, 0, None, parameter)
        return True

    user = SimpleNamespace(GetMonitorInfoW=monitor_info, EnumDisplayMonitors=enumerate_monitors)
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=user))
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

    user = SimpleNamespace(MonitorFromPoint=monitor_from_point, GetMonitorInfoW=monitor_info)
    screen = SimpleNamespace(name=lambda: 'DISPLAY2', devicePixelRatio=lambda: 1.5,
                             geometry=lambda: SimpleNamespace(x=lambda: 1280, y=lambda: 40))
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=user))
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
