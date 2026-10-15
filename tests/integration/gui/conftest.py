from tests.support.template_inputs import install_floor_inputs
from types import SimpleNamespace
from queue import Queue
import pytest
from PyQt6.QtWidgets import QApplication
from src import route_projection_app as app
from src import route_projection_core as core
from src.capture import runtime as capture_runtime
import numpy as np
from src import route_projection_qt_ui as ui
from pathlib import Path
from src.registration.template_matching import direct
from src.capture.map_keys import GlobalMapKeys
from src.capture.mouse_pan import MousePanHook


@pytest.fixture(autouse=True)
def no_live_input(monkeypatch):
    """GUI tests use recorded input; never install operating-system listeners."""
    monkeypatch.setattr(app, 'GlobalMapKeys', lambda *_args: None)
    monkeypatch.setattr(app, 'GlobalActionHotkeys', lambda *_args: SimpleNamespace(
        events=Queue(), error=None, start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(MousePanHook, 'start', lambda self: None)
    monkeypatch.setattr(app.RouteProjectionApp, '_start_layout_retrieval', lambda self: None)
    # Dedicated prewarm tests exercise the real coordinator; ordinary GUI tests stay isolated.
    monkeypatch.setattr(app.RouteProjectionApp, '_prepare_template_sources', lambda self: None)


@pytest.fixture(autouse=True)
def no_live_capture(monkeypatch):
    """Require explicit screenshot fakes, including calls caught by worker threads."""
    attempts = []

    def blocked(*_args, **_kwargs):
        attempts.append('unmocked screen capture')
        raise RuntimeError('GUI tests must supply a fake screenshot')

    # Cover imported aliases as well as the underlying capture service.
    for module in (app, core):
        monkeypatch.setattr(module, 'capture_bgr', blocked)
    monkeypatch.setattr(core, 'capture_region', blocked)
    monkeypatch.setattr(capture_runtime, 'capture_region', blocked)
    monkeypatch.setattr(capture_runtime.mss, 'MSS', blocked)
    yield
    assert not attempts, 'GUI test attempted real screen capture; mock the capture or work request'


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(app, "exclude_window_from_capture", lambda _: False)
    monkeypatch.setattr(app, "game_window_is_foreground", lambda _: True)
    monkeypatch.setattr(app, "own_window_is_foreground", lambda: False)
    monkeypatch.setattr(app, "map_key_down", lambda _: False)
    monkeypatch.setattr(app.RouteProjectionApp, "after", lambda *_: None)
    qt = QApplication.instance() or QApplication([])
    # Ordinary windows start with current defaults; migration has dedicated tests.
    (tmp_path / "settings.json").write_text('{}', encoding="utf-8")
    instance = app.RouteProjectionApp(settings_path=tmp_path / "settings.json")
    instance.locked_layout = "layout_11"
    instance.layout_var.set("layout_11")
    instance.map_state.is_open = True
    instance._update_assets()
    yield instance
    instance.close()
    qt.processEvents()


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui.RouteOverlayWindow, "_prepare_native_window", lambda self: None)
    monkeypatch.setattr(ui, "physical_to_qt_geometry", lambda x, y, w, h: (x, y, w, h, 1.0))
    yield app
    app.processEvents()


@pytest.fixture
def direct_inputs(monkeypatch):
    original=np.full((40,40,3),120,np.uint8)
    rgba=np.dstack((original,np.full((40,40),180,np.uint8)))
    entry={'saved_scale':1.,'source_sha256':'verified','source_image':'original.png'}
    install_floor_inputs(monkeypatch, entry, original, rgba)
    monkeypatch.setattr(direct,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(direct.matcher,'_match_template',lambda *a, **kw:dict(score=2.,location=(0,0)))
    return original


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(GlobalMapKeys, '_read_held', staticmethod(lambda _keys: set()))
    return GlobalMapKeys('G', ['IdentityV'], lambda: None)
