import ctypes
from types import SimpleNamespace

import pytest
from src import route_projection_app as app


@pytest.mark.parametrize('change,valid', [
    ('none', True), ('missing', False), ('minimized', False),
    ('title', False), ('moved', False), ('resized', False),
    ('unrelated_foreground', False), ('client_error', False),
])
def test_proactive_confirmation_validates_original_game_while_app_is_foreground(monkeypatch, change, valid):
    def get_title(_hwnd, buffer, _size):
        buffer.value = 'other' if change == 'title' else 'IdentityV'
        return len(buffer.value)

    def get_rect(_hwnd, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.RECT)).contents
        rect.right, rect.bottom = (900 if change == 'resized' else 800), 600
        return change != 'client_error'

    def get_origin(_hwnd, pointer):
        origin = ctypes.cast(pointer, ctypes.POINTER(ctypes.wintypes.POINT)).contents
        origin.x, origin.y = (200 if change == 'moved' else 100), 100
        return True

    native = SimpleNamespace(IsWindow=lambda _: change != 'missing', IsIconic=lambda _: change == 'minimized',
        GetWindowTextW=get_title, GetClientRect=get_rect, ClientToScreen=get_origin)
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=native), raising=False)
    monkeypatch.setattr(app, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: False)
    monkeypatch.setattr(app, 'own_window_is_foreground', lambda: change != 'unrelated_foreground')
    identity = [123, 100, 100, 800, 600]
    assert app.recognition_window_matches(identity, ['IdentityV'], allow_app=True) is valid
    assert not app.recognition_window_matches(identity, ['IdentityV'])
