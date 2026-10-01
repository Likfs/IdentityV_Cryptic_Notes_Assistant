from types import SimpleNamespace

from tools.common import capture_exclusion as exclusion


class NativeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def test_native_exclusion_requires_successful_set_and_readback(monkeypatch):
    monkeypatch.setattr(exclusion.sys, "platform", "win32")
    monkeypatch.setattr(exclusion.sys, "getwindowsversion", lambda: SimpleNamespace(build=19045), raising=False)
    calls = []
    state = {"set_ok": True, "get_ok": True, "affinity": 0x11}
    def setter(hwnd, affinity):
        calls.append((hwnd, affinity))
        return state["set_ok"]
    def getter(hwnd, output):
        output._obj.value = state["affinity"]
        return state["get_ok"]
    api = SimpleNamespace(SetWindowDisplayAffinity=NativeFunction(setter), GetWindowDisplayAffinity=NativeFunction(getter))
    monkeypatch.setattr(exclusion.ctypes, "WinDLL", lambda *_args, **_kwargs: api, raising=False)
    assert exclusion.exclude_native_window(100)
    assert calls == [(100, 0x11)]
    state["affinity"] = 1  # A black placeholder is not exclusion.
    assert not exclusion.exclude_native_window(100)
    state.update(affinity=0x11, get_ok=False)
    assert not exclusion.exclude_native_window(100)
    state.update(get_ok=True, set_ok=False)
    assert not exclusion.exclude_native_window(100)


def test_old_windows_does_not_claim_exclusion(monkeypatch):
    monkeypatch.setattr(exclusion.sys, "platform", "win32")
    monkeypatch.setattr(exclusion.sys, "getwindowsversion", lambda: SimpleNamespace(build=18363), raising=False)
    assert not exclusion.exclude_native_window(100)
