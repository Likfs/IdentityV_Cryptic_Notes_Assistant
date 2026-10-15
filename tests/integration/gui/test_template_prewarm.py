from types import SimpleNamespace
from src import route_projection_app as app

PREPARE=app.RouteProjectionApp._prepare_template_sources


def install(window,monkeypatch):
    calls=[]
    window._template_prewarmer.close()
    monkeypatch.setattr(window,'_template_prewarmer',SimpleNamespace(
        request=lambda key:calls.append(key),close=lambda:calls.append('closed')))
    monkeypatch.setattr(window,'_prepare_template_sources',PREPARE.__get__(window))
    monkeypatch.setattr(window,'request_work',lambda *a,**kw:None)
    return calls


def test_confirmed_layout_and_version_prepare_without_showing_projection(window,monkeypatch):
    calls=install(window,monkeypatch)
    window.map_state.is_open=False
    window.clear_layout_lock()
    assert calls[-1] is None
    monkeypatch.setattr(app.messagebox,'askyesno',lambda *a,**kw:True)
    window.layout_var.set('layout_12');window._layout_changed()
    assert calls[-1][1:3]==(window.profile.key,'layout_12')
    window.route_combo.setCurrentIndex(window.route_combo.findData('route_lhp_solo'))
    assert calls[-1][-1]=='route_lhp_solo'
    assert window.last_result is None and not window.overlay.visible
    window._shutdown()
    assert calls[-1]=='closed'


def test_reset_and_profile_change_cancel_old_template_preparation(window,monkeypatch):
    calls=install(window,monkeypatch)
    window._update_assets()
    old=calls[-1]
    window.reset_map_state()
    assert calls[-1] is None and window.locked_layout is None
    window.locked_layout='layout_12';window.layout_var.set('layout_12')
    window._update_assets()
    assert calls[-1]!=old
    window._profile_changed()
    assert calls[-1] is None
