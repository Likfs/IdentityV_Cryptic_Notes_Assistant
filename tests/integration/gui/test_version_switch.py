from tests.support.qt_wait import wait_for
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src import route_projection_app as app
from src.app_catalog import load_settings
from src.registration.template_matching import direct


@pytest.mark.parametrize('state', ['open', 'closed', 'background', 'busy'])
@pytest.mark.parametrize('variant_id', ['route_lhp_solo', 'route_clean', 'route_lhp_solo_clean', 'route_zhanshi_fast_clean'])
def test_qt_version_switch_clears_old_result_and_rematches_only_current_state(window, monkeypatch, state, variant_id):
    calls = []
    monkeypatch.setattr(window, 'request_work', lambda mode, **kw: calls.append((mode, kw)))
    versions = json.loads(json.dumps(window.settings['route_versions']))
    window.apply_settings({**window.settings, 'registration_backend': 'template'})
    assert [window.route_combo.itemData(i) for i in range(window.route_combo.count())] == [
        'route_zhanshi_fast', 'route_lhp_solo', 'route_clean', 'route_lhp_solo_clean', 'route_zhanshi_fast_clean']
    generation = window.map_generation
    window.last_result = object()
    window.settings['auto_project'] = False  # Explicit version changes still rematch.
    window.map_state.is_open = state != 'closed'
    window.busy = state == 'busy'
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: state != 'background')
    window.route_combo.setCurrentIndex(window.route_combo.findData(variant_id))
    assert window.map_generation == generation+1 and window.last_result is None and window.live_projection is None
    assert window.settings['route_versions'] == versions
    assert load_settings(window.catalog, window.settings_path)['template_route_versions'][window.profile.key] == variant_id
    assert bool(calls) == (state == 'open')
    if state == 'busy':
        window.busy = False
        window._drain_pending_projection()
        assert calls == [('project', {'auto': False})]
    old = SimpleNamespace(accepted=True, floor='all', layer=np.ones((5, 5, 4), np.uint8))
    window._apply_result('project', True, generation, [0, 0, 5, 5], old, None)
    assert window.last_result is None


def test_reference_and_late_preview_follow_selected_original(window, monkeypatch):
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: None)
    window.apply_settings({**window.settings, 'registration_backend': 'template'})
    ready = SimpleNamespace()
    window.last_result = ready
    generation, session = window.map_generation, window.session_id
    window.route_combo.setCurrentIndex(window.route_combo.findData('route_lhp_solo'))
    monkeypatch.setattr(window, '_set_preview_kind', lambda *a: pytest.fail('late preview'))
    window._receive_work_event(('preview', generation, session, ready, SimpleNamespace()))
    assert window.last_result is None
    seen = []
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: seen.append(a))
    window.floor_combo.setCurrentIndex(window.floor_combo.findData('b1'))
    window.reference_visible = True
    window._refresh_reference_map()
    wait_for(lambda: bool(seen))
    assert seen and '凉哈皮' in seen[-1][2]
    assert seen[-1][0].shape == direct.floors.load(window.profile.key, window.locked_layout, 'route_lhp_solo', 'b1')[3].shape
