from tests.support.qt_wait import wait_for
import numpy as np
import pytest
from src.registration.template_matching import direct


@pytest.mark.parametrize('variant', ['route_lhp_solo_clean', 'route_zhanshi_fast_clean'])
def test_reference_uses_selected_source_version(window, monkeypatch, variant):
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: None)
    window.apply_settings({**window.settings, 'registration_backend': 'template'})
    window.route_combo.setCurrentIndex(window.route_combo.findData(variant))
    seen = []
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *a: seen.append(a))
    window.floor_combo.setCurrentIndex(window.floor_combo.findData('b1'))
    window.reference_visible = True
    window._refresh_reference_map()
    wait_for(lambda: bool(seen))
    assert seen and '底图版' in seen[-1][2]
    expected = direct.floors.load(window.profile.key, window.locked_layout, variant, 'b1')[3]
    np.testing.assert_array_equal(seen[-1][0], expected)
