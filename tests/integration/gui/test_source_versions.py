from tests.support.qt_wait import wait_for
import numpy as np
import pytest
from src.registration.template_matching import direct


def test_partial_route_fallback_keeps_preference_and_reference_identity(qt_app, tmp_path, monkeypatch):
    import json
    import shutil
    from src import asset_sync, route_projection_app as app
    from tests.support.aligned_runtime import activate, build_multifloor_package

    root = build_multifloor_package(tmp_path/'package', layout_floors={
        'layout_01': ('1st',), 'layout_02': ('b1', 'b2', 'b3')})
    shutil.rmtree(root/'assets/maps/phosphorescence_disease/hard/clean-with-route/测试路线/layout_02')
    asset_sync.synchronize(root)
    activate(root, monkeypatch)
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: True)
    monkeypatch.setattr(app, 'own_window_is_foreground', lambda: False)
    monkeypatch.setattr(app, 'exclude_window_from_capture', lambda _: False)
    monkeypatch.setattr(app, 'map_key_down', lambda _: False)
    monkeypatch.setattr(app.RouteProjectionApp, 'after', lambda *_: None)
    personal = tmp_path/'personal.json'
    personal.write_text(json.dumps(dict(map_id='phosphorescence_disease', difficulty_id='hard',
        template_route_versions={'phosphorescence_disease/hard': '测试路线'})), encoding='utf-8')
    before = personal.read_bytes()
    window = app.RouteProjectionApp(settings_path=personal)
    monkeypatch.setattr(window, 'request_work', lambda *_a, **_kw: None)
    try:
        assert window.route_combo.currentText() == '测试路线（1/2 布局）'
        window.locked_layout = 'layout_01'
        window.layout_var.set('layout_01')
        window._update_assets()
        old_key = window._reference_image_key()
        assert window.route_combo.currentData() == '测试路线'
        window.locked_layout = 'layout_02'
        window.layout_var.set('layout_02')
        window._update_assets()
        assert window.route_combo.count() == 1 and window.route_combo.currentData() == 'clean'
        assert '已保留路线偏好' in window.asset_var.get()
        assert window.settings['template_route_versions'][window.profile.key] == '测试路线'
        assert personal.read_bytes() == before
        assert window._recognition_context()[-1] == 'clean'
        assert window._reference_image_key()[3] == 'clean'
        assert window._motion._key()[4] == 'clean'
        window._receive_work_event(('floor_reference', old_key, np.ones((4,4,3), np.uint8), None))
        assert window._floor_reference_image is None
        seen = []
        monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *args: seen.append(args))
        window.reference_visible = True
        window._refresh_reference_map()
        wait_for(lambda: bool(seen))
        expected = direct.floors.load(window.profile.key, 'layout_02', 'clean', 'b1')[3]
        np.testing.assert_array_equal(seen[-1][0], expected)
        assert '原版干净底图' in seen[-1][2]
        window.resize(1080, 680)
        window.show()
        qt_app.processEvents()
        assert window.route_availability_hint.isVisible()
        assert '已使用干净底图' in window.route_availability_hint.text()
        assert window.grab().save(str(tmp_path/'partial_route_fallback.png'))
        window.locked_layout = 'layout_01'
        window.layout_var.set('layout_01')
        window._update_assets()
        assert window.route_combo.currentData() == '测试路线'
        assert window.route_availability_hint.isHidden()
        assert window._reference_image_key()[3] == '测试路线'
        assert window._motion._key()[4] == '测试路线'
        assert personal.read_bytes() == before
    finally:
        window.close()
        direct.floors._close_fixed_session()
        qt_app.processEvents()


@pytest.mark.parametrize('variant', ['route_custom_a', 'route_custom_b'])
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
    assert seen and window.profile.version_names[variant] in seen[-1][2]
    expected = direct.floors.load(window.profile.key, window.locked_layout, variant, 'b1')[3]
    np.testing.assert_array_equal(seen[-1][0], expected)
