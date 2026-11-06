import json
from src import route_projection_app as app
from tests.support.aligned_runtime import activate, build_gui_package, publish


def test_placeholder_switch_and_actions_are_safe(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: True)
    root = build_gui_package(tmp_path/'package')
    catalog = json.loads((root/'configs/map_catalog.json').read_bytes())
    catalog['maps'].append(dict(id='pending_map', name='磷光症', difficulties=[dict(
        id='hard', name='困难', asset_contract='aligned_floor_images', asset_status='pending', floors=[],
        template_versions=[], asset_versions=[])]))
    publish(root, catalog)
    activate(root, monkeypatch)
    window = app.RouteProjectionApp(settings_path=tmp_path/'settings.json')
    try:
        window.map_combo.setCurrentText('磷光症')
        window.difficulty_combo.setCurrentText('困难')
        assert window.profile.key == 'pending_map/hard'
        assert window.profile.floors == ()
        assert window.profile.clean_root is None and not window.profile.template_versions
        assert window.floor_combo.count() == 0
        assert not window.route_combo.isEnabled()
        assert '待配置' in window.asset_var.get()
        assert '待配置' in window.session_label.text()
        assert not window.start_button.isEnabled()
        window.request_work('recognize', auto=False)
        window.request_work('project', auto=False)
        assert window.locked_layout is None
        window._next_reference_floor()
        window._toggle_reference_map()
        assert window.toast.title.text() == '布局尚未选定'
        window.map_combo.setCurrentText('厄运之女')
        for name, difficulty in (('噩梦', 'nightmare'), ('困难', 'hard')):
            window.difficulty_combo.setCurrentText(name)
            profile = window.catalog.profile('a_bringer_of_doom', difficulty)
            assert window.profile == profile
            assert profile.asset_status == 'ready'
            assert [window.floor_combo.itemData(i) for i in range(window.floor_combo.count())] == list(profile.floors)
            assert window.route_combo.isEnabled()
            assert window.start_button.isEnabled()
            assert window.route_combo.count() == len(profile.template_versions)
            assert {window.route_combo.itemData(i) for i in range(window.route_combo.count())} == set(profile.template_versions)
            expected_layouts = sorted(path.name for path in profile.clean_root.glob('layout_*') if path.is_dir())
            assert expected_layouts
            assert [window.layout_combo.itemText(i) for i in range(1, window.layout_combo.count())] == expected_layouts
        window.show()
        qt_app.processEvents()
        assert window.grab().save(str(tmp_path/'pending_window.png'))
    finally:
        window.close()
        qt_app.processEvents()


def test_clean_only_profile_enables_gui_and_reference_without_a_route(qt_app, tmp_path, monkeypatch):
    import shutil
    from src.asset_sync import synchronize
    from tests.support.aligned_runtime import build_multifloor_package
    from tests.support.qt_wait import wait_for
    root = build_multifloor_package(tmp_path/'clean_only')
    shutil.rmtree(root/'assets/maps/phosphorescence_disease/hard/clean-with-route')
    synchronize(root)
    activate(root, monkeypatch)
    monkeypatch.setattr(app, 'game_window_is_foreground', lambda _: True)
    monkeypatch.setattr(app, 'own_window_is_foreground', lambda: False)
    settings_path = tmp_path/'settings.json'
    raw = json.dumps(dict(map_id='phosphorescence_disease', difficulty_id='hard', reference_floor='b4',
                          template_route_versions={'phosphorescence_disease/hard':'测试路线'})).encode('utf-8')
    settings_path.write_bytes(raw)
    window = app.RouteProjectionApp(settings_path=settings_path)
    try:
        assert window.profile.asset_status == 'ready' and window.start_button.isEnabled()
        assert window.route_combo.count() == 1 and window.route_combo.currentData() == 'clean'
        assert window.route_combo.currentText() == '原版干净底图'
        window.locked_layout = 'layout_01'
        window.layout_var.set('layout_01')
        window._update_assets()
        assert window.project_button.isEnabled() and settings_path.read_bytes() == raw
        window._toggle_reference_map()
        wait_for(lambda: window.reference_overlay.visible)
        assert '地下4层' in window.reference_overlay.caption.text()
        assert '原版干净底图' in window.reference_overlay.caption.text()
        window._next_reference_floor()
        wait_for(lambda: window.reference_overlay.visible and '地下1层' in window.reference_overlay.caption.text())
        window.show()
        qt_app.processEvents()
        assert window.grab().save(str(tmp_path/'clean_only.png'))
    finally:
        window.close()
        qt_app.processEvents()
