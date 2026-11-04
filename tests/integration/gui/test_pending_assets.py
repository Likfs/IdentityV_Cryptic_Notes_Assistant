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
