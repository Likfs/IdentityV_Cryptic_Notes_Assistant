"""A freshly started GUI offers newly synchronized difficulties and route versions."""
import json

import numpy as np
import pytest

from src.asset_sync import synchronize
from src import route_projection_app as app
from tests.support.aligned_runtime import activate, difficulty, publish, write_png


@pytest.mark.parametrize('size', [(1080,680), (1280,800)])
def test_new_difficulty_and_route_appear_on_startup(qt_app, tmp_path, monkeypatch, size):
    root = tmp_path/'package'
    image = np.full((80, 100, 3), 100, np.uint8)
    definitions = [difficulty(identifier, name, ('1st',), {'clean': '底图', 'old_route': '原有路线'})
                   for identifier, name in (('hard', '困难'), ('nightmare', '噩梦'))]
    for definition in definitions:
        for version in definition['asset_versions']:
            write_png(root/version['source_root']/'layout_01/layout_01_1st.png', image)
    publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女', difficulties=definitions)]))
    activate(root, monkeypatch)
    for identifier in ('novice', 'easy', 'normal'):
        for directory in ('clean', 'clean-with-route/我的单人路线'):
            write_png(root/f'assets/maps/a_bringer_of_doom/{identifier}/{directory}/layout_01/layout_01_1st.png', image)
    synchronize(root)
    settings = tmp_path/'personal.json'
    settings.write_text(json.dumps(dict(map_id='a_bringer_of_doom', difficulty_id='novice')))
    before = settings.read_bytes()
    window = app.RouteProjectionApp(settings_path=settings)
    try:
        qt_app.processEvents()
        assert window.profile.difficulty_id == 'novice'
        assert window.difficulty_combo.currentText() == '新手'
        window.resize(*size)
        window.show()
        qt_app.processEvents()
        buttons = window.difficulty_selector.buttons
        assert [button.text() for button in buttons] == ['新手', '简单', '普通', '困难', '噩梦']
        assert len({button.y() for button in buttons[:3]}) == 1
        assert buttons[3].y() == buttons[4].y() > buttons[0].y()
        assert buttons[3].x() == buttons[0].x() and buttons[4].x() == buttons[2].x()
        assert len({button.width() for button in buttons}) <= 2
        for button in buttons:
            assert button.width() >= button.fontMetrics().horizontalAdvance(button.text()) + 16
            assert button.height() >= 36
            assert window.difficulty_selector.rect().contains(button.geometry())
        assert window.route_combo.currentData() == '我的单人路线'
        assert window.route_combo.currentText() == '我的单人路线'
        assert window.route_combo.itemText(window.route_combo.findData('clean')) == '原版干净底图'
        window.settings['template_route_versions'].pop(window.profile.key)
        window._update_assets()
        assert window.route_combo.currentData() == '我的单人路线'
        assert settings.read_bytes() == before
        assert window.grab().save(str(tmp_path/'automatic_difficulty.png'))
        for name, identifier in [('新手','novice'),('简单','easy'),('普通','normal'),('困难','hard'),('噩梦','nightmare')]:
            next(button for button in window.difficulty_selector.buttons if button.text()==name).click()
            assert window.profile.difficulty_id == identifier
            assert next(button for button in window.difficulty_selector.buttons if button.text()==name).isChecked()
    finally:
        window.close()
        qt_app.processEvents()


def test_removed_route_preference_falls_back_without_writing_personal_settings(qt_app, tmp_path, monkeypatch):
    root = tmp_path/'package'
    image = np.full((80, 100, 3), 100, np.uint8)
    hard = difficulty('hard', '困难', ('1st',), {'clean': '底图', 'old_route': '旧版本'})
    write_png(root/hard['clean_root']/'layout_01/layout_01_1st.png', image)
    write_png(root/'assets/maps/a_bringer_of_doom/hard/clean-with-route/current/layout_01/layout_01_1st.png', image)
    # The catalog can still declare a removed path before startup synchronization.
    hard['asset_versions'] = [hard['asset_versions'][0]]
    hard['template_versions'] = ['clean']
    publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女', difficulties=[hard])]))
    config = root/'configs/map_catalog.json'
    raw = json.loads(config.read_text(encoding='utf-8'))
    entry = raw['maps'][0]['difficulties'][0]
    entry['asset_versions'].append(dict(id='old_route', name='旧版本', kind='route', source_root='removed/path'))
    entry['template_versions'].append('old_route')
    config.write_text(json.dumps(raw), encoding='utf-8')
    activate(root, monkeypatch)
    synchronize(root)
    settings = tmp_path/'personal.json'
    settings.write_text(json.dumps(dict(map_id='a_bringer_of_doom', difficulty_id='hard', opacity=.61,
        template_route_versions={'a_bringer_of_doom/hard': 'old_route'})), encoding='utf-8')
    before = settings.read_bytes()
    window = app.RouteProjectionApp(settings_path=settings)
    try:
        qt_app.processEvents()
        assert window.route_combo.currentData() == 'current'
        assert window.settings['opacity'] == .61
        assert settings.read_bytes() == before
    finally:
        window.close()
        qt_app.processEvents()


def test_reference_floor_buttons_follow_the_locked_layout(qt_app, tmp_path, monkeypatch):
    root = tmp_path/'package'
    image = np.full((80, 100, 3), 100, np.uint8)
    hard = difficulty('hard', '困难', ('1st',), {'clean': '底图', 'old_route': '原有路线'})
    for version in hard['asset_versions']:
        write_png(root/version['source_root']/'layout_01/layout_01_1st.png', image)
    publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女', difficulties=[hard])]))
    for layout, floors in [('layout_01', ['1st']), ('layout_02', ['1st', '2nd'])]:
        for floor in floors:
            for source in ('clean', 'clean-with-route/current'):
                write_png(root/f'assets/maps/a_bringer_of_doom/easy/{source}/{layout}/{layout}_{floor}.png', image)
    activate(root, monkeypatch)
    synchronize(root)
    settings = tmp_path/'personal.json'
    settings.write_text(json.dumps(dict(map_id='a_bringer_of_doom', difficulty_id='easy', reference_floor='2nd')))
    before = settings.read_bytes()
    window = app.RouteProjectionApp(settings_path=settings)
    try:
        window._lock_recognized_layout('layout_01', automatic=True)
        qt_app.processEvents()
        assert [window.floor_combo.itemData(i) for i in range(window.floor_combo.count())] == ['1st']
        assert window.settings['reference_floor'] == '1st'
        window._next_reference_floor()
        assert window.settings['reference_floor'] == '1st'
        window._lock_recognized_layout('layout_02', automatic=True)
        assert {window.floor_combo.itemData(i) for i in range(window.floor_combo.count())} == {'1st', '2nd'}
        assert settings.read_bytes() == before
    finally:
        window.close()
        qt_app.processEvents()
