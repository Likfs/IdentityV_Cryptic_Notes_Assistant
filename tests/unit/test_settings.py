from __future__ import annotations
import json
import pytest
from src.app_catalog import MapCatalog, load_settings, save_settings, selected_template_variant
from src.chest_counter import CHEST_HOTKEY_FIELDS


@pytest.fixture
def preference_catalog():
    """Valid historical preferences are independent of the user's current route folders."""
    catalog = MapCatalog()
    data = json.loads(catalog.path.read_text(encoding='utf-8-sig'))
    definition = next(d for m in data['maps'] if m['id'] == 'a_bringer_of_doom'
                      for d in m['difficulties'] if d['id'] == 'nightmare')
    known = {v['id'] for v in definition['asset_versions']}
    for identifier in ('route_zhanshi_fast', 'route_lhp_solo'):
        if identifier not in known:
            definition['asset_versions'].append(dict(id=identifier, name=identifier, kind='route',
                source_root=definition['clean_root']))
        if identifier not in definition['template_versions']:
            definition['template_versions'].append(identifier)
    return MapCatalog(catalog.path, data=data)


def test_combination_preferences_normalize_and_roundtrip_without_changing_single_keys(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'recognize_hotkey': 'ctrl+1', 'project_hotkey': 'shift+ctrl+1',
        'reference_hotkey': '1', 'reference_floor_hotkey': 'alt+j', 'chest_gold_add_hotkey': 'ctrl+shift+f5'}))
    before = path.read_bytes()
    settings = load_settings(MapCatalog(), path)
    assert path.read_bytes() == before
    assert [settings[field] for field in ('recognize_hotkey', 'project_hotkey', 'reference_hotkey',
        'reference_floor_hotkey', 'chest_gold_add_hotkey', 'chest_gold_subtract_hotkey')] == [
        'Ctrl+1', 'Ctrl+Shift+1', '1', 'Alt+J', 'Ctrl+Shift+F5', 'Num5']
    save_settings(settings, path)
    assert load_settings(MapCatalog(), path) == settings


def test_counter_hotkeys_migrate_without_overriding_existing_keys(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'recognize_hotkey': 'F5', 'reference_hotkey': 'F6',
                               'chest_gold_add_hotkey': 'F5', 'chest_purple_add_hotkey': [],
                               'capture_exclusion_enabled': False, 'chest_counter_visible': False}))
    before = path.read_bytes()
    settings = load_settings(MapCatalog(), path)
    assert path.read_bytes() == before
    assert settings['recognize_hotkey'] == 'F5' and settings['reference_hotkey'] == 'F6'
    fields = ('map_key', 'recognize_hotkey', 'project_hotkey', 'reference_hotkey', 'reference_floor_hotkey', *CHEST_HOTKEY_FIELDS)
    assert len({settings[field] for field in fields}) == len(fields)
    assert not settings['capture_exclusion_enabled'] and not settings['chest_counter_visible']
    save_settings(settings, path)
    assert load_settings(MapCatalog(), path) == settings


@pytest.mark.parametrize('old,expected', [
    ({}, 'Num0'),
    ({'next_round_hotkey': 'ctrl+numEnter'}, 'Ctrl+NumEnter'),
    ({'next_round_hotkey': []}, 'Num0'),
    ({'recognize_hotkey': 'Num0'}, 'Ctrl+Num0'),
    ({'reference_floor_hotkey': 'Num0'}, 'Ctrl+Num0'),
    ({'map_key': 'Num0'}, 'Ctrl+N'),
    ({'next_round_hotkey': 'G'}, 'Num0'),
])
def test_next_round_hotkey_migrates_after_existing_bindings_without_writing(tmp_path, old, expected):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(old), encoding='utf-8')
    before = path.read_bytes()
    settings = load_settings(MapCatalog(), path)
    assert path.read_bytes() == before
    assert settings['next_round_hotkey'] == expected
    for field in ('map_key', 'recognize_hotkey', 'reference_floor_hotkey'):
        if field in old:
            assert settings[field] == old[field]
    fields = ('map_key', 'recognize_hotkey', 'project_hotkey', 'reference_hotkey',
              'reference_floor_hotkey', 'next_round_hotkey', *CHEST_HOTKEY_FIELDS)
    assert len({settings[field] for field in fields}) == len(fields)
    save_settings(settings, path)
    assert load_settings(MapCatalog(), path) == settings


def test_new_settings_defaults_and_invalid_booleans_do_not_write_on_read(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('{"capture_exclusion_enabled": "false", "chest_counter_visible": null}')
    settings = load_settings(MapCatalog(), path)
    assert settings['capture_exclusion_enabled'] is True
    assert settings['chest_counter_visible'] is True
    assert [settings[field] for field in CHEST_HOTKEY_FIELDS] == ['Num4', 'Num5', 'Num1', 'Num2', 'Num7', 'Num8']
    assert settings['map_capture_delay_ms'] == 30
    assert settings['auto_select_layout'] is True
    assert settings['reference_enabled'] is True and settings['show_projection'] is True
    assert settings['next_round_hotkey'] == 'Num0'


@pytest.mark.parametrize('value,expected', [(True, True), (False, False), ('true', False), (1, False), (None, False), ([], False)])
def test_auto_select_layout_requires_explicit_boolean_and_roundtrips(tmp_path, value, expected):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'skip_layout_confirmation': True}), encoding='utf-8')
    before = path.read_bytes()
    assert load_settings(MapCatalog(), path)['auto_select_layout'] is True
    assert path.read_bytes() == before
    path.write_text(json.dumps({'auto_select_layout': value}), encoding='utf-8')
    settings = load_settings(MapCatalog(), path)
    assert settings['auto_select_layout'] is expected
    save_settings(settings, path)
    assert load_settings(MapCatalog(), path)['auto_select_layout'] is expected


def test_reference_floor_key_preserves_existing_shortcuts_and_normalizes_hard_floor(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'map_key': 'J', 'map_id': 'a_bringer_of_doom', 'difficulty_id': 'hard',
                               'reference_floor': 'b1'}), encoding='utf-8')
    before = path.read_bytes()
    settings = load_settings(MapCatalog(), path)
    assert path.read_bytes() == before
    assert settings['map_key'] == 'J' and settings['reference_floor_hotkey'] == 'K'
    assert settings['reference_floor'] == '2nd'
    save_settings(settings, path)
    assert load_settings(MapCatalog(), path) == settings


@pytest.mark.parametrize('value', [None, 'unknown', [], 'cnn', 'icon_structure', 'pure_structure'])
def test_settings_default_to_structure_and_preserve_explicit_choices(tmp_path, value):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'layout_recognition_backend': value}), encoding='utf-8')
    assert load_settings(MapCatalog(), path)['layout_recognition_backend'] == ('icon_structure' if value == 'icon_structure' else 'pure_structure')


@pytest.mark.parametrize('backend', ['cpu', 'auto', 'legacy', 'template'])
def test_old_settings_migrate_without_touching_user_preferences(tmp_path, backend, preference_catalog):
    path = tmp_path / 'settings.json'
    old = dict(registration_backend=backend, layout_recognition_backend='cnn',
               opacity=.41, map_capture_delay_ms=78, recognize_hotkey='F7',
               template_route_versions={'a_bringer_of_doom/nightmare': 'route_lhp_solo'},
               template_motion_tracking=False, skip_layout_confirmation=True, auto_floor=True)
    path.write_text(json.dumps(old), encoding='utf-8')
    before = path.read_bytes()
    settings = load_settings(preference_catalog, path)
    assert path.read_bytes() == before  # Reading never rewrites personal settings.
    assert settings['registration_backend'] == 'template'
    assert settings['layout_recognition_backend'] == 'pure_structure'
    assert settings['skip_layout_confirmation'] is False
    assert settings['auto_floor'] is False
    for name in ('opacity', 'map_capture_delay_ms', 'recognize_hotkey', 'template_route_versions', 'template_motion_tracking'):
        assert settings[name] == old[name]
    save_settings(settings, path)
    assert load_settings(preference_catalog, path) == settings


def test_retired_route_version_uses_available_default_without_rewriting_settings(tmp_path):
    path = tmp_path/'settings.json'
    profile = MapCatalog().profile('a_bringer_of_doom','nightmare')
    path.write_text(json.dumps(dict(map_id=profile.map_id, difficulty_id=profile.difficulty_id,
        template_route_versions={profile.key:'route_lhp_solo_clean'})), encoding='utf-8')
    before = path.read_bytes()
    settings = load_settings(MapCatalog(), path)
    assert settings['template_route_versions'][profile.key] == profile.template_versions[0]
    assert path.read_bytes() == before


def test_missing_template_preference_uses_declared_default_and_preferences_remain_independent(tmp_path, preference_catalog):
    catalog = preference_catalog
    path = tmp_path/'settings.json'
    profile = next(iter(catalog.profiles.values()))
    path.write_text(json.dumps(dict(route_versions={profile.key: {'layout_01': 'route_clean'}})))
    settings = load_settings(catalog, path)
    assert selected_template_variant(settings, profile.key) == profile.template_versions[0]
    settings['template_route_versions'][profile.key] = 'route_lhp_solo'
    save_settings(settings, path)
    reloaded = load_settings(catalog, path)
    assert selected_template_variant(reloaded, profile.key) == 'route_lhp_solo'
    assert reloaded['route_versions'] == settings['route_versions']


def test_template_preferences_are_validated_per_difficulty_without_writing_settings(tmp_path, preference_catalog):
    catalog = preference_catalog
    nightmare = catalog.profile('a_bringer_of_doom', 'nightmare')
    hard = catalog.profile('a_bringer_of_doom', 'hard')
    path = tmp_path/'settings.json'
    path.write_text(json.dumps(dict(map_id=nightmare.map_id, difficulty_id=nightmare.difficulty_id,
        template_route_versions={nightmare.key: 'route_lhp_solo', hard.key: 'route_lhp_solo'})))
    before = path.read_bytes()
    settings = load_settings(catalog, path)
    assert selected_template_variant(settings, nightmare.key) == 'route_lhp_solo'
    assert hard.key not in settings['template_route_versions']
    assert path.read_bytes() == before


def test_unset_template_preference_does_not_assume_a_named_route():
    assert selected_template_variant({}, 'sample/hard') is None


def test_old_version_settings_migrate_without_losing_user_preferences(tmp_path):
    path = tmp_path / 'settings.json'
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    old = dict(registration_backend='cpu', map_capture_delay_ms=77, opacity=.42,
               route_versions={profile.key: {'layout_01': 'route_01', 'layout_02': 'route_clean'}})
    path.write_text(json.dumps(old), encoding='utf-8')
    loaded = load_settings(catalog, path)
    assert loaded['registration_backend'] == 'template'
    assert loaded['map_capture_delay_ms'] == 77 and loaded['opacity'] == .42
    assert loaded['route_versions'][profile.key] == {'layout_01': 'route_01', 'layout_02': 'route_clean'}
    save_settings(old, path)
    assert json.loads(path.read_text(encoding='utf-8'))['route_versions'][profile.key]['layout_01'] == 'route_01'
    assert old['route_versions'][profile.key]['layout_01'] == 'route_01'


@pytest.mark.parametrize('value,expected', [(None,30),('invalid',30),(True,30),(-4,0),(900,500),(0,0),(35,35),(65,65)])
def test_capture_delay_settings_normalize_and_roundtrip(tmp_path,value,expected):
    path=tmp_path/'settings.json';catalog=MapCatalog()
    path.write_text('{}',encoding='utf-8')
    assert load_settings(catalog,path)['map_capture_delay_ms']==30
    path.write_text(json.dumps({'map_capture_delay_ms':value}),encoding='utf-8')
    settings=load_settings(catalog,path)
    assert settings['map_capture_delay_ms']==expected
    save_settings(settings,path)
    assert load_settings(catalog,path)['map_capture_delay_ms']==expected


def test_projection_settings_keep_map_roi_inside_game_client(tmp_path) -> None:
    catalog = MapCatalog()
    path = tmp_path / "settings.json"
    settings = load_settings(catalog, path)
    assert settings["capture_reference_size"] == [2560, 1440]
    settings["capture_roi"] = [2000, 0, 600, 300]
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="完全位于"):
        load_settings(catalog, path)
