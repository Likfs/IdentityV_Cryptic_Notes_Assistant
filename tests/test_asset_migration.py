"""Public settings and runtime resource checks for the version-first asset tree."""
import json

from src.app_catalog import FLOORS, PROJECT_ROOT, MapCatalog, load_settings, save_settings
from src.floor_indicator import FLOOR_TEMPLATE_DIR
from src.map_presence import MAP_MARKER_PATH
from src.projection.route_overlay import load_route_overlay_asset


def test_old_version_settings_migrate_without_losing_user_preferences(tmp_path):
    path = tmp_path / 'settings.json'
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    old = dict(registration_backend='cpu', map_capture_delay_ms=77, opacity=.42,
               route_versions={profile.key: {'layout_01': 'route_01', 'layout_02': 'route_clean'}})
    path.write_text(json.dumps(old), encoding='utf-8')
    loaded = load_settings(catalog, path)
    assert loaded['registration_backend'] == 'cpu'
    assert loaded['map_capture_delay_ms'] == 77 and loaded['opacity'] == .42
    assert loaded['route_versions'][profile.key] == {'layout_01': 'route_lhp_solo', 'layout_02': 'route_clean'}
    save_settings(old, path)
    assert json.loads(path.read_text(encoding='utf-8'))['route_versions'][profile.key]['layout_01'] == 'route_lhp_solo'
    assert old['route_versions'][profile.key]['layout_01'] == 'route_01'


def test_old_cli_version_selects_canonical_version_and_relocated_assets():
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    saved = {}
    catalog.select_version_for_all(profile, 'route_01', saved)
    for layout in catalog.layouts(profile):
        variant = catalog.selected_variant(profile, layout, saved)
        assert saved[profile.key][layout] == 'route_lhp_solo'
        assert variant.calibration_path == profile.calibration_root / 'route_lhp_solo' / layout / 'calibration.json'
        for floor in FLOORS:
            asset = load_route_overlay_asset(variant.calibration_path, floor)
            assert asset.asset_path.parent == variant.calibration_path.parent


def test_game_templates_and_application_art_are_separate():
    assert FLOOR_TEMPLATE_DIR == PROJECT_ROOT / 'assets/game_ui/floor_indicators'
    assert MAP_MARKER_PATH == PROJECT_ROOT / 'assets/game_ui/my_location.png'
    assert FLOOR_TEMPLATE_DIR.is_dir() and MAP_MARKER_PATH.is_file()
    assert (PROJECT_ROOT / 'assets/app_ui/home_wallpaper.jpg').is_file()
    assert (PROJECT_ROOT / 'assets/app_ui/icons/check.svg').is_file()


def test_calibration_asset_references_are_portable_and_exist():
    def check(value):
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            normalized = value.replace('\\', '/')
            assert not (':/' in normalized and '/assets/' in normalized), value
            if normalized.startswith('assets/'):
                assert (PROJECT_ROOT / normalized).exists(), value
    for path in (PROJECT_ROOT / 'assets/maps').rglob('calibration.json'):
        check(json.loads(path.read_text(encoding='utf-8')))
