from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from src.layout_retrieval import normalize_recognition_backend
from src.chest_counter import CHEST_ACTIONS
from src.scale_config import load_scales, profile_settings, validate_scales
from src.floor_definitions import FLOORS, FLOOR_NAMES, DEFAULT_FLOORS
from src.hotkeys import (HOTKEYS, MAP_KEYS, ACTION_KEYS, key_to_vk, normalize_hotkey,
                         conflicts_with_map)


from src.paths import PROJECT_ROOT, CONFIG_ROOT, SETTINGS_PATH
DEFAULT_CATALOG_PATH = PROJECT_ROOT / 'configs/map_catalog.json'
DEFAULT_SETTINGS_PATH = SETTINGS_PATH
CLEAN_VERSION_ID = 'clean'
LEGACY_CLEAN_VERSION_ID = 'route_clean'


def selected_template_variant(settings: dict, profile_key: str) -> str | None:
    return settings.get('template_route_versions', {}).get(profile_key)


def effective_template_variant(preferred: str | None, available) -> str | None:
    """Resolve a display version without overwriting the user's route preference."""
    choices = tuple(available)
    if preferred in choices:
        return preferred
    if preferred is not None and CLEAN_VERSION_ID in choices:
        return CLEAN_VERSION_ID
    return next(iter(choices), None)


@dataclass(frozen=True)
class DifficultyProfile:
    map_id: str
    map_name: str
    difficulty_id: str
    difficulty_name: str
    clean_root: Path | None
    asset_versions: tuple[dict, ...] = ()
    entrance_root: Path | None = None
    floors: tuple[str, ...] = DEFAULT_FLOORS
    template_versions: tuple[str, ...] = ()
    asset_status: str = 'ready'
    minimum_scale: float | None = None
    asset_contract: str = ''
    layout_floors: dict[str, tuple[str, ...]] = field(default_factory=dict)
    layout_minimum_scales: dict[str, float] = field(default_factory=dict)

    def floors_for_layout(self, layout):
        return self.layout_floors.get(layout, self.floors)

    def minimum_scale_for_layout(self, layout):
        return self.layout_minimum_scales.get(layout, self.minimum_scale)

    @property
    def key(self) -> str:
        return f"{self.map_id}/{self.difficulty_id}"

    @property
    def version_names(self) -> dict[str, str]:
        declared = {v['id']: v['name'] for v in self.asset_versions}
        return declared


@dataclass(frozen=True)
class RouteVariant:
    layout: str
    variant_id: str
    source_name: str
    calibration_path: Path | None
    floor_status: dict[str, str]
    display_name: str = ""
    asset_kind: str = "route"

    @property
    def label(self) -> str:
        return self.display_name or self.variant_id


class MapCatalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH, *, data=None, scales=None) -> None:
        self.path = path.resolve()
        raw = json.loads(self.path.read_text(encoding="utf-8-sig")) if data is None else data
        if raw.get("schema_version") != 2 or not isinstance(raw.get("maps"), list):
            raise ValueError("地图目录配置格式不正确")
        scale_data = load_scales(self.path.parent.parent) if scales is None else validate_scales(scales)
        self.profiles: dict[tuple[str, str], DifficultyProfile] = {}
        self.map_names: dict[str, str] = {}
        for map_item in raw["maps"]:
            map_id = str(map_item["id"])
            if map_id in self.map_names:
                raise ValueError(f"地图 ID 重复：{map_id}")
            self.map_names[map_id] = str(map_item["name"])
            for difficulty in map_item["difficulties"]:
                contract = difficulty.get('asset_contract', '')
                status = difficulty.get('asset_status', 'ready')
                if status not in ('ready', 'pending'):
                    raise ValueError('资产状态必须为 ready 或 pending')
                if contract != 'aligned_floor_images':
                    raise ValueError('未知资产契约')
                if 'minimum_scale' in difficulty or 'layout_minimum_scales' in difficulty:
                    raise ValueError('尺度只能在 configs/scale_config.json 配置，请移除 map_catalog.json 中重复的尺度字段')
                scale_entry = profile_settings(scale_data, map_id, str(difficulty['id']), ready=status == 'ready')
                scale = scale_entry['minimum_scale']
                layout_scales = scale_entry.get('layout_minimum_scales', {})
                definitions = difficulty.get('asset_versions', [])
                names = {version['id']: version['name'] for version in definitions}
                if len(names) != len(definitions) or any(not isinstance(key, str) or not key
                    or not isinstance(value, str) or not value for key, value in names.items()):
                    raise ValueError('路线版本 ID 或名称无效、重复')
                floors = tuple(difficulty.get('floors', [] if status == 'pending' else DEFAULT_FLOORS))
                if (not floors and status != 'pending') or len(set(floors)) != len(floors) or any(f not in FLOORS for f in floors):
                    raise ValueError('地图难度的楼层配置无效')
                layout_floors = difficulty.get('layout_floors', {})
                if (not isinstance(layout_floors, dict) or any(
                        not isinstance(layout, str) or not layout.startswith('layout_')
                        or not isinstance(values, list) or not values or len(values) != len(set(values))
                        or any(floor not in floors for floor in values)
                        for layout, values in layout_floors.items())):
                    raise ValueError('布局实际楼层配置无效')
                versions = tuple(difficulty.get('template_versions',
                                 tuple(names)))
                supported = names
                if ((not versions and status != 'pending') or len(set(versions)) != len(versions)
                        or any(v not in supported for v in versions)):
                    raise ValueError('地图难度的模板版本配置无效')
                clean_root = difficulty.get('clean_root')
                if not clean_root and status == 'ready':
                    raise ValueError('已配置难度必须声明 clean_root')
                difficulty_id = str(difficulty["id"])
                key = (map_id, difficulty_id)
                if key in self.profiles:
                    raise ValueError(f"地图难度重复：{map_id}/{difficulty_id}")
                self.profiles[key] = DifficultyProfile(
                    map_id=map_id,
                    map_name=self.map_names[map_id],
                    difficulty_id=difficulty_id,
                    difficulty_name=str(difficulty["name"]),
                    clean_root=self._resolve(clean_root) if clean_root else None,
                    asset_versions=tuple(difficulty.get("asset_versions", [])),
                    entrance_root=self._resolve(difficulty["entrance_root"]) if difficulty.get("entrance_root") else None,
                    floors=floors, template_versions=versions,
                    asset_status=status, minimum_scale=scale, asset_contract=contract,
                    layout_floors={layout: tuple(values) for layout, values in layout_floors.items()},
                    layout_minimum_scales=dict(layout_scales),
                )

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.path.parent.parent / path).resolve()

    def difficulties(self, map_id: str) -> list[DifficultyProfile]:
        return [profile for (item_map, _), profile in self.profiles.items() if item_map == map_id]

    def profile(self, map_id: str, difficulty_id: str) -> DifficultyProfile:
        try:
            return self.profiles[(map_id, difficulty_id)]
        except KeyError as exc:
            raise ValueError(f"未配置地图难度：{map_id}/{difficulty_id}") from exc

    def layouts(self, profile: DifficultyProfile) -> list[str]:
        if profile.asset_status == 'pending':
            return []
        return sorted(path.name for path in profile.clean_root.glob("layout_*") if path.is_dir())

    def variants(self, profile: DifficultyProfile, layout: str) -> list[RouteVariant]:
        from src.runtime_resources import read_bundle
        bundle = read_bundle(self.path.parent.parent)
        if bundle is None:
            raise ValueError('缺少运行资源清单，请先发布素材')
        return [RouteVariant(layout=layout, calibration_path=None, **record)
                for record in bundle['route_status'].get(profile.key, {}).get(layout, [])]

    def selected_variant(self, profile: DifficultyProfile, layout: str, saved: dict) -> RouteVariant | None:
        variants = self.variants(profile, layout)
        if not variants:
            return None
        preferred = saved.get(profile.key)
        effective = effective_template_variant(preferred, (item.variant_id for item in variants))
        return next(item for item in variants if item.variant_id == effective)

    def clean_template(self, profile: DifficultyProfile, layout: str, floor: str) -> Path:
        if floor not in profile.floors_for_layout(layout):
            raise ValueError(f'{profile.difficulty_name}不包含楼层：{floor}')
        matches = list((profile.clean_root / layout).glob(f"*_{floor}.png"))
        if len(matches) != 1:
            raise RuntimeError(f"{profile.map_name}/{profile.difficulty_name}/{layout}/{floor} 干净底图数量为 {len(matches)}")
        return matches[0]


DEFAULT_MAP_CAPTURE_DELAY_MS = 30
MAX_MAP_CAPTURE_DELAY_MS = 500


def map_capture_delay_ms(settings: dict) -> int:
    """Normalize older/malformed settings without making the application unstartable."""
    try:
        value = settings.get("map_capture_delay_ms", DEFAULT_MAP_CAPTURE_DELAY_MS)
        if isinstance(value, bool):
            return DEFAULT_MAP_CAPTURE_DELAY_MS
        return min(MAX_MAP_CAPTURE_DELAY_MS, max(0, int(value)))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_MAP_CAPTURE_DELAY_MS


def load_default_settings() -> dict:
    return json.loads((CONFIG_ROOT / "app_defaults.json").read_text(encoding="utf-8-sig"))


def load_settings(catalog: MapCatalog, path: Path = DEFAULT_SETTINGS_PATH) -> dict:
    first = next(iter(catalog.profiles.values()), None)
    settings = load_default_settings()
    settings["map_id"], settings["difficulty_id"] = (first.map_id, first.difficulty_id) if first else (None, None)
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(saved, dict):
            settings.update(saved)
    if first and (settings["map_id"], settings["difficulty_id"]) not in catalog.profiles:
        settings["map_id"], settings["difficulty_id"] = first.map_id, first.difficulty_id
    settings["map_key"] = str(settings["map_key"])
    key_to_vk(settings["map_key"])
    titles = settings.get("game_window_titles", [])
    if isinstance(titles, str):
        titles = titles.split(",")
    if not isinstance(titles, list):
        raise ValueError("游戏窗口标题关键词必须是列表")
    settings["game_window_titles"] = list(dict.fromkeys(str(item).strip() for item in titles if str(item).strip()))
    if not settings["game_window_titles"]:
        settings["game_window_titles"] = ["第五人格", "Identity V", "IdentityV"]
    for name in ("recognize_hotkey", "project_hotkey"):
        settings[name] = normalize_hotkey(settings[name], 'F8' if name == 'recognize_hotkey' else 'F10')
    if settings["recognize_hotkey"] == settings["project_hotkey"]:
        settings["project_hotkey"] = "F10" if settings["recognize_hotkey"] != "F10" else "F11"
    if any(conflicts_with_map(settings[name], settings['map_key']) for name in ('recognize_hotkey', 'project_hotkey')):
        raise ValueError('地图按键不能与程序快捷键使用相同主键或修饰键')
    used_keys = {settings[name] for name in ("map_key", "recognize_hotkey", "project_hotkey")}
    def available(key):
        return key not in used_keys and not conflicts_with_map(key, settings['map_key'])
    settings['reference_hotkey'] = normalize_hotkey(settings['reference_hotkey'], 'H')
    if not available(settings['reference_hotkey']):
        settings['reference_hotkey'] = next(key for key in ('H', 'J', 'K', 'L', *ACTION_KEYS) if available(key))
    used_keys.add(settings['reference_hotkey'])
    for action, (_, default, _, _) in CHEST_ACTIONS.items():
        field = f'{action}_hotkey'
        key = normalize_hotkey(settings.get(field, default), default)
        if not available(key):
            key = next(value for value in (default, *ACTION_KEYS) if available(value))
        settings[field] = key
        used_keys.add(key)
    key = normalize_hotkey(settings.get('reference_floor_hotkey', 'J'), 'J')
    if not available(key):
        key = next(value for value in ('J', 'K', 'L', *ACTION_KEYS) if available(value))
    settings['reference_floor_hotkey'] = key
    used_keys.add(key)
    key = normalize_hotkey(settings.get('next_round_hotkey', 'Num0'), 'Num0')
    if not available(key):
        key = next(value for value in ('Num0', 'Ctrl+Num0', 'Shift+Num0', 'Ctrl+N', *ACTION_KEYS)
                   if available(value))
    settings['next_round_hotkey'] = key
    settings['auto_select_layout'] = settings.get('auto_select_layout') is True
    for field in ('chest_counter_visible', 'capture_exclusion_enabled', 'reference_enabled', 'show_projection'):
        settings[field] = settings.get(field) if isinstance(settings.get(field), bool) else True
    profile = catalog.profiles.get((settings['map_id'], settings['difficulty_id']))
    floors = profile.floors if profile else ()
    if settings["floor"] not in floors:
        settings["floor"] = next(iter(floors), None)
    if settings.get('reference_floor') not in floors:
        settings['reference_floor'] = next(iter(floors), None)
    settings["opacity"] = min(1.0, max(0.0, float(settings["opacity"])))
    settings["reference_opacity"] = min(1.0, max(0.0, float(settings["reference_opacity"])))
    roi = settings["capture_roi"]
    if not isinstance(roi, list) or len(roi) != 4 or any(int(value) < 0 for value in roi[:2]) or any(int(value) <= 0 for value in roi[2:]):
        raise ValueError("截图区域必须为 [左, 上, 宽, 高]，宽高需大于 0")
    settings["capture_roi"] = [int(value) for value in roi]
    reference_size = settings.get("capture_reference_size", [2560, 1440])
    if not isinstance(reference_size, list) or len(reference_size) != 2 or any(int(value) <= 0 for value in reference_size):
        raise ValueError("游戏参考分辨率必须为 [宽, 高]，且两项均大于 0")
    settings["capture_reference_size"] = [int(value) for value in reference_size]
    if (settings["capture_roi"][0] + settings["capture_roi"][2] > settings["capture_reference_size"][0]
            or settings["capture_roi"][1] + settings["capture_roi"][3] > settings["capture_reference_size"][1]):
        raise ValueError("地图截图区域必须完全位于游戏参考画面内")
    settings["auto_project"] = bool(settings["auto_project"])
    settings["auto_floor"] = bool(settings["auto_floor"])
    settings["template_motion_tracking"] = bool(settings.get("template_motion_tracking", True))
    settings["map_capture_delay_ms"] = map_capture_delay_ms(settings)
    settings.update(registration_backend="template", layout_recognition_backend=normalize_recognition_backend(settings.get("layout_recognition_backend")),
                    auto_floor=False, skip_layout_confirmation=False)
    if not isinstance(settings["route_versions"], dict):
        settings["route_versions"] = {}
    preferences = settings.get('template_route_versions', {})
    known_versions = {p.key: set(p.template_versions) for p in catalog.profiles.values()}
    settings['template_route_versions'] = {}
    if isinstance(preferences, dict):
        for key, value in preferences.items():
            versions = known_versions.get(key, ())
            if value == LEGACY_CLEAN_VERSION_ID and CLEAN_VERSION_ID in versions:
                value = CLEAN_VERSION_ID
            if isinstance(value, str) and value in versions:
                settings['template_route_versions'][key] = value
    if profile is not None and profile.asset_status != 'pending':
        settings['template_route_versions'].setdefault(profile.key, profile.template_versions[0])
    return settings


def save_settings(settings: dict, path: Path = DEFAULT_SETTINGS_PATH) -> None:
    settings = {**settings, "registration_backend": "template", "layout_recognition_backend": normalize_recognition_backend(settings.get("layout_recognition_backend")),
                "auto_floor": False, "auto_select_layout": settings.get('auto_select_layout', True) is True,
                "skip_layout_confirmation": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
