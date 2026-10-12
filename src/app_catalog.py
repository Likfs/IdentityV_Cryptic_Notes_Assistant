from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from src.layout_retrieval import normalize_recognition_backend


from src.paths import PROJECT_ROOT, CONFIG_ROOT, SETTINGS_PATH
DEFAULT_CATALOG_PATH = PROJECT_ROOT / 'configs/map_catalog.json'
DEFAULT_SETTINGS_PATH = SETTINGS_PATH
FLOORS = ("2nd", "1st", "b1")
FLOOR_NAMES = {"2nd": "2 楼", "1st": "1 楼", "b1": "地下室"}
VERSION_NAMES = {"route_clean": "原版干净底图", "route_lhp_solo": "凉哈皮单人速刷路线图", "route_zhanshi_fast": "展十速刷路线图"}
TEMPLATE_VERSION_NAMES = {key: VERSION_NAMES[key] for key in ('route_zhanshi_fast', 'route_lhp_solo', 'route_clean')}
TEMPLATE_VERSION_NAMES.update({
    'route_lhp_solo_clean': '凉哈皮单人速刷路线图底图版',
    'route_zhanshi_fast_clean': '展十速刷路线图底图版',
})


def selected_template_variant(settings: dict, profile_key: str) -> str:
    return settings.get('template_route_versions', {}).get(profile_key, 'route_zhanshi_fast')


def canonical_route_version(value: str) -> str:
    """Read the historical Lianghapi ID while persisting its descriptive name."""
    return "route_lhp_solo" if value == "route_01" else value


def normalize_route_versions(saved: dict) -> dict:
    return {profile: {layout: canonical_route_version(version) for layout, version in choices.items()}
            for profile, choices in saved.items() if isinstance(choices, dict)}


HOTKEYS = tuple(f"F{i}" for i in range(1, 13))
MAP_KEYS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + (
    "Tab", "Space", "Shift", "Ctrl", "Alt", "CapsLock", "X1", "X2"
) + HOTKEYS
ACTION_KEYS = tuple(key for key in MAP_KEYS if key not in ("Shift", "Ctrl", "Alt", "X1", "X2"))


@dataclass(frozen=True)
class DifficultyProfile:
    map_id: str
    map_name: str
    difficulty_id: str
    difficulty_name: str
    clean_root: Path
    calibration_root: Path
    asset_versions: tuple[dict, ...] = ()
    entrance_root: Path | None = None

    @property
    def key(self) -> str:
        return f"{self.map_id}/{self.difficulty_id}"


@dataclass(frozen=True)
class RouteVariant:
    layout: str
    variant_id: str
    source_name: str
    calibration_path: Path
    floor_status: dict[str, str]
    display_name: str = ""
    asset_kind: str = "route"

    @property
    def label(self) -> str:
        return self.display_name or VERSION_NAMES.get(self.variant_id, self.variant_id)


class MapCatalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = path.resolve()
        raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if raw.get("schema_version") != 1 or not isinstance(raw.get("maps"), list):
            raise ValueError("地图目录配置格式不正确")
        self.profiles: dict[tuple[str, str], DifficultyProfile] = {}
        self.map_names: dict[str, str] = {}
        for map_item in raw["maps"]:
            map_id = str(map_item["id"])
            if map_id in self.map_names:
                raise ValueError(f"地图 ID 重复：{map_id}")
            self.map_names[map_id] = str(map_item["name"])
            for difficulty in map_item["difficulties"]:
                difficulty_id = str(difficulty["id"])
                key = (map_id, difficulty_id)
                if key in self.profiles:
                    raise ValueError(f"地图难度重复：{map_id}/{difficulty_id}")
                self.profiles[key] = DifficultyProfile(
                    map_id=map_id,
                    map_name=self.map_names[map_id],
                    difficulty_id=difficulty_id,
                    difficulty_name=str(difficulty["name"]),
                    clean_root=self._resolve(difficulty["clean_root"]),
                    calibration_root=self._resolve(difficulty["calibration_root"]),
                    asset_versions=tuple(difficulty.get("asset_versions", [])),
                    entrance_root=self._resolve(difficulty["entrance_root"]) if difficulty.get("entrance_root") else None,
                )
        if not self.profiles:
            raise ValueError("地图目录中没有可用的地图难度")

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    def difficulties(self, map_id: str) -> list[DifficultyProfile]:
        return [profile for (item_map, _), profile in self.profiles.items() if item_map == map_id]

    def profile(self, map_id: str, difficulty_id: str) -> DifficultyProfile:
        try:
            return self.profiles[(map_id, difficulty_id)]
        except KeyError as exc:
            raise ValueError(f"未配置地图难度：{map_id}/{difficulty_id}") from exc

    def layouts(self, profile: DifficultyProfile) -> list[str]:
        return sorted(path.name for path in profile.clean_root.glob("layout_*") if path.is_dir())

    def variants(self, profile: DifficultyProfile, layout: str) -> list[RouteVariant]:
        variants = []
        for path in sorted(profile.calibration_root.glob(f"route_*/{layout}/calibration.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("layout") != layout:
                continue
            variants.append(
                RouteVariant(
                    layout=layout,
                    variant_id=str(payload["variant_id"]),
                    source_name=Path(payload["route"]).name,
                    calibration_path=path,
                    floor_status={floor: str(payload["floors"][floor]["status"]) for floor in FLOORS},
                    display_name=next((v["name"] for v in profile.asset_versions if v["id"] == payload["variant_id"]),
                                      VERSION_NAMES.get(payload["variant_id"], str(payload.get("display_name", "")))),
                    asset_kind=str(payload.get("asset_kind", "route")),
                )
            )
        order = {v["id"]: index for index, v in enumerate(profile.asset_versions)}
        return sorted(variants, key=lambda v: order.get(v.variant_id, len(order)))

    def selected_variant(self, profile: DifficultyProfile, layout: str, saved: dict) -> RouteVariant | None:
        variants = self.variants(profile, layout)
        if not variants:
            return None
        preferred = canonical_route_version(saved.get(profile.key, {}).get(layout))
        default = next((item for item in variants if item.variant_id == "route_lhp_solo"), variants[0])
        return next((item for item in variants if item.variant_id == preferred), default)


    def clean_template(self, profile: DifficultyProfile, layout: str, floor: str) -> Path:
        matches = list((profile.clean_root / layout).glob(f"*_{floor}.png"))
        if len(matches) != 1:
            raise RuntimeError(f"{profile.map_name}/{profile.difficulty_name}/{layout}/{floor} 干净底图数量为 {len(matches)}")
        return matches[0]


def key_to_vk(key: str) -> int:
    if len(key) == 1 and key.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        return ord(key.upper())
    if re.fullmatch(r"F([1-9]|1[0-2])", key):
        return 0x70 + int(key[1:]) - 1
    special = {
        "Tab": 0x09,
        "Space": 0x20,
        "Shift": 0x10,
        "Ctrl": 0x11,
        "Alt": 0x12,
        "CapsLock": 0x14,
        "X1": 0x05,
        "X2": 0x06,
    }
    if key not in special:
        raise ValueError(f"不支持的地图按键：{key}")
    return special[key]




DEFAULT_MAP_CAPTURE_DELAY_MS = 150
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
    first = next(iter(catalog.profiles.values()))
    settings = load_default_settings()
    settings["map_id"], settings["difficulty_id"] = first.map_id, first.difficulty_id
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(saved, dict):
            settings.update(saved)
    if (settings["map_id"], settings["difficulty_id"]) not in catalog.profiles:
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
        if settings[name] not in ACTION_KEYS:
            settings[name] = "F8" if name == "recognize_hotkey" else "F10"
    if settings["recognize_hotkey"] == settings["project_hotkey"]:
        settings["project_hotkey"] = "F10" if settings["recognize_hotkey"] != "F10" else "F11"
    if settings["map_key"] in (settings["recognize_hotkey"], settings["project_hotkey"]):
        raise ValueError("地图按键不能与程序快捷键相同")
    used_keys = {settings[name] for name in ("map_key", "recognize_hotkey", "project_hotkey")}
    if settings["reference_hotkey"] not in ACTION_KEYS or settings["reference_hotkey"] in used_keys:
        settings["reference_hotkey"] = next(key for key in ("H", "J", "K", "L") if key not in used_keys)
    if settings["floor"] not in FLOORS:
        settings["floor"] = "2nd"
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
    settings["route_versions"] = normalize_route_versions(settings["route_versions"])
    preferences = settings.get('template_route_versions', {})
    settings['template_route_versions'] = {
        key: value for key, value in preferences.items() if value in TEMPLATE_VERSION_NAMES
    } if isinstance(preferences, dict) else {}
    return settings


def save_settings(settings: dict, path: Path = DEFAULT_SETTINGS_PATH) -> None:
    settings = {**settings, "registration_backend": "template", "layout_recognition_backend": normalize_recognition_backend(settings.get("layout_recognition_backend")),
                "auto_floor": False, "skip_layout_confirmation": False, "route_versions": normalize_route_versions(settings.get("route_versions", {}))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
