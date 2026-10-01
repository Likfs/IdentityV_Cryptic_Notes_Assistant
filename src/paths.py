"""Read-only application resources and writable runtime locations."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
SETTINGS_PATH = RUNTIME_ROOT / "settings.json"
CACHE_ROOT = RUNTIME_ROOT / "cache"
LOG_ROOT = RUNTIME_ROOT / "logs"
DIAGNOSTIC_ROOT = RUNTIME_ROOT / "diagnostics"
