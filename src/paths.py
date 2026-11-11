"""Read-only application resources and writable runtime locations."""
from pathlib import Path
import sys

PROJECT_ROOT = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[1])
CONFIG_ROOT = PROJECT_ROOT / "configs"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
SETTINGS_PATH = RUNTIME_ROOT / "settings.json"
CACHE_ROOT = RUNTIME_ROOT / "cache"
LOG_ROOT = RUNTIME_ROOT / "logs"
DIAGNOSTIC_ROOT = RUNTIME_ROOT / "diagnostics"

TEMPLATE_PROVENANCE_PATH = "src/registration/template_matching/provenance.json"


def published_resource_path(root, name):
    """Resolve manifest paths while keeping packaged code metadata with the code."""
    root = Path(root).resolve()
    if (getattr(sys, "frozen", False) and root == PROJECT_ROOT
            and Path(name).as_posix() == TEMPLATE_PROVENANCE_PATH):
        base = Path(sys._MEIPASS).resolve()
    else:
        base = root
    path = (base / name).resolve()
    path.relative_to(base)
    return path
