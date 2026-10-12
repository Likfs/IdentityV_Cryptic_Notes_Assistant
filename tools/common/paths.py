"""Developer tool locations; no user state is stored in application configs."""
from src.paths import PROJECT_ROOT

WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
ASSET_BUILD_ROOT = WORKSPACE_ROOT / "asset_build"


def asset_output(path, *, publish=False):
    """Stage generated assets unless publication was explicitly requested."""
    path = path.resolve()
    relative = path.relative_to(PROJECT_ROOT)
    return path if publish else ASSET_BUILD_ROOT / relative
