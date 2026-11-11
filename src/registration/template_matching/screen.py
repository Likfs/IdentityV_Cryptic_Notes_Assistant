"""Scale the 2560 x 1440 convention using the captured game's physical client size."""
from src.capture.region import CaptureRegion
from src.game_geometry import resolve_geometry


def factor_for_size(width,height):
    return resolve_geometry(CaptureRegion(0, 0, width, height)).capture_factor

def capture_info(client_size):
    if not isinstance(client_size, (tuple, list)) or len(client_size) != 2:
        raise ValueError('必须提供截图时游戏客户区的宽度和高度')
    width, height = client_size
    geometry = resolve_geometry(CaptureRegion(0, 0, width, height))
    return dict(source='capture_game_client', width=width, height=height,
                reference_size=[2560, 1440], factor=geometry.capture_factor,
                **geometry.diagnostics())
