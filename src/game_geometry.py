"""Desktop game layout, using one GL scale for UI, recognition and registration."""
import math
from dataclasses import dataclass

from src.capture.region import CaptureRegion

DESIGN_WIDTH = 1334.
DESIGN_HEIGHT = 750.
ASPECT_TOLERANCE = .00011
REFERENCE_SIZE = (2560, 1440)
REFERENCE_UI_SCALE = REFERENCE_SIZE[0] / DESIGN_WIDTH
MAP_X = (369., 1141.)
MAP_HALF_HEIGHT = 291.
MATCH_X = (500., 1128.)  # Beyond left UI's x=494 and before floor buttons at x=1134.56.
REFERENCE_MAP_CENTER = (sum(MAP_X)/2 * REFERENCE_SIZE[0]/DESIGN_WIDTH, REFERENCE_SIZE[1]/2)
FLOOR_X = (1134.56, 1200.56)
FLOOR_SLOTS = ((194., 278.), (134., 212.), (74., 152.), (14., 98.))
# Sampling margin around visible button textures, distinct from their hit boxes.
FLOOR_SAMPLE_REFERENCE = (2170, 180, 140, 510)


@dataclass(frozen=True)
class GameGeometry:
    client: CaptureRegion
    policy: str
    visible_size: tuple[int, int]
    scale: float
    viewport_origin: tuple[float, float]

    @property
    def capture_factor(self):
        return REFERENCE_UI_SCALE / self.scale

    def point(self, right_offset, center_offset=0.):
        """Physical screen point; offsets are right-anchored and top-down."""
        return (self.client.left + self.viewport_origin[0] +
                (self.visible_size[0]-right_offset)*self.scale,
                self.client.top + self.client.height/2 + center_offset*self.scale)

    def diagnostics(self):
        return dict(geometry_model=self.policy, resolution_policy=4 if self.policy == 'fixed_width' else 3,
                    base_design_size=[int(DESIGN_WIDTH), int(DESIGN_HEIGHT)],
                    visible_design_size=list(self.visible_size), design_width=self.visible_size[0],
                    design_client_height=self.visible_size[1], ui_scale=self.scale,
                    viewport_origin_client=list(self.viewport_origin),
                    capture_scale_factor=self.capture_factor, aspect_tolerance=ASPECT_TOLERANCE,
                    policy_source='desktop_client_aspect_ratio')


def resolve_geometry(client):
    """Match the reported PC policy and Cocos GLView's ceil/viewport rules.

    The desktop client has phone adaptation disabled. cur_scale is independently
    controlled by map interaction, represented by the existing tracked zoom.
    """
    width, height = client.width, client.height
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('截图时游戏客户区尺寸必须为两个正整数')
    if width/height < DESIGN_WIDTH/DESIGN_HEIGHT - ASPECT_TOLERANCE:
        policy, scale = 'fixed_width', width/DESIGN_WIDTH
        visible = (int(DESIGN_WIDTH), -(-height*int(DESIGN_WIDTH)//width))
    else:
        policy, scale = 'fixed_height', height/DESIGN_HEIGHT
        visible = (-(-width*int(DESIGN_HEIGHT)//height), int(DESIGN_HEIGHT))
    origin = ((width-visible[0]*scale)/2, (height-visible[1]*scale)/2)
    return GameGeometry(client, policy, visible, scale, origin)


def _rectangle(client, left, top, right, bottom, *, clip=True):
    # Avoid adding a pixel when inverse reference transforms land at n + 1e-13.
    left, top = math.floor(left+1e-9), math.floor(top+1e-9)
    right, bottom = math.ceil(right-1e-9), math.ceil(bottom-1e-9)
    if clip:
        left, top = max(client.left, left), max(client.top, top)
        right = min(client.left + client.width, right)
        bottom = min(client.top + client.height, bottom)
    if right <= left or bottom <= top:
        raise ValueError('游戏 UI 区域不在可见客户区内')
    return CaptureRegion(left, top, right-left, bottom-top)


def map_viewport(client):
    geometry = resolve_geometry(client)
    return _rectangle(client, *geometry.point(DESIGN_WIDTH-MAP_X[0], -MAP_HALF_HEIGHT),
                      *geometry.point(DESIGN_WIDTH-MAP_X[1], MAP_HALF_HEIGHT))


def map_center(client):
    return resolve_geometry(client).point(DESIGN_WIDTH-sum(MAP_X)/2)


def map_match_region(client, viewport):
    """Only matching loses the side strips; projection keeps the full viewport."""
    geometry = resolve_geometry(client)
    left = max(viewport.left, math.ceil(geometry.point(DESIGN_WIDTH-MATCH_X[0])[0]-1e-9))
    right = min(viewport.left+viewport.width, math.floor(geometry.point(DESIGN_WIDTH-MATCH_X[1])[0]+1e-9))
    if right <= left:
        raise ValueError('游戏窗口没有可用于匹配的地图区域')
    return CaptureRegion(left, viewport.top, right-left, viewport.height)


def reference_region(client, rect):
    """Reference UI coordinates retain their right/center anchor in either policy."""
    x, y, width, height = rect
    geometry = resolve_geometry(client)
    return _rectangle(client,
        *geometry.point(DESIGN_WIDTH-x/REFERENCE_UI_SCALE, (y-REFERENCE_SIZE[1]/2)/REFERENCE_UI_SCALE),
        *geometry.point(DESIGN_WIDTH-(x+width)/REFERENCE_UI_SCALE,
                        (y+height-REFERENCE_SIZE[1]/2)/REFERENCE_UI_SCALE))


def floor_sample_region(client):
    return reference_region(client, FLOOR_SAMPLE_REFERENCE)


def floor_slot_regions(client):
    geometry = resolve_geometry(client)
    return tuple(_rectangle(client, *geometry.point(DESIGN_WIDTH-FLOOR_X[0], -b),
                            *geometry.point(DESIGN_WIDTH-FLOOR_X[1], -a), clip=False)
                 for a, b in FLOOR_SLOTS)


def reference_floor_slots():
    return tuple(tuple(rect.as_list()) for rect in floor_slot_regions(CaptureRegion(0, 0, *REFERENCE_SIZE)))


def recognition_size(client, viewport):
    # Retain the saved reference convention without rebuilding feature indices.
    scale = resolve_geometry(client).capture_factor
    return round(viewport.width*scale), round(viewport.height*scale)
