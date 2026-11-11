"""Read the selected game floor from the vertical control, independently of map geometry."""
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from src.paths import PROJECT_ROOT
from src.game_geometry import reference_floor_slots

REFERENCE_SIZE = (2560, 1440)
SEARCH_ROI = (2170, 180, 140, 510)
BUTTON_WIDTH, BUTTON_HEIGHT = 88, 115
# Floor order is independent of the control's position and physical size.
CONTROL_KINDS = (('b1', 'b2', 'b3', 'b4'), ('2nd', '1st', 'b1'))
DEFAULT_BUTTONS = reference_floor_slots()
TEMPLATE_PATH = 'assets/app_ui/floor_controls.png'
THRESHOLDS = dict(strip_score=.80, row_min=.45, row_mean=.65, highlight_gap=15, margin=.08)


@dataclass(frozen=True)
class FloorDetection:
    floor: str | None
    buttons: tuple = ()  # Rectangles in the 2560 x 1440 game-client reference coordinates.
    score: float = 0.
    margin: float = 0.
    reason: str | None = None
    candidates: tuple = ()
    control_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    normalized_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    error: str | None = None
    selection_source: str = 'floor_ui'
    button_slots: tuple = ()
    search_path: str = 'not_run'

    def diagnostics(self):
        return dict(floor=self.floor, buttons=[list(rect) for rect in self.buttons],
                    score=self.score, margin=self.margin, reason=self.reason,
                    method='vertical_floor_control', reference_size=list(REFERENCE_SIZE),
                    selection_source=self.selection_source, button_slots=list(self.button_slots),
                    thresholds=dict(THRESHOLDS), candidates=list(self.candidates), error=self.error,
                    input_size=(list(self.control_image.shape[1::-1]) if self.control_image is not None else None),
                    normalized_size=list(SEARCH_ROI[2:]),
                    search_method='client_scaled_ui_templates',
                    search_path=self.search_path,
                    checks=(dict(best_candidate_valid=self.candidates[0]['valid'],
                                 margin=self.margin >= THRESHOLDS['margin']) if self.candidates else None))


@lru_cache(maxsize=1)
def _templates():
    atlas = cv2.imdecode(np.fromfile(PROJECT_ROOT / TEMPLATE_PATH, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if atlas is None or atlas.shape != (460, 616):
        raise ValueError('楼层 UI 模板缺失或尺寸无效，请安装完整资源包')
    result = []
    column = 0
    for labels in CONTROL_KINDS:
        height = len(labels) * BUTTON_HEIGHT
        for selected, floor in enumerate(labels):
            result.append((labels, selected, floor, atlas[460-height:, column*88:(column+1)*88]))
            column += 1
    # Two-floor layouts use the same 2/1 sprites as the first two rows of the
    # supplied three-floor states. Only explicitly declared two-floor layouts
    # select these, so a prefix cannot compete with its complete three-row UI.
    for selected, floor in enumerate(('2nd', '1st')):
        result.append((('2nd', '1st'), selected, floor,
                       atlas[115:345, (4+selected)*88:(5+selected)*88]))
    return result


def _size_hint(input_shape, template_shape):
    # The crop is mapped independently on both axes, whereas game UI scales
    # uniformly. Undo that aspect distortion locally; never change map scale.
    input_height, input_width = input_shape[:2]
    sx, sy = input_width / SEARCH_ROI[2], input_height / SEARCH_ROI[3]
    ui_scale = min(sx, sy)
    width = BUTTON_WIDTH * ui_scale / sx
    height = template_shape[0] * ui_scale / sy
    return width, height


def _candidate_sizes(input_shape, template_shape):
    width, height = _size_hint(input_shape, template_shape)
    # Small finite UI-only adjustments cover rounding and observed button pitch
    # differences. Full-strip and per-label checks still use the same thresholds.
    return sorted({(round(width * dx), round(height * (1 + dy / 100)))
                   for dx in (.975, 1., 1.025) for dy in range(-8, 9, 4)
                   if 0 < round(width * dx) <= SEARCH_ROI[2]
                   and 0 < round(height * (1 + dy / 100)) <= SEARCH_ROI[3]})


@lru_cache(maxsize=6)
def _digit_templates(labels, highlighted):
    return tuple(tuple(template[i*115+22:i*115+90, 18:70]
                       for kind, selected, _, template in _templates()
                       if kind == labels and (selected == i) == highlighted)
                 for i in range(len(labels)))


def _measure_control(gray, labels, selected, floor, template, width, height):
    scaled = (template if template.shape == (height, width) else
              cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA))
    response = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
    _, strip_score, _, (x, y) = cv2.minMaxLoc(response)
    patch = gray[y:y+height, x:x+width]
    if patch.shape != template.shape:
        patch = cv2.resize(patch, (BUTTON_WIDTH, template.shape[0]), interpolation=cv2.INTER_LINEAR)
    rows = [float(cv2.matchTemplate(
        patch[i*115+22:i*115+90, 18:70], template[i*115+22:i*115+90, 18:70],
        cv2.TM_CCOEFF_NORMED)[0, 0]) for i in range(len(labels))]
    background = [float(np.median(patch[i*115+20:i*115+95, 4:16])) for i in range(len(labels))]
    other_background = max(v for i, v in enumerate(background) if i != selected)
    unique_highlight = background[selected] > other_background + THRESHOLDS['highlight_gap']
    row_mean, row_min = float(np.mean(rows)), min(rows)
    score = float((strip_score + row_mean) / 2)
    edges = [round(y + i * height / len(labels)) for i in range(len(labels) + 1)]
    buttons = tuple((SEARCH_ROI[0]+x, SEARCH_ROI[1]+edges[i], width, edges[i+1]-edges[i])
                    for i in range(len(labels)))
    checks = dict(strip_score=strip_score >= THRESHOLDS['strip_score'],
                  row_min=row_min >= THRESHOLDS['row_min'], row_mean=row_mean >= THRESHOLDS['row_mean'],
                  unique_highlight=unique_highlight)
    valid = all(checks.values())
    return dict(score=score, valid=valid, candidate_floor=floor, labels=list(labels),
                           selected_index=selected, strip_score=float(strip_score), row_scores=rows,
                           row_mean=row_mean, row_min=row_min, background_medians=background,
                           highlight_gap=background[selected]-other_background, unique_highlight=unique_highlight,
                           checks=checks, failed_checks=[name for name, passed in checks.items() if not passed],
                           location_in_search=[x, y], template_size=[width, height],
                           scale_in_search=[width / BUTTON_WIDTH, height / template.shape[0]],
                           buttons=[list(rect) for rect in buttons])


def _match_control(gray, input_shape, labels, selected, floor, template):
    sizes = _candidate_sizes(input_shape, template.shape)
    candidates = {}
    # Coarse geometry first, then refine its vertical pitch. Bound the work to
    # the small UI crop rather than exhaustively matching every pixel size.
    for _ in range(2):
        for width, height in sizes:
            if (width, height) not in candidates:
                candidates[width, height] = _measure_control(
                    gray, labels, selected, floor, template, width, height)
        best = max(candidates.values(), key=lambda item: item['score'])
        width, height = best['template_size']
        sizes = [(width, round(height * (1 + dy / 100))) for dy in range(-3, 4)
                 if 0 < round(height * (1 + dy / 100)) <= gray.shape[0]]
    return _check_labels(gray, best, labels, selected, template)


def _check_labels(gray, best, labels, selected, template):
    x, y = best['location_in_search']
    width, height = best['template_size']
    patch = gray[y:y+height, x:x+width]
    if patch.shape != template.shape:
        patch = cv2.resize(patch, (BUTTON_WIDTH, template.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Similar numerals (notably 3 and 4) can exceed an absolute correlation
    # threshold in the wrong order. Check label identity as well as similarity.
    label_scores = [[max(float(cv2.matchTemplate(patch[i*115+22:i*115+90, 18:70], digit,
                                                  cv2.TM_CCOEFF_NORMED)[0, 0]) for digit in variants)
                     for variants in _digit_templates(labels, i == selected)] for i in range(len(labels))]
    label_winners = [int(np.argmax(scores)) for scores in label_scores]
    best['label_scores'] = label_scores
    best['label_winners'] = [labels[i] for i in label_winners]
    best['checks']['label_order'] = label_winners == list(range(len(labels)))
    best['failed_checks'] = [name for name, passed in best['checks'].items() if not passed]
    best['valid'] = all(best['checks'].values())
    return best


def detect_floor(control_bgr, *, control_labels=None):
    """Locate a complete control in the crop, then check every label and highlight.

    Position and bounded UI size adaptation do not change map registration or
    acceptance thresholds. Scores are similarities, not probabilities.
    """
    if control_bgr.ndim != 3 or control_bgr.shape[2] != 3 or not control_bgr.size:
        raise ValueError('楼层 UI 截图必须为非空 BGR 图片')
    gray = cv2.cvtColor(cv2.resize(control_bgr, SEARCH_ROI[2:], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    kinds = CONTROL_KINDS if control_labels is None else (tuple(control_labels),)
    templates = [item for item in _templates() if item[0] in kinds]
    if not templates:
        raise ValueError('未提供此楼层控件样式的模板')
    # Try the existing size estimate once per state, sharing the estimate within
    # a control kind. Every state still gets its label/highlight/margin checks.
    # Uncertain geometry falls back to the unchanged finite size search.
    hints = {}
    for labels, _, _, template in templates:
        if labels not in hints:
            hints[labels] = tuple(round(v) for v in _size_hint(control_bgr.shape, template.shape))
    candidates = []
    for labels, selected, floor, template in templates:
        width, height = hints[labels]
        if 0 < width <= gray.shape[1] and 0 < height <= gray.shape[0]:
            candidate = _measure_control(gray, labels, selected, floor, template, width, height)
            candidates.append(_check_labels(gray, candidate, labels, selected, template))
    candidates.sort(key=lambda item: item['score'], reverse=True)
    search_path = 'size_hint'
    if (len(candidates) != len(templates) or not candidates[0]['valid']
            or candidates[0]['score']-candidates[1]['score'] < THRESHOLDS['margin']):
        candidates = [_match_control(gray, control_bgr.shape, labels, selected, floor, template)
                      for labels, selected, floor, template in templates]
        candidates.sort(key=lambda item: item['score'], reverse=True)
        search_path = 'size_search'
    best = candidates[0]
    margin = float(best['score'] - candidates[1]['score'])
    raw = control_bgr.copy()
    raw.flags.writeable = gray.flags.writeable = False
    details = dict(score=best['score'], margin=margin, candidates=tuple(candidates),
                   control_image=raw, normalized_image=gray, search_path=search_path)
    if not best['valid'] or margin < THRESHOLDS['margin']:
        return FloorDetection(None, reason='楼层 UI 未找到唯一可信的选中项，请保持地图打开后重试', **details)
    return FloorDetection(best['candidate_floor'], tuple(tuple(rect) for rect in best['buttons']), **details)
