"""Screenshot-owned evidence and bounded joint (x, y, zoom) verification.

Unknown colors never become negative evidence. Wall reliability is decided
from the screenshot before looking at a template or any proposed transform.
All error distances and perturbation budgets below use screenshot pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .sparse_pose_refiner import balanced_points


@dataclass
class CredibleEvidence:
    walls: np.ndarray  # Nx4 endpoints in observation coordinates
    directions: np.ndarray  # 0 horizontal, 1 vertical
    points: np.ndarray  # Nx12x2, equal votes per wall segment
    semantic: np.ndarray
    labels: np.ndarray
    background: np.ndarray
    trusted_masks: tuple[np.ndarray, np.ndarray]
    background_mask: np.ndarray
    center: np.ndarray
    diagnostics: dict


def group_wall_edges(walls, directions, image, occupied):
    """Pair the two edges of a thin bright stroke, never across a filled passage.

    Grouping is screenshot-only. Non-overlapping collinear pieces (door gaps)
    remain separate; groups have at most two members to avoid transitive merging.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    groups = []
    for index, wall in enumerate(walls):
        direction = directions[index]
        normal, tangent = 1-direction, direction
        for group in groups:
            if len(group) != 1 or directions[group[0]] != direction:
                continue
            other = walls[group[0]]
            lo, hi = max(wall[tangent], other[tangent]), min(wall[tangent+2], other[tangent+2])
            span = min(wall[tangent+2]-wall[tangent], other[tangent+2]-other[tangent])
            gap = abs(wall[normal]-other[normal])
            if not 1 <= gap <= 8 or hi-lo < .5*span:
                continue
            points = np.zeros((12, 2), np.float32)
            points[:, tangent] = np.linspace(lo,hi,12)
            points[:, normal] = (wall[normal]+other[normal])/2
            def sample(offset):
                p = points.copy(); p[:, normal] += offset
                return cv2.remap(gray,p[:,0][None],p[:,1][None],cv2.INTER_LINEAR)[0]
            # Walls have a thin luminous ridge; uniform floor strips do not.
            if np.median(sample(0)-np.maximum(sample(-gap/2-2),sample(gap/2+2))) < 3:
                continue
            group.append(index)
            break
        else:
            groups.append([index])
    return (np.asarray([np.mean(walls[g],axis=0) for g in groups],np.float32).reshape(-1,4),
            np.asarray([directions[g[0]] for g in groups],int), groups)


def screenshot_evidence(image: np.ndarray, observation, *, reliable=False) -> CredibleEvidence:
    """Model fog as a mixture with the local background, without template data."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = image.astype(np.float32)
    occupied = observation.map_mask > 0
    border = cv2.dilate(observation.map_mask, np.ones((13, 13), np.uint8)) == 0
    bg = np.median(rgb[border], axis=0) if np.any(border) else np.median(rgb.reshape(-1, 3), axis=0)
    prototypes = []
    core_masks = []
    for mask in (observation.corridor_mask, observation.room_mask):
        core = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        core_masks.append(core)
        values = rgb[core]
        if len(values) >= 40:
            bright = values.max(axis=1) >= np.percentile(values.max(axis=1), 65)
            prototypes.append(np.median(values[bright], axis=0))
        else:
            prototypes.append(None)
    errors, alphas = [], []
    for prototype in prototypes:
        if prototype is None or np.linalg.norm(prototype-bg) < 8:
            errors.append(np.full(occupied.shape, 1000, np.float32))
            alphas.append(np.zeros(occupied.shape, np.float32))
            continue
        vector = prototype-bg
        alpha = np.clip(np.sum((rgb-bg)*vector, axis=2)/np.sum(vector*vector), 0, 1.1)
        errors.append(np.linalg.norm(rgb-bg-alpha[:, :, None]*vector, axis=2))
        alphas.append(alpha)
    errors = np.stack(errors, axis=2)
    alphas = np.stack(alphas, axis=2)
    label = errors.argmin(axis=2)
    best = errors.min(axis=2)
    separation = np.abs(errors[:, :, 0]-errors[:, :, 1])
    alpha = np.take_along_axis(alphas, label[:, :, None], axis=2)[:, :, 0]
    trusted = occupied & (best <= 7) & (separation >= 2) & (alpha >= .55)
    # Do not relabel an ambiguous fringe as a confident opposite material.
    trusted_masks = [np.uint8(trusted & (label == i) & core_masks[i])*255 for i in (0, 1)]
    points, labels = [], []
    for i, mask in enumerate(trusted_masks):
        selected, _ = balanced_points(mask, 300)
        points.append(selected)
        labels.extend([i]*len(selected))
    semantic = np.concatenate(points)
    known = observation.known_background_mask
    background_mask = (np.zeros_like(observation.map_mask) if known is None else known.copy())
    # A plausible fogged material or a colorful icon is not known empty space.
    uncertain_material = (errors.min(axis=2) < 10) & (alphas.max(axis=2) > .20)
    background_mask[uncertain_material | (hsv[:, :, 1] > 100) | (hsv[:, :, 2] > 35)] = 0
    # Darkness around the explored patch is fog, not proof of empty map space.
    # Only enclosed, screenshot-supported pockets may contribute negatives.
    count, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(~occupied))
    enclosed = np.zeros_like(background_mask)
    for index, (bx, by, bw, bh, area) in enumerate(stats[1:], start=1):
        if 25 <= area <= 1200 and bx > 0 and by > 0 and bx+bw < occupied.shape[1] and by+bh < occupied.shape[0]:
            enclosed[components == index] = 255
    background_mask &= enclosed
    background_mask = cv2.erode(background_mask, np.ones((5, 5), np.uint8))
    background, _ = balanced_points(background_mask, 160)
    if len(background) < 24:
        background = np.empty((0, 2), np.float32)
        background_mask[:] = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), .55)
    edges = cv2.Canny(gray, 18, 45, L2gradient=True)
    edges[cv2.dilate(observation.map_mask, np.ones((11, 11), np.uint8)) == 0] = 0
    # Bright saturated markers are screenshot UI, not architectural walls.
    icons = np.uint8((hsv[:, :, 1] > 105) & (hsv[:, :, 2] > 135) & (best > 12))*255
    edges[cv2.dilate(icons, np.ones((5, 5), np.uint8)) > 0] = 0
    def filled(mask):
        result = mask.copy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(255-mask)
        for index, (x, y, w, h, area) in enumerate(stats[1:], start=1):
            if area <= 1200 and x > 0 and y > 0 and x+w < mask.shape[1] and y+h < mask.shape[0]:
                result[labels == index] = 255
        return result
    boundaries = cv2.Canny(filled(observation.room_mask), 40, 120) | cv2.Canny(filled(observation.corridor_mask), 40, 120)
    boundary_distance = cv2.distanceTransform(255-boundaries, cv2.DIST_L2, 3)
    walls, directions = [], []
    for direction, kernel in enumerate((np.ones((1, 16), np.uint8), np.ones((16, 1), np.uint8))):
        straight = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(straight)
        for x, y, w, h, area in stats[1:]:
            length, thickness = (w, h) if direction == 0 else (h, w)
            if length < 18 or thickness > 4:
                continue
            if direction == 0:
                segment = [x, y+(h-1)/2, x+w-1, y+(h-1)/2]
            else:
                segment = [x+(w-1)/2, y, x+(w-1)/2, y+h-1]
            probe = np.linspace(segment[:2], segment[2:], 12).astype(int)
            if np.mean(boundary_distance[probe[:, 1], probe[:, 0]] <= 4) < .7:
                continue
            walls.append(segment)
            directions.append(direction)
    # Bound work while retaining both orientations and spatially separated lines.
    walls = np.asarray(walls, np.float32).reshape(-1, 4)
    directions = np.asarray(directions, int)
    raw_count = len(walls)
    if reliable:
        walls, directions, _ = group_wall_edges(walls,directions,image,occupied)
    if len(walls) > 48:
        chosen = np.linspace(0, len(walls)-1, 48, dtype=int)
        walls, directions = walls[chosen], directions[chosen]
    fractions = np.linspace(.08, .92, 12, dtype=np.float32)
    samples = walls[:, None, :2]*(1-fractions[None, :, None]) + walls[:, None, 2:]*fractions[None, :, None]
    yy, xx = np.nonzero(observation.map_mask)
    center = np.float32([xx.mean(), yy.mean()]) if len(xx) else np.float32([0, 0])
    return CredibleEvidence(walls, directions, samples, semantic, np.asarray(labels), background,
        tuple(trusted_masks), background_mask, center,
        {"wall_segments": len(walls), "raw_wall_segments": raw_count, "reliable_wall_groups": reliable,
         "horizontal_segments": int(np.sum(directions == 0)),
         "vertical_segments": int(np.sum(directions == 1)), "trusted_semantic_pixels": sum(cv2.countNonZero(m) for m in trusted_masks),
         "ambiguous_semantic_pixels": int(np.count_nonzero(occupied & ~trusted)),
         "trusted_background_samples": len(background), "prototypes": [p.tolist() if p is not None else None for p in prototypes]})


def directional_walls(template, reliable=False):
    edge = cv2.Canny(template.room_mask,40,120)|cv2.Canny(template.corridor_mask,40,120)
    if not reliable:
        return [cv2.morphologyEx(edge,cv2.MORPH_OPEN,k) for k in
                (np.ones((1,10),np.uint8),np.ones((10,1),np.uint8))]
    result = []
    for direction in (0,1):
        # Three-pixel runs retain staircase edges. Require nearby tangential
        # support over a longer span; keep only original pixels, not bridged gaps.
        short = cv2.morphologyEx(edge,cv2.MORPH_OPEN,
            np.ones((1,3) if direction==0 else (3,1),np.uint8))
        support = cv2.boxFilter(np.float32(short>0),-1,(15,3) if direction==0 else (3,15),normalize=False)
        result.append(np.uint8((short>0)&(support>=6))*255)
    return result


def _fields(template, reliable=False):
    key = "_reliable_fields" if reliable else "_credible_fields"
    fields = getattr(template, key, None)
    if fields is None:
        walls = directional_walls(template,reliable)
        distances = np.stack([cv2.distanceTransform(255-w, cv2.DIST_L2, 3) for w in walls], axis=2)
        semantic = np.stack([cv2.dilate(m, np.ones((3, 3), np.uint8)) for m in
                            (template.corridor_mask, template.room_mask)], axis=2).astype(np.float32)/255
        fields = distances, semantic
        setattr(template,key,fields)
    return fields


def _line_fit(template, evidence, pose, *, omitted=-1):
    """Robust normal constraints for several separate parallel wall segments."""
    reliable = evidence.diagnostics.get("reliable_wall_groups",False)
    key = "_reliable_nearest" if reliable else "_credible_nearest"
    nearest = getattr(template, key, None)
    if nearest is None:
        nearest = []
        for walls in directional_walls(template,reliable):
            _, labels = cv2.distanceTransformWithLabels(255-walls, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            yy, xx = np.nonzero(walls)
            lookup = np.zeros((int(labels.max())+1, 2), np.float32)
            lookup[labels[yy, xx]] = np.column_stack((xx, yy))
            nearest.append(lookup[labels])
        setattr(template,key,nearest)
    points = evidence.points.reshape(-1, 2)-evidence.center
    directions = np.repeat(evidence.directions, 12)
    segment_ids = np.repeat(np.arange(len(evidence.walls)), 12)
    axis = 1-directions
    design = np.zeros((len(points), 3), np.float64)
    design[np.arange(len(points)), axis] = 1
    design[:, 2] = points[np.arange(len(points)), axis]
    updated = pose.copy()
    for _ in range(5):
        xy = points*updated[2]+updated[:2]
        targets = np.zeros_like(xy)
        for direction in (0, 1):
            chosen = directions == direction
            targets[chosen] = cv2.remap(nearest[direction], xy[chosen,0].reshape(1,-1),
                xy[chosen,1].reshape(1,-1),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=(-1000,-1000))[0]
        target = targets[np.arange(len(points)),axis]
        residual = np.linalg.norm(targets-xy,axis=1)/updated[2]
        weights = np.minimum(1.,1.5/np.maximum(residual,.01))
        weights[(residual>8)|(segment_ids==omitted)] = 0
        if np.sum(weights>0) < 36 or np.linalg.matrix_rank(design[weights>0]) < 3:
            break
        weighted = design*np.sqrt(weights[:,None])
        proposal = np.linalg.lstsq(weighted,target*np.sqrt(weights),rcond=None)[0]
        if (not np.isfinite(proposal).all() or abs(proposal[2]/pose[2]-1)>.06
                or np.linalg.norm(proposal[:2]-pose[:2])/pose[2]>12):
            break
        updated=proposal.astype(np.float32)
    return updated


def _sample(field, points, poses, center, border=0):
    xy = (points[None]-center)*poses[:, None, 2:3]+poses[:, None, :2]
    return cv2.remap(field, np.ascontiguousarray(xy[:, :, 0]), np.ascontiguousarray(xy[:, :, 1]),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(border, border))


def score_joint(template, evidence, poses):
    fields, semantic_field = _fields(template,evidence.diagnostics.get("reliable_wall_groups",False))
    count = len(poses)
    wall_values = _sample(fields, evidence.points.reshape(-1, 2), poses, evidence.center, 1000)
    channels = np.repeat(evidence.directions, 12)
    distances = wall_values[:, np.arange(len(channels)), channels]/np.maximum(poses[:, 2:3], .01)
    distances = distances.reshape(count, -1, 12)
    per_wall = np.exp(-distances/2.5).mean(axis=2)
    wall_score = per_wall.mean(axis=1) if per_wall.shape[1] else np.zeros(count)
    semantic = np.zeros(count)
    if len(evidence.semantic):
        values = _sample(semantic_field, evidence.semantic, poses, evidence.center)
        semantic = values[:, np.arange(len(evidence.semantic)), evidence.labels].mean(axis=1)
    negative = np.zeros(count)
    if len(evidence.background):
        negative = _sample(semantic_field, evidence.background, poses, evidence.center).max(axis=2).mean(axis=1)
    return .75*wall_score + .25*semantic - .10*negative, distances, semantic, negative


def pose_matrix(pose, center):
    x, y, s = pose
    return np.float32([[s, 0, x-s*center[0]], [0, s, y-s*center[1]]])


def corridor_structure_signature(walls, directions, supported):
    """A single-material corridor can have an asymmetric, identifiable outline.

    Require >=3 separated levels in one direction and >=2 in the other. Their
    unequal spacing distinguishes a corridor with side openings from a plain
    rectangle or uniformly repeated stair strokes. This only adds evidence;
    candidate uniqueness and geometric stability are still checked separately.
    """
    positions = []
    for direction, coordinate in ((0, 1), (1, 0)):
        kept = []
        for value in sorted(walls[supported & (directions == direction), coordinate]):
            if not kept or value-kept[-1] >= 8:
                kept.append(float(value))
        positions.append(kept)
    distinctive = False
    for levels, across in (positions, positions[::-1]):
        if len(levels) < 3 or len(across) < 2 or np.ptp(across) < 25:
            continue
        gaps = np.diff(levels)
        if gaps.max()-gaps.min() >= 16 and gaps.max() >= 1.5*gaps.min():
            distinctive = True
    return {'distinctive': bool(distinctive), 'normal_positions': positions}


def review_geometry(template, evidence, seeds, *, scale_min, scale_max, viewport, origin,
                    allow_corridor_structure=False):
    """Refine all selected scale/location branches under the same objective."""
    started = perf_counter()
    result = {**evidence.diagnostics, "position_status": "insufficient_evidence", "scale_status": "insufficient_walls", "accepted": False}
    if min(np.sum(evidence.directions == 0), np.sum(evidence.directions == 1)) < 2 or len(evidence.semantic) < 40:
        return None, {**result, "elapsed_ms": (perf_counter()-started)*1000}
    center = evidence.center
    unique = []
    for matrix in seeds:
        s = float(matrix[0, 0])
        anchor = matrix[:, :2]@center+matrix[:, 2]
        if s <= 0 or not scale_min <= s <= scale_max:
            continue
        if any(np.linalg.norm(anchor-p[:2])/s < 12 and abs(np.log(s/p[2])) < .045 for p in unique):
            continue
        unique.append(np.float32([*anchor, s]))
    if not unique:
        return None, {**result, "elapsed_ms": (perf_counter()-started)*1000}
    initial = np.asarray(unique)
    # Rank proposals using this objective, then reserve distinct spatial/scale modes.
    initial_scores = score_joint(template, evidence, initial)[0]
    order = np.argsort(-initial_scores)
    selected = []
    for index in order:
        p = initial[index]
        if not any(np.linalg.norm(p[:2]-initial[j,:2])/p[2] < 30
                   and abs(np.log(p[2]/initial[j,2])) < .10 for j in selected):
            selected.append(index)
        if len(selected) == 6:
            break
    selected.extend(index for index in order if index not in selected)
    selected = selected[:6]
    poses = initial[selected].copy()
    initial = poses.copy()
    offsets = np.float32([(dx, dy, ds) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for ds in (-1, 0, 1)])
    for translation, zoom in ((8, .04), (4, .02), (2, .01), (1, .005), (.5, .0025)):
        for _ in range(2):
            steps = np.column_stack((poses[:, 2]*translation, poses[:, 2]*translation, poses[:, 2]*zoom))
            trials = poses[:, None]+offsets[None]*steps[:, None]
            trials[:, :, :2] = np.clip(trials[:, :, :2], initial[:, None, :2]-24*initial[:, None, 2:3],
                                      initial[:, None, :2]+24*initial[:, None, 2:3])
            trials[:, :, 2] = np.clip(trials[:, :, 2], np.maximum(scale_min, initial[:, None, 2]*.86),
                                     np.minimum(scale_max, initial[:, None, 2]*1.14))
            scores = score_joint(template, evidence, trials.reshape(-1, 3))[0].reshape(len(poses), -1)
            poses = trials[np.arange(len(poses)), scores.argmax(axis=1)]
    fitted = np.asarray([_line_fit(template,evidence,p) for p in poses])
    in_bounds = (fitted[:, 2] >= scale_min) & (fitted[:, 2] <= scale_max)
    poses[in_bounds] = fitted[in_bounds]
    scores, distances, semantic, negative = score_joint(template, evidence, poses)
    best = int(scores.argmax())
    winner = poses[best]
    competitors = [i for i in range(len(poses)) if i != best
        and (np.linalg.norm(poses[i, :2]-winner[:2])/winner[2] > 20 or abs(np.log(poses[i, 2]/winner[2])) > .04)
        and scores[i] >= scores[best]-.06]
    # Joint perturbations expose translation/zoom coupling; include grid resolution.
    deltas = np.float32([(dx, dy, ds) for ds in np.linspace(-.025, .025, 21)
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    probes = winner + deltas*np.float32([winner[2], winner[2], winner[2]])
    probe_scores = score_joint(template, evidence, probes)[0]
    plausible = probes[probe_scores >= max(probe_scores.max()-.012, scores[best]-.012)]
    height, width = viewport
    xx, yy = np.meshgrid(np.linspace(0, width-1, 5), np.linspace(0, height-1, 5))
    screen = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)-origin
    clean = (screen-center)*winner[2]+winner[:2]
    differences = [float(np.percentile(np.linalg.norm((clean-p[:2])/p[2]+center-screen, axis=1),95)) for p in plausible]
    # Jackknife independent wall constraints; a single segment must not pin zoom.
    holdouts = np.asarray([_line_fit(template, evidence, winner, omitted=i) for i in range(len(evidence.walls))])
    holdout_errors = [float(np.percentile(np.linalg.norm((clean-p[:2])/p[2]+center-screen, axis=1),95)) for p in holdouts]
    boundary = bool(len(plausible) and np.max(np.abs(plausible[:, 2]/winner[2]-1)) >= .0249)
    wall_medians = np.median(distances[best], axis=1)
    reliable = evidence.diagnostics.get("reliable_wall_groups",False)
    # Half-pixel quantization allowance; retain continuous support as well.
    supported = wall_medians <= (3.5 if reliable else 3)
    horizontal = evidence.directions == 0
    vertical = ~horizontal
    orientation_support = [int(np.sum(supported & mask)) for mask in (horizontal, vertical)]
    # Support at separate parallel walls constrains scale, unlike one line.
    h = evidence.walls[supported & horizontal, 1]
    v = evidence.walls[supported & vertical, 0]
    distributed = len(h) >= 2 and len(v) >= 2 and np.ptp(h) >= 25 and np.ptp(v) >= 25
    def separate_normals(values):
        kept = []
        for value in sorted(values):
            if not kept or value-kept[-1] >= 8:
                kept.append(value)
        return len(kept)
    # A cropped, single-colour rectangle can fit many rooms at a different
    # zoom. Good residuals alone do not make its absolute location identifiable.
    complexity = (min(np.count_nonzero(evidence.labels == i) for i in (0, 1)) >= 40
                  or min(separate_normals(h), separate_normals(v)) >= 3)
    corridor_signature = corridor_structure_signature(evidence.walls, evidence.directions, supported)
    other_scores = [float(scores[i]) for i in range(len(poses)) if i != best
        and (np.linalg.norm(poses[i, :2]-winner[:2])/winner[2] > 20
             or abs(np.log(poses[i, 2]/winner[2])) > .04)]
    corridor_margin = float(scores[best])-max(other_scores, default=-1.)
    corridor_confirmed = (allow_corridor_structure and reliable and bool(other_scores)
                          and corridor_signature['distinctive'] and corridor_margin >= .10
                          and semantic[best] >= .98)
    complexity = complexity or corridor_confirmed
    position = complexity and not competitors and semantic[best] >= .94 and negative[best] <= .12
    stable = (distributed and np.mean(supported) >= .75 and np.median(wall_medians) <= 1.5
              and max(differences, default=1000) <= 8 and max(holdout_errors, default=1000) <= 8 and not boundary)
    local_stable = False
    local_difference = local_holdout = 1000.
    if reliable:
        all_points = evidence.points.reshape(-1,2)
        low, high = all_points.min(axis=0),all_points.max(axis=0)
        xx, yy = np.meshgrid(np.linspace(low[0],high[0],5),np.linspace(low[1],high[1],5))
        local = np.column_stack((xx.ravel(),yy.ravel()))
        mapped = (local-center)*winner[2]+winner[:2]
        def deviation(poses):
            return max((float(np.max(np.linalg.norm((mapped-p[:2])/p[2]+center-local,axis=1))) for p in poses),default=1000.)
        local_difference,local_holdout = deviation(plausible),deviation(holdouts)
        # One half template pixel is magnified on zoomed screenshots. This is
        # a measurement quantization allowance, not a larger projection budget.
        median_budget = 1.5 + .5/max(float(winner[2]),.25)
        quality = (position and distributed and np.mean(supported)>=.75
                   and np.median(wall_medians)<=median_budget and semantic[best]>=.98 and not boundary)
        stable = quality and max(differences,default=1000)<=8 and max(holdout_errors,default=1000)<=10
        # Actual rendered pixels are individually clipped to <=4 px below;
        # allow a slightly larger bounding rectangle to be trimmed at its edges.
        local_stable = quality and local_difference<=6 and local_holdout<=4
    result.update(position_status="confirmed" if position else "ambiguous" if competitors else "insufficient_evidence",
        scale_status="confirmed" if stable else "uncertain", accepted=bool(position and stable),
        semantic_precision=float(semantic[best]), background_conflict=float(negative[best]),
        wall_median_screen_px=float(np.median(wall_medians)), wall_p90_screen_px=float(np.percentile(wall_medians,90)),
        wall_p95_screen_px=float(np.percentile(wall_medians,95)),
        leave_one_wall_out_p95_px=max(holdout_errors,default=None),
        wall_support_fraction=float(np.mean(supported)), orientation_support=orientation_support,
        viewport_disagreement_p95_px=max(differences,default=None), boundary=boundary,
        candidate_count=len(poses), competitive_branches=len(competitors), score=float(scores[best]),
        distinctive_structure=bool(complexity),
        corridor_structure={**corridor_signature, 'score_margin': corridor_margin,
                            'confirmed': bool(corridor_confirmed)},
        branches=[{"anchor":p[:2].tolist(), "scale":float(p[2]), "score":float(s), "semantic":float(sm)} for p,s,sm in zip(poses,scores,semantic)],
        selected_scale=float(winner[2]), wall_residuals_screen_px=wall_medians.tolist(),
        alternative_local_poses=[pose_matrix(p,center).tolist() for p in plausible[::max(1,len(plausible)//8)]],
        elapsed_ms=(perf_counter()-started)*1000)
    if reliable:
        result.update(local_accepted=bool(local_stable), projection_scope="full" if stable else "local" if local_stable else "none",
            selected_local_pose=pose_matrix(winner,center).tolist(),
            local_disagreement_max_px=local_difference,local_holdout_max_px=local_holdout,
            wall_median_budget_screen_px=median_budget,
            local_bounds=[*low.tolist(),*high.tolist()],
            support_weight=float(np.mean(np.clip((4.5-wall_medians)/2,0,1))),
            # Include every probe/holdout when clipping the rendered layer.
            uncertainty_poses=[pose_matrix(p,center).tolist() for p in np.concatenate((plausible,holdouts))])
    return pose_matrix(winner, center), result
