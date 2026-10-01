"""Screenshot-only detection of smooth, dim corridor-color fog fringes."""
from dataclasses import replace

import cv2
import numpy as np

from .color_segmenter import edges_from_map_mask


def corridor_fog_mask(image, segmentation):
    """Return uncertain pixels, not empty space; never infer from a template.

    Require a brighter corridor reference, a plausible background/material
    mixture, and diffuse boundaries. Crisp narrow passages remain protected.
    """
    out = np.zeros(image.shape[:2], np.uint8)
    x, y, w, h = cv2.boundingRect(segmentation.map_mask)
    if w == 0 or h == 0:
        return out
    x0, y0 = max(0, x-16), max(0, y-16)
    x1, y1 = min(image.shape[1], x+w+16), min(image.shape[0], y+h+16)
    area = np.s_[y0:y1, x0:x1]
    rgb = image[area].astype(np.float32)
    occupied = segmentation.map_mask[area]
    corridor = segmentation.corridor_mask[area]
    background = cv2.dilate(occupied, np.ones((13, 13), np.uint8)) == 0
    core = cv2.erode(corridor, np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(background) < 80 or np.count_nonzero(core) < 80:
        return out
    bg = np.median(rgb[background], axis=0)
    values = rgb[core]
    bright = values.max(axis=1) >= np.percentile(values.max(axis=1), 65)
    prototype = np.median(values[bright], axis=0)
    vector = prototype-bg
    if np.linalg.norm(vector) < 35:
        return out
    alpha = np.sum((rgb-bg)*vector, axis=2) / np.sum(vector*vector)
    error = np.linalg.norm(rgb-bg-alpha[:, :, None]*vector, axis=2)
    # Dimness alone is not enough: a real corridor can also be fogged. Only
    # suppress pixels better explained by the other material (room) plus fog.
    fog = np.zeros_like(occupied, dtype=bool)
    room_core = cv2.erode(segmentation.room_mask[area], np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(room_core) >= 80:
        room_values = rgb[room_core]
        room_bright = room_values.max(axis=1) >= np.percentile(room_values.max(axis=1), 65)
        room_vector = np.median(room_values[room_bright], axis=0)-bg
        if np.linalg.norm(room_vector) >= 35:
            room_alpha = np.sum((rgb-bg)*room_vector, axis=2) / np.sum(room_vector*room_vector)
            room_error = np.linalg.norm(rgb-bg-room_alpha[:, :, None]*room_vector, axis=2)
            # Fogged brown rooms can cross the blue corridor HSV threshold.
            fog = ((room_alpha >= .15) & (room_alpha < .55) & (room_error < 7)
                    & (room_error+2 < error))
    candidate = (occupied > 0) & (corridor > 0) & fog
    if not np.any(candidate):
        return out
    gray = cv2.cvtColor(image[area], cv2.COLOR_BGR2GRAY)
    # Detect abrupt color/luminance boundaries independently of the binary mask.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    crisp = cv2.dilate(np.uint8(np.hypot(gx, gy) >= 5)*255, np.ones((7, 7), np.uint8)) > 0
    # Reject isolated color noise, and abstain if the model questions most of
    # the observation. This is fringe filtering, not wholesale resegmentation.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(candidate))
    keep = np.zeros_like(occupied)
    for index, (_, _, _, _, pixels) in enumerate(stats[1:], 1):
        if pixels < 24:
            continue
        component = labels == index
        # Keep the entire narrow passage when its sides have strong image
        # edges; removing only its center would turn a real corridor hollow.
        if np.mean(crisp[component]) >= .4:
            continue
        keep[component & ~crisp] = 255
    if cv2.countNonZero(keep) > .15 * cv2.countNonZero(occupied):
        return out
    out[area] = keep
    return out


def filter_corridor_fog(image, segmentation, config):
    """Suppress only demonstrably uncertain foreground before coarse search."""
    uncertain = corridor_fog_mask(image, segmentation)
    if not np.any(uncertain):
        return segmentation
    corridor = segmentation.corridor_mask.copy()
    corridor[uncertain > 0] = 0
    mask = (corridor | segmentation.room_mask) & segmentation.visible_mask
    prep = config.get('preprocess', {})
    edge = edges_from_map_mask(mask, int(prep.get('edge_canny1', 45)), int(prep.get('edge_canny2', 135)),
        mode=str(prep.get('edge_mode', 'canny')),
        cleanup_open_kernel=int(prep.get('edge_cleanup_open_kernel', 3)),
        cleanup_close_kernel=int(prep.get('edge_cleanup_close_kernel', 5)),
        semantic_masks=(corridor, segmentation.room_mask),
        cleanup_min_component_area=int(prep.get('edge_cleanup_min_component_area', 24)),
        cleanup_max_hole_area=int(prep.get('edge_cleanup_max_hole_area', 320)))
    # The artificial cut around unknown fog is not a new architectural wall.
    edge[cv2.dilate(uncertain, np.ones((5, 5), np.uint8)) > 0] = 0
    known = (np.zeros_like(mask) if segmentation.known_background_mask is None
             else segmentation.known_background_mask.copy())
    known[uncertain > 0] = 0
    ignored = segmentation.ignore_mask
    if ignored is None:
        ignored = np.zeros_like(mask)
    valid = mask | known
    return replace(segmentation, corridor_mask=corridor, map_mask=mask, edge_mask=edge,
        known_background_mask=known, valid_mask=valid, unknown_mask=~(valid | ignored),
        uncertain_mask=uncertain)
