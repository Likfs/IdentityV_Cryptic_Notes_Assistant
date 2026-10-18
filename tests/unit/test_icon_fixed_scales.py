import cv2
import numpy as np
import pytest

from src.layout_retrieval import fallback_templates, fallback_fit, anchor_fit, target_maps, floor_masks, query_data


@pytest.mark.parametrize('scale',[3.18,1.31])
def test_icon_and_wall_matching_use_configured_source_scale(scale):
    image = np.zeros((240, 280, 3), np.uint8)
    cv2.rectangle(image, (25, 30), (240, 200), (120, 120, 120), -1)
    cv2.rectangle(image, (25, 30), (240, 200), (235, 235, 235), 3)
    cv2.line(image, (90, 30), (90, 125), (235, 235, 235), 3)
    source = cv2.resize(image, None, fx=scale, fy=scale)
    query = query_data(image)
    target = dict(walls=target_maps(source), floors=floor_masks(source), scales=(3.18,1.31),
                  icon=dict(confidence=0))
    templates = fallback_templates(image, query['box'], scales=target['scales'])
    fit = fallback_fit(templates, query['box'], source, target, dict(confidence=0), [])
    assert fit['scale'] == pytest.approx(scale)
    assert fit['wall_score'] > .8
    qi = dict(cx=50, cy=50, confidence=1)
    # cv2.resize maps pixel centers by (x + .5)*scale - .5.
    ti = dict(cx=50.5*scale-.5, cy=50.5*scale-.5, confidence=1)
    fit = anchor_fit(query, target, qi, ti)
    assert fit['scale'] == pytest.approx(scale)
    assert fit['wall_score'] > .8
