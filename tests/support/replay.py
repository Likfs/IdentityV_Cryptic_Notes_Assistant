"""Reproducible image inputs. Expected labels come from source provenance, not search output."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tests/fixtures/retrieval/cases.json'


def read_cases():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))['cases']


def replay_image(case):
    source = ROOT / case['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case['sha256'], source
    image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    if 'crop_xywh' in case:
        x, y, w, h = case['crop_xywh']
        image = image[y:y+h, x:x+w].copy()
        assert image.shape[:2] == (h, w)
    if 'erase_xywh' in case:
        x, y, w, h = case['erase_xywh']
        image[y:y+h, x:x+w] = (24, 32, 39)
    width, height = case['reference_size']
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image
