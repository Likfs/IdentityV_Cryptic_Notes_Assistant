from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.registration_core.color_segmenter import SegmentationResult
from src.registration_core.local_structure_index import build_or_load_local_structure_index
from src.registration_core.local_structure_refiner import refine_local_structure
from src.registration_core.structural_features import FEATURE_NAMES, extract_structural_features
from src.registration_core.template_preprocessor import ProcessedTemplate
from src.registration.runtime_registration import _known_background_conflict


def _template(room: np.ndarray, corridor: np.ndarray) -> ProcessedTemplate:
    combined = cv2.bitwise_or(room, corridor)
    zeros = np.zeros_like(room)
    return ProcessedTemplate(
        map_id="synthetic_floor", image=np.zeros((*room.shape, 3), dtype=np.uint8),
        corridor_mask=corridor, room_mask=room, map_mask=combined,
        edge_mask=zeros, stair_mask=zeros, distance_transform=zeros.astype(np.float32),
        crop_offset=(0, 0), raw_path="synthetic.png",
    )


def test_local_structure_index_recalls_patch_and_rebuilds_after_asset_change(tmp_path: Path) -> None:
    cv2.setNumThreads(4)
    room = np.zeros((256, 256), dtype=np.uint8)
    corridor = np.zeros_like(room)
    room[44:92, 52:108] = 255
    corridor[90:120, 70:82] = 255
    corridor[108:120, 70:172] = 255
    corridor[108:186, 158:172] = 255
    template = _template(room, corridor)
    source = tmp_path / "source.png"
    source.write_bytes(b"first asset revision")
    index = build_or_load_local_structure_index(template, source, tmp_path / "cache")
    again = build_or_load_local_structure_index(template, source, tmp_path / "cache")
    assert np.array_equal(index.channels, again.channels)
    assert np.array_equal(index.feature_channels, again.feature_channels)
    assert index.source_sha256 == again.source_sha256
    assert index.feature_channels.shape[2] == len(FEATURE_NAMES)

    x, y, w, h = 46, 38, 140, 160
    observed_room = room[y:y+h, x:x+w]
    observed_corridor = corridor[y:y+h, x:x+w]
    observed_map = cv2.bitwise_or(observed_room, observed_corridor)
    observation = SegmentationResult(
        corridor_mask=observed_corridor, room_mask=observed_room,
        map_mask=observed_map, edge_mask=np.zeros_like(observed_map),
        stair_mask=np.zeros_like(observed_map), visible_mask=observed_map,
    )
    query = index.query(observation, scale_min=1.0, scale_max=1.0, scale_step=0.1)
    candidates = query.candidates
    assert candidates
    assert query.coarse_score >= 0.80
    assert query.fine_scale_count == 1
    assert query.coarse_ms > 0.0
    assert query.fine_ms > 0.0
    assert candidates[0].score > 0.94
    assert np.linalg.norm(np.subtract(candidates[0].position, (x, y))) <= 4.0

    skipped = index.query(observation, scale_min=1.0, scale_max=1.0,
                          scale_step=0.1, coarse_min_score=1.1)
    assert skipped.candidates
    assert skipped.candidates[0].source == "evidence_rescue"
    assert np.linalg.norm(np.subtract(skipped.candidates[0].position, (x, y))) <= 4.0
    assert skipped.fine_ms == 0.0
    assert skipped.feature_ms > 0.0
    assert skipped.coarse_margin_status == "not_evaluated_low_score"
    assert query.coarse_margin_status in ("measured", "no_competitor_found")

    source.write_bytes(b"second asset revision")
    rebuilt = build_or_load_local_structure_index(template, source, tmp_path / "cache")
    assert rebuilt.source_sha256 != index.source_sha256

    changed = _template(room.copy(), corridor.copy())
    changed.room_mask[180:208, 30:58] = 255
    changed_index = build_or_load_local_structure_index(changed, source, tmp_path / "cache")
    assert not np.array_equal(changed_index.channels, rebuilt.channels)


def test_structural_features_include_directions_corners_contacts_and_room_links() -> None:
    room = np.zeros((120, 120), dtype=np.uint8)
    corridor = np.zeros_like(room)
    room[20:52, 30:70] = 255
    corridor[50:90, 44:56] = 255
    corridor[72:84, 18:96] = 255
    features = extract_structural_features(room, corridor)
    counts = features.counts
    assert counts["horizontal_edge"] > 0
    assert counts["vertical_edge"] > 0
    assert sum(counts[name] for name in FEATURE_NAMES[2:6]) > 0
    assert counts["narrow_corridor"] > 0
    assert features.room_connections
    assert sum(features.room_connections[0]) >= 1


def test_local_refiner_recovers_position_near_template_border() -> None:
    room = np.zeros((100, 100), dtype=np.uint8)
    corridor = np.zeros_like(room)
    room[12:42, 4:35] = 255
    corridor[39:50, 18:75] = 255
    template = _template(room, corridor)
    x, y = -3, 8
    canvas_room = np.zeros((70, 75), np.uint8)
    canvas_corridor = np.zeros_like(canvas_room)
    canvas_room[:, 3:] = room[y:y + 70, :72]
    canvas_corridor[:, 3:] = corridor[y:y + 70, :72]
    observed_map = cv2.bitwise_or(canvas_room, canvas_corridor)
    observation = SegmentationResult(
        corridor_mask=canvas_corridor, room_mask=canvas_room,
        map_mask=observed_map, edge_mask=np.zeros_like(observed_map),
        stair_mask=np.zeros_like(observed_map), visible_mask=observed_map,
    )
    refined = refine_local_structure(template, observation,
                                     position=(x + 3, y - 4), scales=[1.0], radius=8)
    assert refined is not None
    assert refined.position == (x, y)
    assert refined.score > 0.95


def test_known_background_conflict_uses_crop_origin() -> None:
    room = np.zeros((40, 40), dtype=np.uint8)
    room[10:20, 10:20] = 255
    template = _template(room, np.zeros_like(room))
    known = np.full((10, 10), 255, dtype=np.uint8)
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    count, collision = _known_background_conflict(identity, known, (0, 0), template)
    assert count == 100 and collision == 0.0
    count, collision = _known_background_conflict(identity, known, (10, 10), template)
    assert count == 100 and collision == 1.0
