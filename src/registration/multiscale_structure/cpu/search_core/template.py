"""Adapt the application's content-addressed template to component_tolerance search."""
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import cv2
import numpy as np
from src.registration_core.template_preprocessor import ProcessedTemplate


@dataclass
class Template:
    layout: str
    floor: str
    path: Path
    processed: ProcessedTemplate
    registration_scale: float
    coarse_channels: np.ndarray


_initialization_lock = RLock()


def prepared_adapter(prepared):
    # The owner already keys preparation by source fingerprint/layout/floor.
    # Cache on that exact object, not on a global layout or path key.
    with _initialization_lock:
        if not hasattr(prepared, '_structure_template'):
            p = prepared.processed
            channels = np.stack((p.corridor_mask.astype(np.float32)/255,
                                 p.room_mask.astype(np.float32)/255), axis=2)
            prepared._structure_template = Template(prepared.layout, prepared.floor, prepared.path,
                p, prepared.registration_scale,
                cv2.resize(channels, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA))
            prepared._structure_lock = RLock()
        return prepared._structure_template, prepared._structure_lock
