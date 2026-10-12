"""Fixed-scale registration results in original-source-image coordinates."""
from dataclasses import dataclass
import numpy as np


@dataclass
class OriginalRegistrationResult:
    accepted: bool
    rejection_reasons: list[str]
    validation: dict
    timings: dict
    original_to_screen: np.ndarray | None = None
    screenshot_to_original: np.ndarray | None = None

    def diagnostics(self) -> dict:
        return {
            'accepted': self.accepted,
            'rejection_reasons': self.rejection_reasons,
            'coordinate_space': 'original_source_image',
            'original_to_screen': self.original_to_screen.tolist() if self.original_to_screen is not None else None,
            'screenshot_to_original': self.screenshot_to_original.tolist() if self.screenshot_to_original is not None else None,
            'validation': self.validation,
            'timings_ms': self.timings,
            # Wire compatibility for existing diagnostic readers; never populated with original coordinates.
            'clean_to_screen': None, 'screenshot_to_clean': None,
            'observation_bbox': None, 'match': None,
        }
