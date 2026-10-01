"""Component-tolerance entry for isolated search-core regression."""
from dataclasses import replace
from .engine import register
from .template import prepared_adapter


def register_screenshot(screenshot, prepared, prior_screenshot_to_clean=None,
                        temporal_evidence=None, soft_pose_hint=None):
    template, lock = prepared_adapter(prepared)
    # Geometry lazily caches template fields. Serialize users of one template.
    with lock:
        result = register(screenshot, template)
    return replace(result, validation={**result.validation,
        'registration_algorithm': 'component_tolerance',
        'route_validation_enabled': False,
        'trusted_history_eligible': False,
        'history_policy': 'independent_frame_no_history_read_or_write'})
