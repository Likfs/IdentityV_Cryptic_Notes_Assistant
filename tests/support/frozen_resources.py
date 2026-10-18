"""Model a fresh template subsystem; integration tests also exercise real processes."""
from src.registration.template_matching import floors, matcher, resources


def restart_template_resources():
    floors._close_fixed_session()
    floors._config_cache.clear()
    floors._images.clear()
    matcher._map_cache.clear()
    resources.verified_fingerprints.clear()
