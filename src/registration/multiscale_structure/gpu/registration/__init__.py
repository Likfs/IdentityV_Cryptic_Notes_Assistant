"""Runtime clean map registration for route projection."""

from .runtime_registration import (
    PreparedRegistrationTemplate,
    RuntimeRegistrationResult,
    prepare_registration_template,
    register_screenshot,
    validate_route_overlay,
)

__all__ = [
    "PreparedRegistrationTemplate",
    "RuntimeRegistrationResult",
    "prepare_registration_template",
    "register_screenshot",
    "validate_route_overlay",
]
