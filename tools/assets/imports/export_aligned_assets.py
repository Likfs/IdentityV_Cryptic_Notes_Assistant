"""Offline entry to the shared same-canvas resource audit."""
from pathlib import Path

from src.aligned_assets import generate as generate_assets
from src.registration.template_matching.resources import verify_module_provenance


def generate(root):
    root = Path(root).resolve()
    verify_module_provenance(root/'src/registration/template_matching')
    result = generate_assets(root)
    verify_module_provenance(root/'src/registration/template_matching')
    return result
