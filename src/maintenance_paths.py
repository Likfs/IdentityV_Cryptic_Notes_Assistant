"""Locations used by offline audits; runtime bundles do not load these files."""
from pathlib import Path

TEMPLATE_CONFIG = 'maintenance/configs/template_matching.json'


def profile_root(clean_root):
    """Mirror a canonical assets/maps/<map>/<difficulty>/clean in maintenance.

    Standalone tool fixtures may supply an arbitrary clean directory. Their
    explicitly supplied local evidence stays next to that directory.
    """
    clean_root = Path(clean_root).resolve()
    if (clean_root.name == 'clean' and len(clean_root.parents) >= 5
            and clean_root.parents[2].name == 'maps'
            and clean_root.parents[3].name == 'assets'):
        return clean_root.parents[4]/'maintenance/maps'/clean_root.parents[1].name/clean_root.parent.name
    return clean_root.parent


def evidence_path(clean_root, value):
    """Resolve profile-relative evidence or a bounded project-relative input."""
    base = profile_root(clean_root)
    canonical = base != Path(clean_root).resolve().parent
    root = base.parents[3] if canonical else base
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError('维护依赖必须使用相对路径')
    anchor = root if canonical and relative.parts[:1] in (('assets',), ('maintenance',)) else base
    path = (anchor/relative).resolve()
    path.relative_to(root)
    return path
