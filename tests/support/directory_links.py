"""Real directory links, including unprivileged Windows junctions, for resource tests."""
import os
from pathlib import Path


def create_directory_link(link, target):
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def remove_directory_link(link, *, root):
    # Validate the link's parent, not its target. Never recursively remove a target.
    Path(link).parent.resolve().relative_to(Path(root).resolve())
    if os.name == 'nt':
        if not link.is_junction():
            raise ValueError('Expected a test directory junction')
        os.rmdir(link)
    else:
        if not link.is_symlink():
            raise ValueError('Expected a test directory symlink')
        link.unlink()
