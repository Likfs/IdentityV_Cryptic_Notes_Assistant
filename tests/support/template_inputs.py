"""In-memory single-floor sources for GUI delivery tests."""
from pathlib import Path
from src.registration.template_matching import floors


def install_floor_inputs(monkeypatch, entry, image, rgba):
    records = {floor: dict(entry) for floor in floors.FLOORS}
    monkeypatch.setattr(floors, 'sources', lambda *a, **kw: (records, {'perf_mode': 2}))
    monkeypatch.setattr(floors, 'FrozenSourceSnapshot', floors.SourceSnapshot)
    monkeypatch.setattr(floors, 'load', lambda *a, **kw:
                        (records[a[3]], Path(entry['source_image']), {'perf_mode': 2}, image, rgba))
    monkeypatch.setattr(floors, '_load_verified', lambda item, config, **kw:
                        (item, Path(entry['source_image']), config, image, rgba))
