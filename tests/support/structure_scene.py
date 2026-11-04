"""Publish test-owned clean images using the current source-image contract."""
from pathlib import Path
import shutil

from tests.support.aligned_runtime import difficulty, publish


def publish_structure(root, source, *, scale=1.5, floors=('2nd', '1st', 'b1'), overrides=None):
    root, source = Path(root), Path(source)
    definition = difficulty('hard', '困难', floors, {'clean': '底图'})
    clean = root/definition['clean_root']
    paths = list(source.glob('layout_*/*.png'))
    layout_floors = {}
    for path in paths:
        if path.stem.rsplit('_', 1)[-1] not in floors:
            continue
        layout_floors.setdefault(path.parent.name, set()).add(path.stem.rsplit('_', 1)[-1])
        target = clean/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != path:
            shutil.copyfile(path, target)
    definition['layout_floors'] = {layout: [floor for floor in floors if floor in values]
                                  for layout, values in layout_floors.items()}
    definition['floors'] = [floor for floor in floors if any(floor in values for values in layout_floors.values())]
    catalog = dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='测试地图', difficulties=[definition])])
    publish(root, catalog, scales={'maps': {'a_bringer_of_doom': {'hard': dict(
        minimum_scale=scale, layout_minimum_scales=dict(overrides or {}))}}})
    return clean, catalog
