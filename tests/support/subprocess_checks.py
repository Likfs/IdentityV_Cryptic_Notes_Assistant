"""Assertions shared by source and explicit release acceptance."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run_probe(python, mode, output, *, pending_profiles=(), ready_floors=None):
    completed = subprocess.run([str(python), '-B', str(ROOT/'tests/support/runtime_probe.py'),
        '--root', str(ROOT), '--mode', mode, '--output', str(output)],
        cwd=output.parent, capture_output=True, encoding='utf-8', timeout=120)
    assert completed.returncode == 0, f'Exit {completed.returncode}: {completed.stdout}\n{completed.stderr}'
    payload = json.loads(completed.stdout)
    if mode == 'check':
        assert payload['layout_recognition_backend'] == 'pure_structure'
        assert payload['icon_templates_exist'] and payload['profiles']
        for profile in payload['profiles']:
            if profile.get('status') == 'pending':
                assert profile['message'] == '资产待配置' and profile['errors'] == []
                continue
            assert 'model_exists' not in profile and not profile['review']
            from src.app_catalog import MapCatalog
            catalog = MapCatalog()
            definition = next(p for p in catalog.profiles.values()
                              if p.map_name == profile['map'] and p.difficulty_name == profile['difficulty'])
            layouts = catalog.layouts(definition)
            assert profile['pure_structure_floors'] == sum(len(definition.floors_for_layout(layout)) for layout in layouts)
            assert profile['icon_structure_layouts'] == sum('2nd' in definition.floors_for_layout(layout) for layout in layouts)
            for variant in profile['template_originals'].values():
                if profile['difficulty'] in pending_profiles:
                    ready = (ready_floors or {}).get(profile['difficulty'], {})
                    assert variant['resource_layouts'] == profile['layouts']
                    assert len(variant['pending_scales']) == profile['layouts']
                    assert all(floors == [f for f in ('2nd', '1st') if f not in ready.get(layout, [])]
                               for layout, floors in variant['pending_scales'].items())
                    assert variant['fixed_candidates'] == sum(map(len, ready.values()))
                    assert variant['partial_layouts'] == len(ready)
                    assert variant['errors'] == [f"{layout}: 屏幕尺度待标定（{'、'.join(floors)}）"
                                                 for layout, floors in variant['pending_scales'].items()]
                else:
                    assert not variant['errors']
    elif payload.get('pending'):
        assert mode in ('recognize', 'project') and payload['accepted'] is False
    elif mode == 'project':
        assert payload['accepted'] and payload['layout'] == 'layout_05'
        from src.app_catalog import MapCatalog
        profile = next(iter(MapCatalog().profiles.values()))
        assert payload['variant'] in {version['id'] for version in profile.asset_versions
                                      if version['kind'] == 'route' and version['id'] in profile.template_versions}
        check_projection(output/'projection')
    elif mode == 'recognize':
        assert payload['candidate_count'] > 0 and payload['requires_confirmation']
    else:
        assert payload['gui_closed']
    return payload


def check_projection(directory):
    import numpy as np
    from src.floor_definitions import FLOORS
    report = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    assert report['accepted'] and report['floor'] in FLOORS
    registration = report['registration']
    assert registration['coordinate_space'] == 'single_floor_source_image'
    assert registration['clean_to_screen'] is None
    np.testing.assert_allclose(np.array(registration['original_to_screen']) @
                               np.array(registration['screenshot_to_original']), np.eye(3), atol=1e-9)
    return report
