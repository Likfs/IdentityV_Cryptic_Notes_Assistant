import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.maintenance_paths import profile_root, evidence_path
from src.structure_layout.scales import read_scale_overrides
from tests.support.directory_links import create_directory_link, remove_directory_link
from tests.support.normalized_index import make_profile
from tools.assets.imports import build_recognition_index, prepare_profile_assets


def test_offline_recognition_build_uses_separate_evidence_and_live_pixels(tmp_path, monkeypatch):
    local = tmp_path/'local'
    local.mkdir()
    clean = make_profile(local)
    import shutil
    runtime = tmp_path/'assets/maps/map/diff/clean'
    shutil.copytree(clean, runtime)
    evidence = profile_root(runtime)
    evidence.mkdir(parents=True)
    for path in local.iterdir():
        if path.is_file():
            shutil.copy2(path, evidence/path.name)
    monkeypatch.setattr(build_recognition_index, 'project_relative', lambda p:p.relative_to(tmp_path).as_posix())
    expected = read_scale_overrides(clean, offline=True)
    assert read_scale_overrides(runtime, offline=True) == expected
    output = tmp_path/'staged'
    data = build_recognition_index.build(runtime, output)
    assert set(data['targets']) == set(expected)
    assert not (runtime.parent/'recognition_scales.json').exists()
    (evidence/'recognition_scales.json').unlink()
    # No fallback to stale files beside the runtime images.
    shutil.copy2(local/'recognition_scales.json', runtime.parent/'recognition_scales.json')
    assert read_scale_overrides(runtime, offline=True) == {}


def test_maintenance_dependencies_reject_paths_and_directory_links_outside_project(tmp_path):
    root = tmp_path/'project'
    clean = root/'assets/maps/map/diff/clean'
    clean.mkdir(parents=True)
    evidence = profile_root(clean)
    evidence.mkdir(parents=True)
    inside = root/'assets/inside'
    outside = tmp_path/'outside'
    for directory in (inside, outside):
        directory.mkdir(parents=True)
        (directory/'evidence.json').write_text('{}')
    assert evidence_path(clean,'assets/inside/evidence.json') == inside/'evidence.json'
    with pytest.raises(ValueError):
        evidence_path(clean,'../../../../../outside/evidence.json')
    with pytest.raises(ValueError):
        evidence_path(clean,str(outside/'evidence.json'))
    link=evidence/'linked'
    create_directory_link(link,inside)
    try:
        assert evidence_path(clean,'linked/evidence.json') == inside/'evidence.json'
        remove_directory_link(link,root=root)
        create_directory_link(link,outside)
        with pytest.raises(ValueError):
            evidence_path(clean,'linked/evidence.json')
    finally:
        remove_directory_link(link,root=root)


@pytest.mark.parametrize('publish',[False,True])
def test_profile_import_separates_live_images_from_calibration_and_masks(tmp_path,monkeypatch,publish):
    module=prepare_profile_assets
    profile=SimpleNamespace(key='map/diff',map_id='map',difficulty_id='diff',floors=('2nd','1st'),
        clean_root=tmp_path/'assets/maps/map/diff/clean',template_versions=('route_clean','route_zhanshi_full'))
    evidence=profile_root(profile.clean_root)
    source=evidence/'sources/route_zhanshi_full/1.png'
    source.parent.mkdir(parents=True)
    Image.new('RGB',(40,30),'gray').save(source)
    for floor in profile.floors:
        path=profile.clean_root/'layout_01'/f'layout_01_{floor}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(40,30),'gray').save(path)
    config=tmp_path/'maintenance/configs/template_matching.json'
    config.parent.mkdir(parents=True)
    config.write_text('{"floor_variants": {}}')
    for name in ('tools/assets/calibration/calibrate_zhanshi_routes.py','tools/common/route_calibration.py'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('tool')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'MapCatalog',lambda:SimpleNamespace(profile=lambda *a:profile))
    def calibrate(layout,source,*,clean_root,floors,asset_root,metadata_root,output_root,variant,working_max_size):
        items={}
        for floor in floors:
            asset=asset_root/variant/layout/f'{floor}.png'
            mask=metadata_root/variant/layout/f'{floor}_structure.png'
            for p in (asset,mask):
                p.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(40,30),'gray').save(p)
            items[floor]=dict(status='PASS',quality={'iou':1.},asset=asset.relative_to(tmp_path).as_posix(),
                structure_mask=mask.relative_to(tmp_path).as_posix())
        return dict(layout=layout,variant_id=variant,route=source.relative_to(tmp_path).as_posix(),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),floors=items)
    monkeypatch.setattr(module,'calibrate',calibrate)
    import sys
    monkeypatch.setattr(sys,'argv',['prepare','--profile','map/diff']+(['--publish'] if publish else []))
    module.main()
    prefix=tmp_path if publish else tmp_path/'workspace/asset_build'
    metadata=prefix/'maintenance/maps/map/diff/routes/route_zhanshi_full/layout_01'
    live=prefix/'assets/maps/map/diff/routes/route_zhanshi_full/layout_01'
    payload=json.loads((metadata/'calibration.json').read_text(encoding='utf-8'))
    for floor,item in payload['floors'].items():
        assert (live/f'{floor}.png').is_file()
        assert (metadata/f'{floor}_structure.png').is_file()
        assert item['asset'].startswith('assets/maps/')
        assert item['structure_mask'].startswith('maintenance/maps/')
    assert not (live/'calibration.json').exists()
    if publish:
        records=json.loads(config.read_text(encoding='utf-8'))['floor_variants']['map/diff']['layout_01']
        for floors in records.values():
            for record in floors.values():
                assert record['saved_scale'] is None
                for dep in record['dependencies']:
                    assert hashlib.sha256((tmp_path/dep['path']).read_bytes()).hexdigest()==dep['sha256']


def test_inventory_stages_two_floor_metadata_only_under_maintenance(tmp_path,monkeypatch):
    from tools.assets.calibration import organize_map_assets as module
    clean=tmp_path/'assets/maps/map/diff/clean'
    offline=profile_root(clean)/'routes'
    calibration=offline/'route_zhanshi_full/layout_01/calibration.json'
    calibration.parent.mkdir(parents=True)
    calibration.write_text(json.dumps(dict(route='maintenance/maps/map/diff/sources/1.png',
        floors={f:dict(status='PASS') for f in ('2nd','1st')})))
    profile=SimpleNamespace(key='map/diff',clean_root=clean,calibration_root=offline,floors=('2nd','1st'),
        asset_versions=[dict(id='route_zhanshi_full',name='全图',kind='route',source_root='maintenance/maps/map/diff/sources')])
    catalog=SimpleNamespace(profiles={'map/diff':profile},layouts=lambda p:['layout_01'],
        clean_template=lambda p,l,f:clean/l/f'{l}_{f}.png',
        variants=lambda p,l:[SimpleNamespace(variant_id='route_zhanshi_full',calibration_path=calibration)])
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'MapCatalog',lambda **kw:catalog)
    monkeypatch.setattr(module,'asset_output',lambda p,**kw:tmp_path/'staged'/p.relative_to(tmp_path))
    import sys
    monkeypatch.setattr(sys,'argv',['organize'])
    module.main()
    report=json.loads((tmp_path/'staged/maintenance/maps/inventory.json').read_text(encoding='utf-8'))
    assert report['profiles'][0]['versions'][0]['floor_count']==2
    data=json.loads((tmp_path/'staged/maintenance/maps/map/diff/routes/route_clean/layout_01/calibration.json').read_text(encoding='utf-8'))
    assert set(data['floors'])=={'2nd','1st'}
    assert not (tmp_path/'staged/assets').exists()
