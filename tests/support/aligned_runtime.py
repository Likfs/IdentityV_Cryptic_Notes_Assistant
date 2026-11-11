"""Small published same-canvas packages; never enable pending production assets."""
import json
import shutil

import cv2
import numpy as np

from src.paths import PROJECT_ROOT
from tools.assets.imports.export_aligned_assets import generate
from tools.assets.imports.export_runtime_resources import publish_package


def publish(root, catalog, *, scales=None):
    if scales is None:
        scale_path = root/'configs/scale_config.json'
        scales = json.loads(scale_path.read_text(encoding='utf-8')) if scale_path.exists() else {'maps': {}}
        for item in catalog['maps']:
            entries = scales['maps'].setdefault(item['id'], {})
            for definition in item['difficulties']:
                entries.setdefault(definition['id'], dict(minimum_scale=1.5 if definition.get('asset_status') != 'pending' else None, layout_minimum_scales={}))
    for name in ('configs/app_defaults.json', 'src/registration/template_matching/provenance.json',
                 'assets/app_ui/home_wallpaper.jpg', 'assets/app_ui/icons/check.svg',
                 'assets/app_ui/floor_controls.png',
                 'assets/layout_retrieval/icons/query.png', 'assets/layout_retrieval/icons/clean_09.png',
                 'assets/layout_retrieval/icons/clean_15.png'):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/name, target)
    for path in (PROJECT_ROOT/'src/registration/template_matching').glob('*.py'):
        shutil.copyfile(path, root/'src/registration/template_matching'/path.name)
    (root/'configs/map_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False), encoding='utf-8')
    (root/'configs/scale_config.json').write_text(json.dumps(scales), encoding='utf-8')
    data, scales = generate(root)
    publish_package(root/'configs/runtime_resources.json', data, scales)
    return root


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', image)[1].tofile(str(path))


def build_multifloor_package(root, *, layout_floors=None, image_size=(360, 450)):
    """Synthetic independent floor geometry; no claim about real Phosphorescence layouts."""
    from src.floor_definitions import FLOORS
    layout_floors = layout_floors or {
        'layout_01': ('b1', 'b2', 'b3', 'b4'), 'layout_02': FLOORS}
    base = 'assets/maps/phosphorescence_disease/hard'
    definition = dict(id='hard', name='困难', asset_contract='aligned_floor_images', asset_status='ready',
        clean_root=base+'/clean', floors=[f for f in FLOORS if any(f in v for v in layout_floors.values())],
        layout_floors={layout: [f for f in FLOORS if f in values] for layout, values in layout_floors.items()},
        asset_versions=[dict(id='clean', name='原版干净底图', kind='clean', source_root=base+'/clean'),
                        dict(id='测试路线', name='测试路线', kind='route', source_root=base+'/clean-with-route/测试路线')],
        template_versions=['测试路线', 'clean'])
    height, width = image_size
    for number, (layout, values) in enumerate(layout_floors.items(), 1):
        for floor in values:
            rng = np.random.default_rng(number*100+FLOORS.index(floor))
            image = np.full((height, width, 3), 35, np.uint8)
            for row in range(3):
                for column in range(3):
                    x = int(width*(.07+column*.3+rng.uniform(0,.04)))
                    y = int(height*(.07+row*.3+rng.uniform(0,.04)))
                    right = x+int(width*rng.uniform(.12,.2))
                    bottom = y+int(height*rng.uniform(.12,.2))
                    cv2.rectangle(image, (x,y), (right,bottom), (170,170,170), max(3,width//180))
                    cv2.line(image, (x,y), (right,y if rng.integers(2) else bottom), (100,100,100), 3)
            for version in definition['asset_versions']:
                pixels = image.copy()
                if version['kind'] == 'route':
                    cv2.line(pixels, (width//6,height//6), (width*5//6,height*5//6), (0,0,255), max(3,width//150))
                write_png(root/version['source_root']/layout/f'{layout}_{floor}.png', pixels)
    return publish(root, dict(schema_version=2, maps=[dict(id='phosphorescence_disease', name='测试多楼层地图',
                                                         difficulties=[definition])]))


def build_hard_replay_package(root):
    """Pin the original 29-layout candidate atlas for already labelled historical captures."""
    import hashlib
    source = PROJECT_ROOT/'tests/fixtures/retrieval/aligned_hard'
    records = json.loads((source/'provenance.json').read_text(encoding='utf-8'))['sources']
    for name, expected in records.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected
    definition = difficulty('hard', '困难', ('2nd', '1st'), {
        'clean': '原版干净底图', 'route_zhanshi_full': '展十全图路线'})
    for clean in sorted((source/'clean').glob('layout_*/*.png')):
        relative = clean.relative_to(source/'clean')
        for version in definition['asset_versions']:
            pixels = source/'routes/route_zhanshi_full'/relative if version['kind'] == 'route' else clean
            if not pixels.exists():
                pixels = clean  # Unexercised display variants do not affect candidate recognition.
            target = root/version['source_root']/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pixels, target)
    return publish(root, dict(schema_version=2, maps=[dict(
        id='a_bringer_of_doom', name='厄运之女', difficulties=[definition])]))


def difficulty(identifier, name, floors, versions):
    prefix = f'assets/maps/a_bringer_of_doom/{identifier}'
    return dict(id=identifier, name=name, floors=list(floors), asset_status='ready',
                asset_contract='aligned_floor_images',
                clean_root=f'{prefix}/clean', template_versions=list(versions),
                asset_versions=[dict(id=key, name=label, kind='clean' if key == 'clean' else 'route',
                    source_root=f'{prefix}/clean' if key == 'clean' else f'{prefix}/routes/{key}')
                    for key, label in versions.items()])


def build_gui_package(root):
    nightmare = difficulty('nightmare', '噩梦', ('2nd','1st','b1'), {
        'route_zhanshi_fast': '展十速刷路线图', 'route_lhp_solo': '凉哈皮单人速刷路线图',
        'clean': '原版干净底图', 'route_custom_a': '自定义路线甲', 'route_custom_b': '自定义路线乙'})
    hard = difficulty('hard', '困难', ('2nd','1st'), {
        'clean': '原版干净底图', 'route_zhanshi_full': '展十全图路线'})
    for definition in (nightmare, hard):
        for number in range(1, 31):
            if definition is hard and number == 24:
                continue
            layout = f'layout_{number:02d}'
            for index, floor in enumerate(definition['floors']):
                clean = np.full((280+index*12,320,3), (41,33,28), np.uint8)
                cv2.rectangle(clean, (30,30), (280,230), (100,100,100), 4)
                cv2.rectangle(clean, (60+number,60), (140+number,150), (140,140,140), 3)
                for offset, version in enumerate(definition['asset_versions']):
                    image = clean.copy()
                    if version['kind'] == 'route':
                        cv2.line(image, (45,75+offset*15), (270,200-offset*15),
                                 ((offset*71)%256,(offset*119)%256,255), 7)
                    write_png(root/version['source_root']/layout/f'{layout}_{floor}.png', image)
    return publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女',
                                                         difficulties=[nightmare,hard])]))


def activate(root, monkeypatch):
    from src import app_catalog, runtime_resources
    from src.registration.template_matching import floors
    original = app_catalog.MapCatalog.__init__
    def init(self, path=None, **kwargs):
        original(self, root/'configs/map_catalog.json' if path is None else path, **kwargs)
    floors._close_fixed_session()
    monkeypatch.setattr(app_catalog.MapCatalog, '__init__', init)
    for module in (app_catalog, runtime_resources, floors):
        monkeypatch.setattr(module, 'PROJECT_ROOT', root)


def build_nightmare_replay_package(root):
    definition = difficulty('nightmare', '噩梦', ('2nd','1st','b1'), {
        'clean': '原版干净底图', 'route_zhanshi_fast': '展十速刷路线图'})
    shutil.copytree(PROJECT_ROOT/'tests/fixtures/registration/aligned_nightmare',
                    root/'assets/maps/a_bringer_of_doom/nightmare')
    return publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='厄运之女',
                                                         difficulties=[definition])]), scales={'maps': {'a_bringer_of_doom': {'nightmare': {
        'minimum_scale': 1.49, 'layout_minimum_scales': {}}}}})


def build_phosphorescence_replay_package(root):
    base = 'assets/maps/phosphorescence_disease/nightmare'
    definition = dict(id='nightmare', name='噩梦', floors=['b1', 'b2', 'b3'],
        asset_status='ready', asset_contract='aligned_floor_images', clean_root=base+'/clean',
        template_versions=['clean'], asset_versions=[dict(
            id='clean', name='原版干净底图', kind='clean', source_root=base+'/clean')])
    shutil.copytree(PROJECT_ROOT/'tests/fixtures/registration/aligned_phosphorescence', root/base)
    return publish(root, dict(schema_version=2, maps=[dict(id='phosphorescence_disease',
        name='磷光症', difficulties=[definition])]))
