"""Audit same-canvas PNG inputs for startup synchronization and offline publication."""
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.app_catalog import CLEAN_VERSION_ID, MapCatalog
from src.runtime_resources import digest, SCALE_PATH
from src.scale_config import load_scales, profile_keys
from src.floor_definitions import FLOORS
from src.paths import TEMPLATE_PROVENANCE_PATH, published_resource_path


class IncompleteAssets(ValueError):
    """Missing images may become available again on a later startup."""


def floor_images(root, directory, floors=None, *, progress=None, strict_names=True, audit=None, read_size=None):
    """Validate each layout's actual floors, or enforce an explicit uniform floor set."""
    root = Path(root).resolve()
    directory = (root/directory).resolve(strict=True)
    directory.relative_to(root)
    layouts = sorted(p for p in directory.glob('layout_*') if p.is_dir())
    if not layouts:
        raise IncompleteAssets(f'{directory.relative_to(root)}：没有布局')
    layout_paths = {
        layout: [path for path in sorted(layout.glob('*.png')) if strict_names or path.name in
                 {f'{layout.name}_{floor}.png' for floor in FLOORS}]
        for layout in layouts
    }
    total = sum(len(paths) for paths in layout_paths.values())
    if audit is not None:
        # One batch per source tree allows four workers even for single-floor layouts.
        audit.images([path for paths in layout_paths.values() for path in paths], progress=progress,
                     stage=f'校验素材 · {directory.relative_to(root).as_posix()}')
    images = {}
    found = set()
    for layout in layouts:
        if not re.fullmatch(r'layout_\d+', layout.name):
            raise ValueError(f'布局目录编号无效：{layout.name}')
        paths = layout_paths[layout]
        if not paths:
            raise IncompleteAssets(f'{layout.name}：缺少分层 PNG')
        for path in paths:
            if progress is not None:
                progress(f'校验素材 · {path.relative_to(root).as_posix()}', len(images), total)
            path.resolve(strict=True).relative_to(directory)
            floor = path.stem.removeprefix(layout.name+'_')
            if floor not in FLOORS or path.name != f'{layout.name}_{floor}.png':
                raise ValueError(f'分层 PNG 命名无效：{path.relative_to(root)}')
            if audit is not None:
                size = audit.image(path)['size']
            elif read_size is not None:
                size = read_size(path)
            else:
                with Image.open(path) as image:
                    if image.format != 'PNG':
                        raise ValueError(f'楼层图片必须为 PNG：{path.relative_to(root)}')
                    image.load()
                    size = image.size
            images[path.relative_to(directory).as_posix()] = size
            found.add(floor)
    expected_floors = set(floors) if floors is not None else found
    expected = ({f'{layout.name}/{layout.name}_{floor}.png'
                 for layout in layouts for floor in expected_floors} if floors is not None else set(images))
    if set(images) != expected:
        missing = sorted(expected-set(images))
        extra = sorted(set(images)-expected)
        raise IncompleteAssets(f'{directory.relative_to(root)}：楼层不完整，缺少 {missing}，多余 {extra}')
    if progress is not None:
        progress(f'校验素材 · {directory.relative_to(root).as_posix()}', len(images), total)
    return images, [floor for floor in FLOORS if floor in expected_floors]


def validate_route_coverage(clean_images, route_images, version):
    """Whole layouts are optional; every provided layout has exactly its clean floors."""
    layouts = {name.split('/')[0] for name in route_images}
    unknown = layouts - {name.split('/')[0] for name in clean_images}
    if unknown:
        raise ValueError(f'{version}：路线引用了不存在的底图布局：{", ".join(sorted(unknown))}')
    expected = {name for name in clean_images if name.split('/')[0] in layouts}
    missing, extra = expected-set(route_images), set(route_images)-expected
    if missing or extra:
        raise IncompleteAssets(f'{version}：已提供布局的楼层不完整，缺少 {sorted(missing)}，多余 {sorted(extra)}')
    for name, size in route_images.items():
        if size != clean_images[name]:
            raise ValueError(f'{version}/{name}：路线图与底图画布尺寸不一致')
    return layouts


def image_inventory(root, catalog_data):
    root = Path(root).resolve()
    names = set()
    for map_item in catalog_data['maps']:
        for definition in map_item['difficulties']:
            if definition.get('asset_status', 'ready') != 'ready':
                continue
            for version in definition['asset_versions']:
                if version['id'] in definition['template_versions']:
                    names.update(path.relative_to(root).as_posix()
                                 for path in (root/version['source_root']).glob('layout_*/*.png')
                                 if path.name in {f'{path.parent.name}_{floor}.png'
                                                  for floor in FLOORS})
    return names


def generate(root, *, catalog_bytes=None, scale_bytes=None, progress=None, audit=None):
    root = Path(root).resolve()
    catalog_path = root/'configs/map_catalog.json'
    original_catalog = catalog_path.read_bytes()
    catalog_bytes = original_catalog if catalog_bytes is None else catalog_bytes
    scale_path = root/SCALE_PATH
    original_scales = scale_path.read_bytes()
    scale_bytes = original_scales if scale_bytes is None else scale_bytes
    scale_data = load_scales(root, raw=scale_bytes)
    catalog_data = json.loads(catalog_bytes.decode('utf-8-sig'))
    expected_profiles = {f'{item["id"]}/{definition["id"]}' for item in catalog_data['maps']
                         for definition in item['difficulties']}
    if profile_keys(scale_data) != expected_profiles:
        raise ValueError('尺度配置与地图／难度清单不一致')
    catalog = MapCatalog(catalog_path, data=catalog_data, scales=scale_data)
    if catalog_path.read_bytes() != original_catalog:
        raise ValueError('读取期间地图目录发生变化，请重新导出')
    # MapCatalog resolves project paths for normal runtime; custom export roots
    # resolve input paths here so validation cannot escape the publishing tree.
    inputs = {catalog_path.resolve(): catalog_bytes, scale_path.resolve(): scale_bytes}
    input_names = {}

    def read(name):
        path = published_resource_path(root, name)
        input_names[path] = name
        if path in inputs:
            return inputs[path]
        raw = path.read_bytes()
        inputs[path] = raw
        return raw

    image_cache = {}
    audited_files = {}
    image_total = len(image_inventory(root, catalog_data))

    def image(name):
        if name in image_cache:
            return image_cache[name]
        if progress is not None:
            progress(f'审核运行图片 · {name}', len(image_cache), image_total)

        if audit is not None:
            verified = audit.image(name)
            size = list(verified['size'])
            fingerprint = verified['sha256']
            audited_files[name] = {key: verified[key] for key in ('bytes', 'sha256')}
        else:
            raw = read(name)
            with Image.open(BytesIO(raw)) as source:
                if source.format != 'PNG':
                    raise ValueError(f'楼层图片必须为 PNG：{name}')
                source.load()
                size = list(source.size)
            fingerprint = hashlib.sha256(raw).hexdigest()
        # Aligned images retain the game's dark gray background. The legacy
        # black-background threshold would turn that entire canvas into structure.
        record = dict(source_sha256=fingerprint, size=size,
                      map_threshold=56, scale_status='configured', preserve_colored_routes=True)
        image_cache[name] = record
        return record

    raw_catalog = json.loads(catalog_bytes.decode('utf-8-sig'))
    read('configs/app_defaults.json')
    read(TEMPLATE_PROVENANCE_PATH)
    # UI assets remain required even in a package containing only placeholders.
    for name in ('assets/app_ui/home_wallpaper.jpg', 'assets/app_ui/icons/check.svg',
                 'assets/app_ui/floor_controls.png',
                 'assets/layout_retrieval/icons/query.png', 'assets/layout_retrieval/icons/clean_09.png',
                 'assets/layout_retrieval/icons/clean_15.png'):
        read(name)
    groups, images, recognition, statuses = {}, {}, {}, {}
    for map_item in raw_catalog['maps']:
        for definition in map_item['difficulties']:
            profile = catalog.profile(map_item['id'], definition['id'])
            groups[profile.key], statuses[profile.key] = {}, {}
            if profile.asset_status == 'pending':
                continue
            targets = {}
            recognition[definition['clean_root']] = dict(targets=targets, source_images=True, profile_key=profile.key)
            if profile.asset_contract != 'aligned_floor_images':
                raise ValueError('新目录只能发布 aligned_floor_images 资产')
            clean = root/definition['clean_root']
            layouts = sorted(p for p in clean.glob('layout_*') if p.is_dir())
            if not layouts:
                raise ValueError(f'{profile.key}：没有布局，请标记 pending 或补充图片')
            versions = {v['id']: v for v in profile.asset_versions}
            if CLEAN_VERSION_ID not in versions:
                raise ValueError('必须声明 clean 干净底图版本')
            if not set(profile.template_versions).issubset(versions) or CLEAN_VERSION_ID not in profile.template_versions:
                raise ValueError('显示版本必须在 asset_versions 声明，并包含 clean')
            if versions[CLEAN_VERSION_ID]['source_root'] != definition['clean_root']:
                raise ValueError('clean 必须引用 clean_root')
            def read_size(path):
                return tuple(image(path.relative_to(root).as_posix())['size'])
            clean_images, _ = floor_images(root, definition['clean_root'], audit=audit, read_size=read_size)
            coverage = {CLEAN_VERSION_ID: {layout.name for layout in layouts}}
            for version in profile.template_versions:
                if version != CLEAN_VERSION_ID:
                    route_images, _ = floor_images(root, versions[version]['source_root'],
                                                   audit=audit, read_size=read_size)
                    coverage[version] = validate_route_coverage(clean_images, route_images, version)
            for layout in layouts:
                groups[profile.key][layout.name] = {}
                available = [version for version in profile.template_versions if layout.name in coverage[version]]
                for version in available:
                    variant = versions[version]
                    records = {}
                    for floor in profile.floors_for_layout(layout.name):
                        filename = f'{layout.name}_{floor}.png'
                        base_name = f'{definition["clean_root"]}/{layout.name}/{filename}'
                        base = image(base_name)
                        name = f'{variant["source_root"]}/{layout.name}/{filename}'
                        record = image(name)
                        if record['size'] != base['size']:
                            raise ValueError(f'路线图与干净底图画布尺寸不同：{name}')
                        images[name] = record
                        records[floor] = name
                        targets[f'{layout.name}/{filename}'] = dict(
                            source_sha256=base['source_sha256'])
                    groups[profile.key][layout.name][version] = records
                statuses[profile.key][layout.name] = [dict(
                    variant_id=v, source_name=v, floor_status={f:'PASS' for f in profile.floors_for_layout(layout.name)},
                    display_name=versions[v]['name'], asset_kind=versions[v].get('kind', 'route'))
                    for v in available]
    files = {(input_names[p] if p in input_names else p.relative_to(root).as_posix()):
             dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
             for p, raw in inputs.items()}
    files.update(audited_files)
    files[SCALE_PATH] = dict(bytes=len(scale_bytes), sha256=hashlib.sha256(scale_bytes).hexdigest())
    # Re-enumerate and read inputs: additions, removals and edits invalidate export.
    for map_item in raw_catalog['maps']:
        for definition in map_item['difficulties']:
            if definition.get('asset_status') == 'ready':
                expected = {p for p in (*inputs, *((root/name).resolve() for name in audited_files))
                            if p.is_relative_to((root/definition['clean_root']).resolve())}
                actual = {p.resolve() for p in (root/definition['clean_root']).glob('layout_*/*.png')}
                if expected != actual:
                    raise ValueError('导出期间底图清单变化或包含未声明楼层')
    if set(images) != image_inventory(root, raw_catalog):
        raise ValueError('图片清单变化或包含未声明布局／楼层，请重新审核')
    changed = catalog_path.read_bytes() != original_catalog or scale_path.read_bytes() != original_scales
    for index, (path, raw) in enumerate(inputs.items(), 1):
        if progress is not None:
            progress('核对导出输入', index, len(inputs))
        if path not in (catalog_path, scale_path) and path.read_bytes() != raw:
            changed = True
    if changed:
        raise ValueError('导出期间资源发生变化，请重新导出')
    if audit is not None:
        audit.verify(audited_files, progress=progress, stage='核对导出输入')
    payload = dict(template=dict(groups=groups, images=images, perf_mode=2,
                                  capture=dict(screen_name=''), asset_contract='aligned_floor_images'),
                   recognition=recognition, route_status=statuses, files=files,
                   maintenance_revision=digest(files))
    return dict(schema_version=2, payload_sha256=digest(payload), payload=payload), scale_bytes
