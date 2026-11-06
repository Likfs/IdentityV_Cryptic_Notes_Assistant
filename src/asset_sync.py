"""Discover complete floor-image difficulties and publish a recoverable configuration batch."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from src.aligned_assets import IncompleteAssets, floor_images, generate, image_inventory, validate_route_coverage
from src.asset_audit import AssetAudit
from src.app_catalog import CLEAN_VERSION_ID, LEGACY_CLEAN_VERSION_ID, MapCatalog
from src.runtime_resources import BUNDLE_PATH, SCALE_PATH, read_bundle
from src.scale_config import load_scales

CATALOG_PATH = 'configs/map_catalog.json'
CONFIG_FILES = (CATALOG_PATH, SCALE_PATH, BUNDLE_PATH)
SYNC_DIRECTORY = 'runtime/cache/asset_sync'
DIFFICULTY_NAMES = dict(novice='新手', easy='简单', normal='普通', hard='困难', nightmare='噩梦')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def _inside(root, path):
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def _route_version(root, directory, previous=None):
    version = dict(previous or {})
    version.update(id=directory.name, name=directory.name, kind='route',
                   source_root=directory.relative_to(root).as_posix())
    return version


def _set_pending(definition):
    values = dict(asset_status='pending', floors=[], layout_floors={},
                  asset_versions=[], template_versions=[])
    changed = (any(definition.get(key) != value for key, value in values.items())
               or 'clean_root' in definition or 'entrance_root' in definition)
    definition.update(values)
    definition.pop('clean_root', None)
    definition.pop('entrance_root', None)
    return changed


def _directory_present(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f'素材路径必须是目录：{path}')
    return True


def discover(root, original, *, progress=None, audit=None):
    """Reconcile complete floor trees and keep missing difficulties as placeholders."""
    result = deepcopy(original)
    messages = []
    maps_root = _inside(root, root/'assets/maps')
    known_ids = {item['id'].casefold() for item in result['maps']}
    new_maps = []
    if _directory_present(maps_root):
        for directory in sorted(maps_root.iterdir()):
            if directory.is_dir() and directory.name.casefold() not in known_ids:
                new_maps.append(dict(id=directory.name, name=directory.name, difficulties=[]))
    retained_maps = []
    for item in [*result['maps'], *new_maps]:
        directory = _inside(maps_root, maps_root/item['id'])
        if not _directory_present(directory):
            messages.append(f'{item["id"]}：地图目录已删除，已移除配置')
            continue
        retained_difficulties = []
        for definition in item['difficulties']:
            difficulty_path = _inside(directory, directory/definition['id'])
            if _directory_present(difficulty_path):
                retained_difficulties.append(definition)
            else:
                messages.append(f'{item["id"]}/{definition["id"]}：难度目录已删除，已移除配置')
        item['difficulties'] = retained_difficulties
        existing = {definition['id']: definition for definition in item['difficulties']}
        for definition in existing.values():
            versions = {version['id']: version for version in definition['asset_versions']}
            old_clean = versions.get(LEGACY_CLEAN_VERSION_ID)
            if old_clean is not None and old_clean.get('kind') == 'clean':
                if CLEAN_VERSION_ID in versions:
                    raise ValueError(f'{item["id"]}/{definition["id"]}：clean 已被占用，请先重命名该路线目录')
                old_clean['id'] = CLEAN_VERSION_ID
                definition['template_versions'] = [CLEAN_VERSION_ID if value == LEGACY_CLEAN_VERSION_ID else value
                                                   for value in definition['template_versions']]
        for difficulty in sorted(directory.iterdir()) if directory.is_dir() else []:
            if not difficulty.is_dir():
                continue
            _inside(directory, difficulty)
            if progress is not None:
                progress(f'扫描地图 · {item["name"]} / {difficulty.name}', 0, 0)
            definition = existing.get(difficulty.name)
            if definition is not None and definition.get('asset_contract') != 'aligned_floor_images':
                continue
            pending = definition is not None and definition.get('asset_status') == 'pending'
            route_root = difficulty/'clean-with-route'
            try:
                _inside(difficulty, route_root)
                if route_root.exists() and not route_root.is_dir():
                    raise ValueError('clean-with-route 必须是目录')
                base = ((definition.get('clean_root') if definition else None)
                        or (difficulty/'clean').relative_to(root).as_posix())
                configured = {version['id']: version for version in definition['asset_versions']} if definition else {}
                # A removed optional canonical tree must not keep stale route references.
                # Explicit routes configured elsewhere still use their existing paths.
                canonical = (route_root.exists() or definition is None or pending or all(
                    (root/version['source_root']).resolve().is_relative_to(route_root.resolve())
                    for version in configured.values() if version['id'] != CLEAN_VERSION_ID))
                clean_images, floors = floor_images(root, base, progress=progress, strict_names=canonical, audit=audit)
                layout_floors = {}
                for name in clean_images:
                    layout, filename = name.split('/')
                    layout_floors.setdefault(layout, []).append(filename.removeprefix(layout+'_').removesuffix('.png'))
                layout_floors = {layout: [floor for floor in floors if floor in values]
                                 for layout, values in layout_floors.items()}
                if canonical:
                    routes = sorted(p for p in route_root.iterdir() if p.is_dir()) if route_root.is_dir() else []
                    versions = []
                    for route in routes:
                        _inside(difficulty, route)
                        if route.name.casefold() in (CLEAN_VERSION_ID, LEGACY_CLEAN_VERSION_ID):
                            raise ValueError(f'{route.name} 是底图保留 ID，请使用其他路线目录名')
                        versions.append(_route_version(root, route, configured.get(route.name)))
                else:
                    # Explicit paths remain supported when a difficulty has no new route tree.
                    versions = [version for version in definition['asset_versions'] if version['id'] != CLEAN_VERSION_ID]
                for version in versions:
                    route_images, _ = floor_images(root, version['source_root'], progress=progress,
                                                   strict_names=canonical, audit=audit)
                    validate_route_coverage(clean_images, route_images, version['id'])
                if definition is None:
                    definition = dict(id=difficulty.name,
                        name=DIFFICULTY_NAMES.get(difficulty.name, difficulty.name),
                        clean_root=base, asset_contract='aligned_floor_images', asset_status='ready',
                        floors=floors, asset_versions=[], template_versions=[])
                    item['difficulties'].append(definition)
                    existing[difficulty.name] = definition
                clean_version = dict(configured.get(CLEAN_VERSION_ID, {}))
                clean_version.update(id=CLEAN_VERSION_ID, kind='clean', source_root=base)
                if canonical:
                    clean_version['name'] = '原版干净底图'
                identifiers = [version['id'] for version in versions]
                if canonical:
                    selected = [value for value in definition['template_versions'] if value in identifiers]
                    selected.extend(value for value in identifiers if value not in selected)
                    selected.append(CLEAN_VERSION_ID)
                else:
                    selected = definition['template_versions']
                active_versions = [clean_version, *versions]
                # Avoid rewriting unchanged explicit-path declarations simply to add redundant per-layout floors.
                expected_floors = {layout: definition.get('layout_floors', {}).get(layout, definition['floors'])
                                   for layout in layout_floors}
                floor_changes = (expected_floors != layout_floors or definition['floors'] != floors
                                 or ('layout_floors' in definition and definition['layout_floors'] != layout_floors))
                if (pending or definition['asset_versions'] != active_versions or definition['template_versions'] != selected
                        or floor_changes or (canonical and definition.get('layout_floors') != layout_floors)):
                    removed = sorted(set(configured)-set(identifiers)-{CLEAN_VERSION_ID})
                    definition.update(clean_root=base, asset_versions=active_versions, template_versions=selected, floors=floors,
                                      layout_floors=layout_floors, asset_status='ready')
                    messages.append(f'{item["id"]}/{difficulty.name}：已同步布局、楼层和路线清单'
                                    + ('，已启用待配置难度' if pending else '')
                                    + (f'，移除旧引用：{", ".join(removed)}' if removed else ''))
            except (FileNotFoundError, IncompleteAssets) as exc:
                if definition is not None:
                    if _set_pending(definition):
                        messages.append(f'{item["id"]}/{difficulty.name}：素材不完整，已转为待配置：{exc}')
                    else:
                        messages.append(f'{item["id"]}/{difficulty.name}：暂未接入：{exc}')
                else:
                    messages.append(f'{item["id"]}/{difficulty.name}：暂未接入：{exc}')
            except (OSError, ValueError) as exc:
                messages.append(f'{item["id"]}/{difficulty.name}：暂未接入：{exc}')
        if item['difficulties']:
            retained_maps.append(item)
    result['maps'] = retained_maps
    return result, messages


def _atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', mode='w+b', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def recover(root):
    """An interrupted three-file publication restores its verified previous batch first."""
    directory = _inside(root, root/SYNC_DIRECTORY)
    journal = directory/'pending.json'
    if not journal.exists():
        return False
    previous = (directory/'previous.json').read_bytes()
    if json.loads(journal.read_text(encoding='utf-8')) != dict(backup_sha256=hashlib.sha256(previous).hexdigest()):
        raise ValueError('资源同步恢复备份指纹不一致，请检查 runtime/cache/asset_sync')
    saved = json.loads(previous)
    if set(saved) != set(CONFIG_FILES):
        raise ValueError('资源同步恢复备份文件集合无效')
    decoded = {name: None if value is None else base64.b64decode(value, validate=True)
               for name, value in saved.items()}
    for name in CONFIG_FILES:
        path = _inside(root, root/name)
        if decoded[name] is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write(path, decoded[name])
    journal.unlink()
    return True


def publish(root, replacements):
    """Write the catalog and scales before the manifest; a journal covers interruption."""
    if set(replacements) != set(CONFIG_FILES):
        raise ValueError('资源同步必须成组发布目录、尺度及运行清单')
    recover(root)
    directory = _inside(root, root/SYNC_DIRECTORY)
    saved = {}
    for name in CONFIG_FILES:
        path = _inside(root, root/name)
        saved[name] = base64.b64encode(path.read_bytes()).decode('ascii') if path.exists() else None
    previous = json_bytes(saved)
    _atomic_write(directory/'previous.json', previous)
    _atomic_write(directory/'pending.json', json_bytes(dict(backup_sha256=hashlib.sha256(previous).hexdigest())))
    try:
        for name in CONFIG_FILES:
            _atomic_write(_inside(root, root/name), replacements[name])
    except Exception:
        recover(root)
        raise
    (directory/'pending.json').unlink()


def _current(root, catalog_data, *, progress=None, audit=None):
    try:
        bundle = read_bundle(root)
        if bundle is None or image_inventory(root, catalog_data) != set(bundle['template']['images']):
            return False
        if audit is not None:
            audit.verify(bundle['files'], progress=progress)
            return True
        for index, (name, record) in enumerate(bundle['files'].items(), 1):
            if progress is not None:
                progress('核验已发布资源', index, len(bundle['files']))
            path = _inside(root, root/name)
            if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def synchronize(root, *, progress=None):
    """Run once before GUI resources are loaded; checks and offline replay remain read-only."""
    root = Path(root).resolve()
    with AssetAudit(root) as audit:
        return _synchronize(root, progress=progress, audit=audit)


def _synchronize(root, *, progress, audit):
    if progress is not None:
        progress('恢复并读取资源配置', 0, 0)
    restored = recover(root)
    path = _inside(root, root/CATALOG_PATH)
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes.decode('utf-8-sig'))
    scale_path = _inside(root, root/SCALE_PATH)
    original_scales = scale_path.read_bytes()
    scale_data = load_scales(root, raw=original_scales)
    MapCatalog(path, data=original, scales=scale_data)
    candidate, messages = discover(root, original, progress=progress, audit=audit)
    candidate_bytes = original_bytes if candidate == original else json_bytes(candidate)
    retained = {item['id']: {definition['id'] for definition in item['difficulties']} for item in candidate['maps']}
    updated_scales = dict(maps=deepcopy({
        map_id: {difficulty: entry for difficulty, entry in entries.items() if difficulty in retained[map_id]}
        for map_id, entries in scale_data['maps'].items() if map_id in retained
    }))
    for item in candidate['maps']:
        for definition in item['difficulties']:
            if definition.get('asset_status', 'ready') == 'ready':
                entries = updated_scales['maps'].setdefault(item['id'], {})
                entry = entries.setdefault(definition['id'], dict(minimum_scale=1.5, layout_minimum_scales={}))
                if entry['minimum_scale'] is None:
                    entry['minimum_scale'] = 1.5
    if candidate == original and updated_scales == scale_data and _current(root, original, progress=progress, audit=audit):
        return (['已恢复上次中断的资源发布'] if restored else [])+messages
    scale_bytes = original_scales if updated_scales == scale_data else json_bytes(updated_scales)
    data, scales = generate(root, catalog_bytes=candidate_bytes, scale_bytes=scale_bytes, progress=progress, audit=audit)
    repeated, repeated_messages = discover(root, original, progress=progress, audit=audit)
    if (path.read_bytes() != original_bytes or scale_path.read_bytes() != original_scales or repeated != candidate
            or repeated_messages != messages
            or image_inventory(root, candidate) != set(data['payload']['template']['images'])):
        raise ValueError('启动同步期间素材目录或配置发生变化，请放完图片后重新启动')
    audit.verify({name: record for name, record in data['payload']['files'].items()
                  if name not in (CATALOG_PATH, SCALE_PATH)}, progress=progress, stage='核对发布输入')
    if progress is not None:
        progress('更新配置和资源清单', 0, 0)
    publish(root, {CATALOG_PATH: candidate_bytes, SCALE_PATH: scales, BUNDLE_PATH: json_bytes(data)})
    return ['已校验素材并更新地图目录、路线版本、尺度及运行资源清单']+messages
