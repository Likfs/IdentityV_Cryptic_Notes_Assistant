"""Standalone subprocess probe; no pytest dependency, no personal runtime state."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys

FORBIDDEN = ('torch', 'torchvision', 'retired_features', 'tools', 'tests', 'tkinter',
             'src.layout_classifier', 'src.registration_core',
             'src.registration.multiscale_structure', 'src.registration.runtime_registration')


class BlockRetired(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in FORBIDDEN):
            raise AssertionError('Production imported a non-runtime dependency: ' + fullname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('check', 'recognize', 'project', 'gui'), required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockRetired())
    sys.path.insert(0, str(args.root.resolve()))
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args.output.mkdir(parents=True, exist_ok=True)
    from src import paths
    paths.RUNTIME_ROOT = args.output
    paths.SETTINGS_PATH = args.output / 'settings.json'
    paths.CACHE_ROOT = args.output / 'cache'
    paths.LOG_ROOT = args.output / 'logs'
    paths.DIAGNOSTIC_ROOT = args.output / 'diagnostics'
    from src import __main__ as entry
    if args.mode == 'recognize':
        from src.app_catalog import MapCatalog
        from src.layout_retrieval import LayoutRetriever
        from src.projection_workflow import process_screenshot
        catalog = MapCatalog()
        profile = next(iter(catalog.profiles.values()))
        image = entry._read_image(args.root / 'tests/fixtures/registration/game2.png')
        result = process_screenshot(image, catalog=catalog, profile=profile,
            layout_override=None, opacity=.5, recognize_only=True,
            layout_retriever=LayoutRetriever(profile.clean_root, cache_root=paths.CACHE_ROOT))
        assert result.accepted, result.reasons
        assert result.registration is None and result.layer is None
        assert not result.decision['accepted'] and result.decision['layout'] is None
        assert len(result.layout_retrieval.candidates) == len(catalog.layouts(profile))
        print(json.dumps(dict(candidate_count=len(result.layout_retrieval.candidates),
                              requires_confirmation=True, accuracy_label=None)))
    else:
        if args.mode == 'check':
            sys.argv = ['src', '--check']
        elif args.mode == 'project':
            sys.argv = ['src', '--project-once', str(args.root / 'tests/fixtures/registration/game2.png'),
                        '--layout', 'layout_05', '--route-version', 'route_lhp_solo',
                        '--output', str(args.output / 'projection')]
        else:
            from types import SimpleNamespace
            from queue import Queue
            from src import route_projection_app as app
            from src.capture.mouse_pan import MousePanHook
            app.GlobalMapKeys = lambda *_: None
            app.GlobalActionHotkeys = lambda *_: SimpleNamespace(
                events=Queue(), error=None, start=lambda: None, stop=lambda: None)
            app.map_key_down = lambda *_: False
            app.escape_key_down = lambda: False
            app.floor_control_mouse_down = lambda *_: False
            app.game_window_is_foreground = lambda *_: False
            MousePanHook.start = lambda self: None
            app.RouteProjectionApp._start_layout_retrieval = lambda self: None
            sys.argv = ['src', '--smoke']
        entry.main()
        if args.mode == 'gui':
            print(json.dumps(dict(gui_closed=True)))
    assert not any(name == p or name.startswith(p + '.') for name in sys.modules for p in FORBIDDEN)


if __name__ == '__main__':
    main()
