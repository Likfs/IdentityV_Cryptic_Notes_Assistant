from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QLabel

from src import layout_retrieval as retrieval
from src import projection_workflow as workflow
from src import route_projection_app as app
from src.app_catalog import MapCatalog, load_settings
from src.route_projection_qt_ui import LayoutConfirmationDialog


def candidates():
    return retrieval.LayoutRetrievalResult(candidates=[
        retrieval.LayoutCandidate(1, 'layout_22', 'layout_22/layout_22_2nd.png', 'a',
                                  .95, .9, .8, .98, '图标位置通过', {}),
        retrieval.LayoutCandidate(2, 'layout_27', 'layout_27/layout_27_2nd.png', 'b',
                                  .2, .3, .1, .1, '入口图标未可靠定位', {}),
    ], timings_ms={'total': 10.})


@pytest.mark.parametrize('value', [None, 'unknown', [], 'cnn', 'icon_structure'])
def test_settings_default_to_icon_structure_and_preserve_explicit_choices(tmp_path, value):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'layout_recognition_backend': value}), encoding='utf-8')
    assert load_settings(MapCatalog(), path)['layout_recognition_backend'] == (
        value if value in ('cnn', 'icon_structure') else 'icon_structure')


def test_index_tracks_official_layout_names_and_invalidates_content(tmp_path, monkeypatch):
    clean, resources = tmp_path / 'clean', tmp_path / 'resources'
    shutil.copytree(retrieval.RESOURCE_ROOT / 'icons', resources / 'icons')
    source = next(iter(MapCatalog().profiles.values())).clean_root / 'layout_01/layout_01_2nd.png'
    target = clean / 'layout_22/layout_22_2nd.png'
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    calls = []
    def detect(image, reference):
        calls.append(True)
        return dict(confidence=.9, cx=20., cy=20., w=20, h=20, x=10, y=10)
    monkeypatch.setattr(retrieval, 'detect_target', detect)
    first = retrieval.LayoutRetriever(clean, resources, tmp_path / 'cache')
    original = first.prepare()
    assert [t['layout'] for t in first._targets] == ['layout_22'] and len(calls) == 1
    second = retrieval.LayoutRetriever(clean, resources, tmp_path / 'cache')
    assert second.prepare() == original and len(calls) == 1
    image = retrieval.read_image(target)
    image[0, 0] = 255 - image[0, 0]
    cv2.imencode('.png', image)[1].tofile(str(target))
    changed = second.prepare()
    assert changed != original and len(calls) == 2
    extra = clean / 'layout_28/layout_28_2nd.png'
    extra.parent.mkdir()
    shutil.copy2(target, extra)
    assert second.prepare() != changed
    assert [t['layout'] for t in second._targets] == ['layout_22', 'layout_28']
    icon = resources / 'icons/query.png'
    image = retrieval.read_image(icon)
    image[0, 0] = 255 - image[0, 0]
    cv2.imencode('.png', image)[1].tofile(str(icon))
    before = second._fingerprint
    assert second.prepare() != before and len(calls) == 3


@pytest.mark.parametrize('image', [None, np.zeros((0, 0, 3), np.uint8),
                                  np.zeros((80, 80), np.uint8), np.zeros((80, 80, 3), np.float32),
                                  np.zeros((1050, 1180, 3), np.uint8)])
def test_invalid_or_empty_input_does_not_return_a_layout(tmp_path, image):
    result = retrieval.LayoutRetriever(tmp_path / 'absent', cache_root=tmp_path / 'cache').recognize(image)
    assert not result.accepted and result.decision() is None and result.reasons


def test_icon_workflow_returns_manual_candidates_without_cnn_or_registration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Recognition called CNN or registration')
    monkeypatch.setattr(workflow, 'predict_image', forbidden)
    monkeypatch.setattr(workflow, 'register_screenshot', forbidden)
    profile = next(iter(MapCatalog().profiles.values()))
    calls = []
    retriever = SimpleNamespace(recognize=lambda image, size: calls.append(size) or candidates())
    kwargs = dict(catalog=MapCatalog(), profile=profile, model_bundle=(None, {}, None),
                  floor='2nd', layout_override=None, route_versions={}, opacity=1.,
                  prepared_cache={}, asset_cache={}, layout_recognition_backend='icon_structure',
                  layout_retriever=retriever, recognition_reference_size=(900, 800))
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), recognize_only=True, **kwargs)
    assert result.accepted and result.selected_layout == 'layout_22'
    assert not result.decision['accepted'] and 'confidence' not in result.decision
    assert calls == [(900, 800)] and result.registration is None
    assert result.diagnostics()['layout_retrieval']['parameters']['rotation_degrees'] == [0]
    with pytest.raises(ValueError, match='先识别并确认'):
        workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), **kwargs)


def test_confirmed_icon_layout_enters_template_projection_without_retrieval(monkeypatch):
    from src.registration.template_matching import direct
    calls = []
    def project(image, profile, layout, opacity, **kwargs):
        calls.append((profile, layout))
        registration = SimpleNamespace(accepted=True, rejection_reasons=[])
        return registration, Path('original.png'), image, image, {'registration': 1.}
    monkeypatch.setattr(direct, 'project_original', project)
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=None,
        profile=SimpleNamespace(key='test'), model_bundle=None, floor='2nd', layout_override='layout_27',
        route_versions={}, opacity=1., prepared_cache={}, asset_cache={},
        layout_recognition_backend='icon_structure', registration_backend='template')
    assert result.accepted and result.floor == 'all' and calls == [('test', 'layout_27')]
    assert result.layout_retrieval is None


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    qt = QApplication.instance() or QApplication([])
    for name in ('_start_model_load', '_start_registration_backend', '_start_layout_retrieval'):
        monkeypatch.setattr(app.RouteProjectionApp, name, lambda self: None)
    class Hotkeys:
        error = None
        def __init__(self, *args): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(app, 'GlobalActionHotkeys', Hotkeys)
    monkeypatch.setattr(app.RouteProjectionApp, '_handle_map_key', lambda self: None)
    monkeypatch.setattr(app.RouteProjectionApp, '_preload_locked_layout', lambda *args: None)
    item = app.RouteProjectionApp(settings_path=tmp_path / 'settings.json')
    yield item
    item.close()
    qt.processEvents()


def recognized(window):
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=window.catalog,
        profile=window.profile, model_bundle=None, floor='2nd', layout_override=None,
        route_versions={}, opacity=1., prepared_cache={}, asset_cache={}, recognize_only=True,
        layout_recognition_backend='icon_structure',
        layout_retriever=SimpleNamespace(recognize=lambda *args: candidates()))
    result.session_id = window.session_id
    return result


def test_icon_candidates_require_confirmation_and_can_lock_second_choice(window, monkeypatch):
    window.apply_settings({**window.settings, 'layout_recognition_backend': 'icon_structure',
                           'skip_layout_confirmation': True}, save=False)
    result = recognized(window)
    prompts = []
    monkeypatch.setattr(app.messagebox, 'confirm_layout',
        lambda items, *args, **kwargs: (prompts.append(items) or 'layout_27', False))
    monkeypatch.setattr(window.toast, 'show_message', lambda *args, **kwargs: None)
    window._apply_result('recognize', False, window.map_generation, [0, 0, 80, 80], result, None)
    assert window.locked_layout is None
    window._confirm_recognition(window.map_generation, result)
    assert window.locked_layout == 'layout_27' and len(prompts) == 1
    assert 'score' in prompts[0][0] and 'confidence' not in prompts[0][0]
    assert window.layout_combo.isEnabled()


@pytest.mark.parametrize('change', ['generation', 'session', 'profile', 'backend', 'window'])
def test_late_icon_result_cannot_restore_old_candidate(window, monkeypatch, change):
    window.apply_settings({**window.settings, 'layout_recognition_backend': 'icon_structure'}, save=False)
    result = recognized(window)
    generation = window.map_generation
    if change == 'generation': window.map_generation += 1
    elif change == 'session': window.session_id = 'different'
    elif change == 'profile': result.profile_key = 'other'
    elif change == 'backend': window.settings['layout_recognition_backend'] = 'cnn'
    else:
        result.recognition_window = [1, 0, 0, 80, 80]
        monkeypatch.setattr(app, 'recognition_window_identity', lambda titles: [2, 0, 0, 80, 80])
    monkeypatch.setattr(window, '_set_preview', lambda *args: pytest.fail('Stale preview displayed'))
    window._apply_result('recognize', False, generation, [0, 0, 80, 80], result, None)
    assert window._pending_confirmation is None and window.locked_layout is None


def test_settings_keep_projection_controls_and_clear_previous_lock(window):
    window.apply_settings({**window.settings, 'layout_recognition_backend': 'cnn'}, save=False)
    window.locked_layout = 'layout_01'
    window.preview_images['capture'] = (np.zeros((80, 80, 3), np.uint8), 'old')
    window.open_settings()
    page = window.settings_dialog
    page.layout_recognition_combo.setCurrentIndex(page.layout_recognition_combo.findData('icon_structure'))
    assert page.registration_backend_combo.isEnabled()
    assert not page.skip_layout_confirmation_check.isEnabled()
    assert page._persist_settings()
    assert load_settings(window.catalog, window.settings_path)['layout_recognition_backend'] == 'icon_structure'
    assert window.locked_layout is None
    assert 'capture' not in window.preview_images and window.reference_context is None


def test_confirmation_renders_scores_without_probability_or_skip(window):
    dialog = LayoutConfirmationDialog(candidates().decision()['ranking'][:2], window.profile.entrance_root)
    try:
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert '分数 0.9500' in texts and not any('95.0%' in text for text in texts)
        assert not dialog.skip_confirmation.isEnabled()
    finally:
        dialog.close()


def test_release_includes_required_icons_and_excludes_trial():
    from tools.release.build_windows_exe import runtime_files, ROOT
    paths = [p.relative_to(ROOT).as_posix() for p in runtime_files()]
    assert all(f'assets/layout_retrieval/icons/{name}.png' in paths for name in ('query', 'clean_09', 'clean_15'))
    assert not any(p.startswith('tmp/') for p in paths)


def test_offline_icon_projection_requires_a_confirmed_layout(monkeypatch, capsys):
    from src import __main__ as main
    catalog = MapCatalog()
    settings = load_settings(catalog)
    settings['layout_recognition_backend'] = 'icon_structure'
    monkeypatch.setattr(main, 'load_settings', lambda _: settings)
    monkeypatch.setattr(main, 'load_classifier', lambda *args, **kwargs: pytest.fail('Unexpected CNN fallback'))
    monkeypatch.setattr(main.sys, 'argv', ['src', '--project-once', 'unused.png'])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2 and '--layout' in capsys.readouterr().err
