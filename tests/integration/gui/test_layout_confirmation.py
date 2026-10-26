from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel
from src import projection_workflow as workflow
from src import route_projection_app as app
from src.route_projection_qt_ui import LayoutConfirmationDialog
from tests.support.layout_candidates import candidates


def recognized(window):
    window.locked_layout = None
    backend = window.settings['layout_recognition_backend']
    retrieval = candidates()
    retrieval.backend = backend
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=window.catalog, profile=window.profile, layout_override=None, opacity=1.0, recognize_only=True, layout_recognition_backend=backend, layout_retriever=SimpleNamespace(recognize=lambda *args: retrieval))
    result.session_id = window.session_id
    return result


def test_icon_candidates_require_confirmation_and_can_lock_second_choice(window, monkeypatch):
    window.apply_settings({**window.settings, 'layout_recognition_backend': 'icon_structure',
                           'skip_layout_confirmation': True}, save=False)
    result = recognized(window)
    prompts = []
    monkeypatch.setattr(app.messagebox, 'confirm_layout',
        lambda items, *args, **kwargs: (prompts.append(items) or 'layout_27'))
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


def test_confirmation_renders_scores_without_probability_or_skip(window):
    dialog = LayoutConfirmationDialog(candidates().decision()['ranking'][:2], window.profile.entrance_root)
    try:
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert '分数 0.9500' in texts and not any('95.0%' in text for text in texts)
        assert not dialog.confirmation_required.isEnabled()
    finally:
        dialog.close()


@pytest.mark.parametrize('backend', ['pure_structure', 'icon_structure'])
def test_auto_select_locks_first_candidate_without_dialog_or_projection(window, monkeypatch, backend):
    window.apply_settings({**window.settings, 'layout_recognition_backend': backend, 'auto_select_layout': True}, save=False)
    result = recognized(window)
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Unexpected confirmation'))
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Recognition reused for projection'))
    window._apply_result('recognize', False, window.map_generation, [0, 0, 80, 80], result, None)
    assert window.locked_layout == result.selected_layout == result.decision['ranking'][0]['layout']
    assert window.layout_var.get() == result.selected_layout
    assert window._pending_confirmation is None and window.live_projection is None
    assert result.registration is None and result.layer is None
    assert '已自动选择' in window.decision_label.text() and '缩到最小' in window.status_var.get()
    # Turning the preference off restores the existing choice between candidates.
    window.apply_settings({**window.settings, 'auto_select_layout': False}, save=False)
    window.clear_layout_lock()
    result = recognized(window)
    window._apply_result('recognize', False, window.map_generation, [0, 0, 80, 80], result, None)
    assert window.locked_layout is None and window._pending_confirmation is not None


@pytest.mark.parametrize('outcome', ['no_candidates', 'error', 'unknown_layout'])
def test_auto_select_does_not_lock_when_recognition_fails(window, monkeypatch, outcome):
    window.settings['auto_select_layout'] = True
    result = recognized(window)
    error = None
    if outcome == 'no_candidates':
        result.accepted = False
        result.selected_layout = None
        result.decision = None
    elif outcome == 'unknown_layout':
        result.selected_layout = 'layout_999'
    else:
        result, error = None, 'test failure'
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Unexpected confirmation'))
    window._apply_result('recognize', False, window.map_generation, [0, 0, 80, 80], result, error)
    assert window.locked_layout is None and window._pending_confirmation is None


@pytest.mark.parametrize('change', ['generation', 'session', 'profile', 'backend', 'window'])
def test_auto_select_rejects_stale_result(window, monkeypatch, change):
    window.settings['auto_select_layout'] = True
    result = recognized(window)
    generation = window.map_generation
    if change == 'generation': window.map_generation += 2
    elif change == 'session': window.session_id = 'next'
    elif change == 'profile': result.profile_key = 'other'
    elif change == 'backend': result.layout_recognition_backend = 'other'
    else:
        result.recognition_window = ['old']
        monkeypatch.setattr(app, 'recognition_window_identity', lambda _: ['new'])
    monkeypatch.setattr(window, '_set_preview', lambda *a: pytest.fail('Stale screenshot'))
    window._apply_result('recognize', False, generation, [0, 0, 80, 80], result, None)
    assert window.locked_layout is None and window._pending_confirmation is None


@pytest.mark.parametrize('backend', ['pure_structure', 'icon_structure'])
def test_hard_large_zoom_auto_select_locks_only_layout_and_requires_fresh_projection(window, monkeypatch, backend):
    from src.layout_retrieval import LayoutCandidate, LayoutRetrievalResult
    window.difficulty_var.set('困难')
    window._profile_changed()
    window.apply_settings({**window.settings, 'auto_select_layout': True, 'layout_recognition_backend': backend}, save=False)
    retrieval = LayoutRetrievalResult(backend=backend, timings_ms={'total': 1}, candidates=[
        LayoutCandidate(1, 'layout_26', 'layout_26/layout_26_2nd.png', 'hash', .88, .95, 0, 0, '候选',
                        {'floor': '2nd', 'scale': 1.308, 'recognition_zoom': '50_percent'})])
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=window.catalog,
        profile=window.profile, opacity=.5, recognize_only=True, layout_recognition_backend=backend,
        layout_retriever=SimpleNamespace(recognize=lambda *a: retrieval))
    result.session_id = window.session_id
    monkeypatch.setattr(app.messagebox, 'confirm_layout', lambda *a, **kw: pytest.fail('Automatic choice prompted'))
    monkeypatch.setattr(window, 'request_work', lambda *a, **kw: pytest.fail('Large-zoom capture reused for projection'))
    window._apply_result('recognize', False, window.map_generation, [0, 0, 80, 80], result, None)
    assert window.locked_layout == 'layout_26' and window._pending_confirmation is None
    assert '缩到最小' in window.status_var.get()
    assert result.registration is None and result.layer is None and window.live_projection is None
