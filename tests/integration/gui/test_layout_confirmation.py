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
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=window.catalog, profile=window.profile, layout_override=None, opacity=1.0, recognize_only=True, layout_recognition_backend='icon_structure', layout_retriever=SimpleNamespace(recognize=lambda *args: candidates()))
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
