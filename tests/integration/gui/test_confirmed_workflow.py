"""Real retrieval -> modal choice -> real template projection; human/window boundaries are faked."""
import numpy as np
import pytest

from src import projection_workflow as workflow
from src import route_projection_app as app
from tests.support.replay import read_cases, replay_image


@pytest.mark.parametrize('choice', [0, 1, None], ids=['first', 'second', 'cancel'])
def test_real_candidates_only_project_after_explicit_confirmation(window, retrieval_scene, monkeypatch, choice):
    window.settings['layout_recognition_backend'] = 'icon_structure'
    case = read_cases()[0]
    screenshot = replay_image(case)
    window.locked_layout = None
    window.settings['template_motion_tracking'] = False
    recognized = workflow.process_screenshot(
        screenshot, catalog=window.catalog, profile=window.profile, layout_override=None,
        opacity=.5, recognize_only=True, layout_retriever=retrieval_scene,
        recognition_reference_size=case['reference_size'])
    recognized.session_id = window.session_id
    assert recognized.accepted and recognized.registration is None
    assert recognized.layer is None and not recognized.decision['accepted']
    generation = window.map_generation
    roi = [0, 0, *case['reference_size']]
    window._apply_result('recognize', False, generation, roi, recognized, None)
    assert window.locked_layout is None and window.live_projection is None
    displayed = []
    def confirm(candidates, *args, **kwargs):
        displayed.extend(candidates)
        return None if choice is None else candidates[choice]['layout']
    monkeypatch.setattr(app.messagebox, 'confirm_layout', confirm)
    window._confirm_recognition(generation, recognized, roi)
    assert [c['layout'] for c in displayed] == [c.layout for c in recognized.layout_retrieval.candidates[:2]]
    if choice is None:
        assert window.locked_layout is None
        monkeypatch.setattr(app, 'capture_bgr', lambda *_: pytest.fail('Cancelled confirmation captured a frame'))
        window.request_work('project', auto=False)
        assert not window.busy and window.live_projection is None
        return
    selected = displayed[choice]['layout']
    assert window.locked_layout == selected
    projected = workflow.process_screenshot(
        screenshot, catalog=window.catalog, profile=window.profile,
        layout_override=window.locked_layout, opacity=.5, template_variant_id='route_lhp_solo')
    assert projected.accepted, projected.reasons
    assert projected.selected_layout == selected and projected.layout_retrieval is None
    assert projected.floor == 'all' and projected.diagnostics()['floor_detection'] is None
    np.testing.assert_allclose(projected.registration.original_to_screen @
                               projected.registration.screenshot_to_original, np.eye(3), atol=1e-9)
    window.settings['template_route_versions'][window.profile.key] = 'route_lhp_solo'
    window.show_overlay_check.setChecked(True)
    window._apply_result('project', False, window.map_generation, roi, projected, None)
    assert window.live_projection[0] is projected and window.overlay.visible


@pytest.mark.parametrize('change', ['session', 'generation', 'profile'])
def test_context_change_inside_confirmation_does_not_lock(window, retrieval_scene, monkeypatch, change):
    window.settings['layout_recognition_backend'] = 'icon_structure'
    case = read_cases()[0]
    window.locked_layout = None
    result = workflow.process_screenshot(replay_image(case), catalog=window.catalog,
        profile=window.profile, layout_override=None, opacity=.5, recognize_only=True,
        layout_retriever=retrieval_scene, recognition_reference_size=case['reference_size'])
    result.session_id = window.session_id
    generation = window.map_generation
    window._apply_result('recognize', False, generation, [0, 0, *case['reference_size']], result, None)
    def confirm(candidates, *args, **kwargs):
        if change == 'session':
            window.session_id = 'replacement'
        elif change == 'generation':
            window.map_generation += 2
        else:
            from dataclasses import replace
            window.profile = replace(window.profile, map_id='replacement')
        return candidates[0]['layout']
    monkeypatch.setattr(app.messagebox, 'confirm_layout', confirm)
    window._confirm_recognition(generation, result)
    assert window.locked_layout is None and window.live_projection is None
