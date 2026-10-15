from __future__ import annotations
import shutil
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from src import layout_retrieval as retrieval
from src import projection_workflow as workflow
from src.app_catalog import MapCatalog
from tests.support.layout_candidates import candidates
from tests.support.replay import read_cases, replay_image


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
                                  np.zeros((80, 80), np.uint8), np.zeros((80, 80, 4), np.uint8),
                                  np.zeros((80, 80, 3), np.float32)],
                         ids=['none', 'empty', 'grayscale', 'bgra', 'float32'])
def test_invalid_input_is_rejected_before_resource_lookup(retrieval_scene, monkeypatch, image):
    monkeypatch.setattr(retrieval_scene, 'prepare',
                        lambda: pytest.fail('Invalid input reached resource preparation'))
    result = retrieval_scene.recognize(image)
    assert not result.accepted and result.decision() is None and not result.candidates
    assert result.reasons == ['截图必须为非空 BGR 三通道 uint8 图片']


def test_valid_image_with_missing_resources_reports_resource_error(tmp_path):
    case = read_cases()[0]
    result = retrieval.LayoutRetriever(tmp_path / 'absent', cache_root=tmp_path / 'cache').recognize(
        replay_image(case), case['reference_size'])
    assert not result.accepted and result.decision() is None and not result.candidates
    assert result.reasons == ['新算法识别资源不完整，缺少候选底图或图标模板']


def test_icon_workflow_returns_manual_candidates_without_cnn_or_registration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Recognition called CNN or registration')
    from src.registration.template_matching import direct
    monkeypatch.setattr(direct, 'project_original', forbidden)
    profile = next(iter(MapCatalog().profiles.values()))
    calls = []
    retriever = SimpleNamespace(recognize=lambda image, size: calls.append(size) or candidates())
    kwargs = dict(catalog=MapCatalog(), profile=profile, layout_override=None, opacity=1.,
                  layout_recognition_backend='icon_structure', layout_retriever=retriever,
                  recognition_reference_size=(900, 800))
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
        registration = SimpleNamespace(accepted=True, rejection_reasons=[], validation={'template': {'candidate_floor': 'b1'}})
        return registration, Path('original.png'), image, image, {'registration': 1.}
    monkeypatch.setattr(direct, 'project_original', project)
    result = workflow.process_screenshot(np.zeros((80, 80, 3), np.uint8), catalog=None, profile=SimpleNamespace(key='test'), layout_override='layout_27', opacity=1.0, layout_recognition_backend='icon_structure', registration_backend='template')
    assert result.accepted and result.floor == 'b1' and calls == [('test', 'layout_27')]
    assert result.layout_retrieval is None
