"""Cross-zoom consistency on unlabelled captures; not a layout accuracy benchmark."""
import json
import pytest

from src.app_catalog import MapCatalog, PROJECT_ROOT
from src.structure_layout.retriever import PureStructureRetriever
from src.projection_workflow import process_screenshot
from tests.support.replay import read_cases, replay_image


@pytest.fixture(scope='module')
def zoom_retriever():
    profile=MapCatalog().profile('a_bringer_of_doom','hard')
    retriever=PureStructureRetriever(profile.clean_root,floors=profile.floors)
    yield retriever
    retriever.close()


@pytest.mark.parametrize('pair', [1,2])
def test_paired_game_captures_agree_without_transferring_recognition_pose(zoom_retriever,pair,monkeypatch):
    from src.registration.template_matching import direct,matcher
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','hard')
    inputs={c['zoom_mode']:c for c in read_cases() if c.get('zoom_pair')==pair}
    results={mode:process_screenshot(replay_image(case),catalog=catalog,profile=profile,opacity=.5,
        recognize_only=True,layout_recognition_backend='pure_structure',layout_retriever=zoom_retriever)
        for mode,case in inputs.items()}
    minimum=results['minimum'];large=results['50_percent']
    assert minimum.decision['best_label']==large.decision['best_label']
    for mode,result in results.items():
        assert result.decision['ranking'][0]['recognition_zoom']==mode
        assert result.accepted  # Available candidates do not confirm a layout.
        assert not result.decision['accepted'] and result.decision['layout'] is None
        assert result.registration is None and result.layer is None
    # Explicitly simulate confirming the layout only, then a new minimum-zoom capture.
    layout=large.decision['best_label']
    monkeypatch.setattr(direct,'screen_info',lambda _:dict(factor=1))
    monkeypatch.setattr(matcher,'find_best_scale',lambda *a:pytest.fail('Live scale search'))
    projected=process_screenshot(replay_image(inputs['minimum']),catalog=catalog,profile=profile,
        opacity=.5,layout_override=layout,template_variant_id='route_zhanshi_full')
    assert projected.accepted and projected.layer is not None
    assert projected.layout_retrieval is None
    config=json.loads((PROJECT_ROOT/'maintenance/configs/template_matching.json').read_text(encoding='utf-8'))
    saved=config['floor_variants'][profile.key][layout]['route_zhanshi_full'][projected.floor]['saved_scale']
    assert projected.registration.screenshot_to_original[0,0]==pytest.approx(saved)
    assert saved > 3  # Projection must not inherit the approximately 1.31 recognition scale.
