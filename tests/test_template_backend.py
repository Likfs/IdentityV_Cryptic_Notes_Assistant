from types import SimpleNamespace
import json
import cv2
import numpy as np
import pytest
from src.registration.multiscale_structure import runtime as dispatch
from src.registration.template_matching import runtime as template
from src.app_catalog import MapCatalog, load_settings


def test_template_uses_saved_scale_original_coordinates_and_no_search(tmp_path,monkeypatch):
    monkeypatch.setattr(dispatch,'cpu_register',lambda *_:pytest.fail('定尺度模板匹配 called CPU registration'))
    monkeypatch.setattr(dispatch,'_gpu_register',lambda *_:pytest.fail('定尺度模板匹配 called GPU registration'))
    monkeypatch.setattr(template.matcher,'find_best_scale',lambda *_:pytest.fail('Saved scale must not search'))
    raw=np.zeros((180,220,3),np.uint8)
    h=np.float64([[2,0,100],[0,2,80],[.001,0,1]])
    entry=dict(saved_scale=1.5,clean_to_original=h.tolist(),source_reference='old.jpg',source_sha256='test')
    path=tmp_path/'original.jpg'
    monkeypatch.setattr(template,'_load_entry',lambda _:(entry,path,{'perf_mode':2}))
    monkeypatch.setattr(template,'screen_info',lambda _:dict(factor=4/3,width=1920,height=1080))
    def match(image,source,scale):
        assert source==str(path) and scale==2. and template.matcher.TEMPLATE_DS==.25
        return dict(score=12.,location=(30,40),scale=scale)
    monkeypatch.setattr(template.matcher,'_match_template',match)
    prepared=SimpleNamespace(raw_image=raw,registration_scale=.5,processed=SimpleNamespace(crop_offset=(120,80)))
    result=dispatch.register_screenshot(np.zeros((70,80,3),np.uint8),prepared,
        prior_screenshot_to_clean=np.eye(2,3),temporal_evidence=object(),soft_pose_hint=object(),backend='template')
    p=result.clean_to_screen@np.array([20.,10.,1.]);p=p[:2]/p[2]
    np.testing.assert_allclose(p,[(140/1.02-30)/2,(100/1.02-40)/2])
    np.testing.assert_allclose(result.screenshot_to_clean@result.clean_to_screen,np.eye(3),atol=1e-12)
    assert result.accepted and not result.validation['trusted_history_eligible']
    assert result.timings['calibration']==0
    assert dispatch.initialize('template')['effective_backend']=='template'
    assert dispatch.prewarm_template(prepared,'template')['effective_backend']=='template'


def test_missing_saved_scale_stays_rejected_without_search_or_fallback(monkeypatch):
    monkeypatch.setattr(dispatch,'cpu_register',lambda *_:pytest.fail('Legacy error retried CPU'))
    monkeypatch.setattr(template.matcher,'find_best_scale',lambda *_:pytest.fail('Missing scale triggers implicit search'))
    def fail(*_):raise ValueError('missing saved scale')
    monkeypatch.setattr(template,'_load_entry',fail)
    result=dispatch.register_screenshot(np.zeros((20,20,3),np.uint8),object(),backend='template')
    assert not result.accepted and result.clean_to_screen is None
    assert 'missing saved scale' in result.rejection_reasons[0]


def test_screen_factor_matches_original_convention():
    from src.registration.template_matching.screen import factor_for_size
    assert factor_for_size(2560,1440)==1
    assert factor_for_size(1920,1080)==pytest.approx(4/3)
    assert factor_for_size(3840,2160)==pytest.approx(2/3)
    assert factor_for_size(0,0)==1


def test_source_change_invalidates_saved_calibration(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.setattr(template,'PROJECT_ROOT',tmp_path)
    source=tmp_path/'source.jpg';source.write_bytes(b'old image')
    clean=tmp_path/'clean.png';clean.write_bytes(b'clean image')
    calibration=tmp_path/'calibration.json';calibration.write_bytes(b'calibration')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    entry=dict(source_image='source.jpg',source_sha256=sha(source),clean_sha256=sha(clean),
        calibration='calibration.json',calibration_sha256=sha(calibration))
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/template_matching.json').write_text(json.dumps({'entries':{'clean.png':entry}}),encoding='utf-8')
    prepared=SimpleNamespace(path=clean)
    assert template._load_entry(prepared)[1]==source
    source.write_bytes(b'new image content')
    with pytest.raises(ValueError,match='资源已改变'):template._load_entry(prepared)


def test_template_ui_choice_persists_and_invalidates_pending_projection(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PyQt6.QtWidgets import QApplication
    from src import route_projection_app as app
    class NoHotkeys:
        events=None;error=None
        def __init__(self,*_):pass
        def start(self):pass
        def stop(self):pass
    monkeypatch.setattr(app,'GlobalActionHotkeys',NoHotkeys)
    monkeypatch.setattr(app.RouteProjectionApp,'_start_model_load',lambda _:None)
    monkeypatch.setattr(app.RouteProjectionApp,'_start_registration_backend',lambda _:None)
    qt=QApplication.instance() or QApplication([])
    path=tmp_path/'settings.json'
    window=app.RouteProjectionApp(settings_path=path)
    try:
        window.open_settings()
        page=window.settings_dialog
        assert [page.registration_backend_combo.itemData(i) for i in range(3)]==['auto','cpu','template']
        generation=window.map_generation
        page.registration_backend_combo.setCurrentIndex(2)
        assert page._persist_settings()
        assert load_settings(MapCatalog(),path)['registration_backend']=='template'
        assert window.map_generation>generation and not window.live_projection
    finally:window.close()


def test_template_route_switch_reuses_pose_without_route_revalidation(monkeypatch):
    from src import projection_workflow as workflow
    from src.registration.runtime_registration import RuntimeRegistrationResult
    from src.projection.route_overlay import load_route_overlay_asset
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    variant=next(v for v in catalog.variants(profile,'layout_05') if v.variant_id=='route_lhp_solo')
    asset=load_route_overlay_asset(variant.calibration_path,'2nd')
    result=RuntimeRegistrationResult(True,[],np.eye(3),np.eye(3),None,None,
        {'registration_algorithm':'fixed_scale_template','trusted_history_eligible':False},{},None,None,None)
    monkeypatch.setattr(workflow,'validate_route_overlay',lambda *_:pytest.fail('Old candidate passed through new route checks'))
    checked=workflow.validate_display_asset(result,np.zeros((80,100,4),np.uint8),asset,
        catalog=catalog,profile=profile,asset_cache={})
    assert checked['accepted'] and not checked['enabled']
    result.accepted=False
    assert not workflow._route_can_corroborate(result)
    checked=workflow.validate_display_asset(result,None,asset,catalog=catalog,profile=profile,asset_cache={})
    assert not checked['accepted']
