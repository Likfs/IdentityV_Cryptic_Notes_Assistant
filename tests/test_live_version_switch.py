from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.app_catalog import MapCatalog, load_settings
from src import projection_workflow as workflow
from src.registration.runtime_registration import runtime_registration_config
from src.registration_core.color_segmenter import segment_map


def accepted_capture(catalog, profile, floor="1st"):
    path = catalog.clean_template(profile, "layout_01", floor)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
    mask = segment_map(image, runtime_registration_config(), infer_known_background=False, include_stairs=False).map_mask
    registration = SimpleNamespace(accepted=True, clean_to_screen=np.eye(2, 3), validation={},
                                   observation_bbox=(0, 0, image.shape[1], image.shape[0]), observation_mask=mask)
    return workflow.WorkflowResult(screenshot=image, decision=None, selected_layout="layout_01", variant_id="route_lhp_solo",
        variant_source=None, registration=registration, overlay_validation=None, layer=None, projection=None,
        accepted=True, reasons=[], timings_ms={}, profile_key=profile.key, floor=floor,
        template_path=str(path), asset_path=None, calibration_path=None)


def test_switch_uses_pose_without_recognition_and_clean_has_independent_evidence(monkeypatch):
    catalog = MapCatalog()
    profile = next(iter(catalog.profiles.values()))
    original = accepted_capture(catalog, profile)
    monkeypatch.setattr(workflow, "register_screenshot", lambda *_a, **_k: pytest.fail("must not register again"))
    cache = {}
    images = []
    for version in ("route_clean", "route_zhanshi_fast", "route_lhp_solo"):
        switched = workflow.switch_result_version(original, catalog=catalog, profile=profile, variant_id=version,
            opacity=.72, asset_cache=cache, prepared_cache={})
        assert switched.accepted, switched.reasons
        assert switched.variant_id == version
        assert switched.registration is original.registration
        assert switched.run_directory is None
        assert switched.overlay_validation["evidence_variant"] == ("route_lhp_solo" if version == "route_clean" else version)
        images.append(switched.layer)
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])
    assert original.variant_id == "route_lhp_solo" and original.layer is None
    original.accepted = False
    with pytest.raises(ValueError, match="有效配准"):
        workflow.switch_result_version(original, catalog=catalog, profile=profile, variant_id="route_clean",
            opacity=.72, asset_cache=cache, prepared_cache={})


def test_independent_pose_support_does_not_approve_a_failed_display_asset():
    supporting = {"accepted": True, "observed_pixels": 10000,
                  "structure_coverage": .95, "distance_p95_px": 2., "variant": "route_zhanshi_fast"}
    display = {"accepted": True, "observed_pixels": 10000, "structure_coverage": .62,
               "distance_p95_px": 6., "independent_pose_corroboration": supporting}
    assert workflow._route_strongly_confirms(display)
    assert not workflow._route_strongly_confirms({**display, "accepted": False})


def test_local_evidence_displays_full_route_when_switching_all_versions():
    catalog=MapCatalog();profile=next(iter(catalog.profiles.values()))
    original=accepted_capture(catalog,profile)
    h,w=original.screenshot.shape[:2]
    left,top,right,bottom=w//2-60,h//2-60,w//2+60,h//2+60
    original.registration.validation.update(projection_scope='local',reliable_geometry={
        'local_bounds':[left,top,right,bottom],'selected_local_pose':[[1,0,0],[0,1,0]],
        'uncertainty_poses':[[[1,0,0],[0,1,0]]]})
    cache={}
    for version in ('route_clean','route_lhp_solo','route_zhanshi_fast'):
        result=workflow.switch_result_version(original,catalog=catalog,profile=profile,
            variant_id=version,opacity=.72,asset_cache=cache,prepared_cache={})
        assert result.accepted and result.registration is original.registration
        assert result.overlay_validation['projection_scope']=='full'
        assert result.overlay_validation['evidence_scope']=='local'
        assert result.registration.validation['projection_scope']=='local'
        expected,_=workflow.warp_route_overlay(cache[(profile.key,'layout_01',version,'1st')],
            original.registration.clean_to_screen,(w,h),.72)
        assert np.array_equal(result.layer,expected)
        assert np.any(result.layer[:top,:,3]) or np.any(result.layer[bottom+1:,:,3])


def test_live_window_switch_replaces_overlay_but_closed_map_stays_hidden(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src import route_projection_app as app
    class NoHotkeys:
        events = None
        error = None
        def __init__(self, *_args): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(app, "GlobalActionHotkeys", NoHotkeys)
    monkeypatch.setattr(app, "exclude_window_from_capture", lambda _: False)
    monkeypatch.setattr(app.RouteProjectionApp, "_start_model_load", lambda self: None)
    qt = QApplication.instance() or QApplication([])
    (tmp_path / "settings.json").write_text('{"registration_backend": "cpu"}', encoding="utf-8")
    window = app.RouteProjectionApp(settings_path=tmp_path / "settings.json")
    shown = []
    monkeypatch.setattr(window.overlay, "show", lambda layer, x, y: shown.append((layer, x, y)))
    try:
        assert window.route_combo.count() == 3  # Available before layout recognition.
        window.layout_var.set("layout_01")
        window.floor_var.set("1st")
        window._update_assets()
        previous = accepted_capture(window.catalog, window.profile)
        window.last_result = previous
        window.map_state.is_open = True
        roi = [321, 123, previous.screenshot.shape[1], previous.screenshot.shape[0]]
        window.live_projection = (previous, roi, window.map_generation)
        window.last_result = None  # A background recognition result is not the displayed pose.
        for version in ("route_clean", "route_zhanshi_fast", "route_lhp_solo"):
            window.route_combo.setCurrentIndex(window.route_combo.findData(version))
            assert window.last_result.variant_id == version
            assert window.live_projection[0] is window.last_result
            assert shown[-1][1:] == (321, 123)
            settings = load_settings(window.catalog, window.settings_path)
            assert set(settings["route_versions"][window.profile.key].values()) == {version}
        assert len(shown) == 3
        current = window.last_result
        window._apply_result("project", False, window.map_generation-1, roi, previous, None)
        assert window.last_result is current  # An old worker cannot replace the new version.
        window.hide_overlay()
        window.map_state.is_open = False
        window.route_combo.setCurrentIndex(window.route_combo.findData("route_clean"))
        assert len(shown) == 3 and window.live_projection is None
    finally:
        window.close()
        qt.processEvents()


def test_template_reported_basement_initial_projection_is_full_without_promoting_evidence(monkeypatch):
    # Preserve the old route-corroboration contract as a template regression.
    from src.registration.runtime_registration import register_screenshot as baseline_register
    monkeypatch.setattr(workflow, 'register_screenshot', baseline_register)
    from src.registration.temporal_evidence import TemporalEvidence
    path = Path(__file__).resolve().parents[1] / (
        'runtime/diagnostics/route_projection/app_runs/20260926_184200_218934_layout_10_b1/01_screenshot.png')
    if not path.exists():
        pytest.skip('Local diagnostic is not distributed')
    image = cv2.imdecode(np.fromfile(path, np.uint8), 1)
    catalog = MapCatalog()
    profile = catalog.profile('a_bringer_of_doom', 'nightmare')
    versions = {}; prepared = {}; assets = {}
    catalog.select_version_for_all(profile, 'route_zhanshi_fast', versions)
    result = workflow.process_screenshot(image, catalog=catalog, profile=profile,
        model_bundle=None, floor='b1', layout_override='layout_10', route_versions=versions,
        opacity=.65, prepared_cache=prepared, asset_cache=assets)
    assert result.accepted, result.reasons
    assert result.registration.validation['route_corroborated']
    assert result.registration.validation['projection_scope'] == 'local'
    search = result.registration.search_segmentation
    source = result.registration.source_segmentation
    fog = search.uncertain_mask > 0
    assert np.any(fog)
    assert not np.any(search.map_mask[fog])
    assert np.all(source.map_mask[fog] > 0)  # search does not rewrite verification evidence
    assert result.registration.validation['fog_filter']['scope'] == 'candidate_search_only'
    assert result.overlay_validation['projection_scope'] == 'full'
    assert result.overlay_validation['evidence_scope'] == 'local'
    expected, _ = workflow.warp_route_overlay(assets[(profile.key, 'layout_10', 'route_zhanshi_fast', 'b1')],
        result.registration.clean_to_screen, (image.shape[1], image.shape[0]), .65)
    assert np.array_equal(result.layer, expected)
    assert np.count_nonzero(result.layer[:, :, 3]) > 200000
    template = prepared[(profile.key, 'layout_10', 'b1')]
    assert not TemporalEvidence.empty(template).add_registration(result.registration, template)
