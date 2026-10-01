"""Recovery must improve recall without converting local evidence to a prior."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.registration.recovery import boundary_interval,clip_local_layer
from src.registration_core.credible_geometry import directional_walls,group_wall_edges
from src.registration.runtime_registration import prepare_registration_template,register_screenshot
from src.registration.temporal_evidence import TemporalEvidence


def test_scale_boundary_recovery_is_bounded_and_not_used_inside_range():
    assert boundary_interval(.4858,(.48578,1.45)) == (.25,.53578)
    assert boundary_interval(.80,(.48578,1.45)) is None
    assert boundary_interval(.25,(.25,1.45)) is None
    assert boundary_interval(2.,(.6,2.)) == (1.95,2.75)
    assert boundary_interval(2.75,(.6,2.75)) is None


def test_template_jagged_wall_is_retained_without_filling_doorway():
    mask=np.zeros((80,160),np.uint8)
    for x in range(15,145):
        if 70<=x<88:continue
        mask[30+(x//4)%2:60,x]=255
    template=SimpleNamespace(room_mask=mask,corridor_mask=np.zeros_like(mask))
    old=directional_walls(template)[0]
    new=directional_walls(template,True)[0]
    assert np.count_nonzero(new[27:34,20:65])>20
    assert np.count_nonzero(new[27:34,20:65])>np.count_nonzero(old[27:34,20:65])
    assert not np.any(new[27:34,74:84])


def test_pair_bright_wall_edges_but_keep_filled_passage_and_door_gap_separate():
    image=np.full((60,100,3),40,np.uint8)
    image[20:24,10:80]=170
    walls=np.float32([[10,20,79,20],[12,23,77,23],[10,40,39,40],[55,40,80,40]])
    directions=np.zeros(4,int)
    merged,_,groups=group_wall_edges(walls,directions,image,np.zeros((60,100),bool))
    assert len(merged)==3 and groups[0]==[0,1]
    # A uniformly filled narrow passage has no bright ridge.
    image[:]=80
    assert len(group_wall_edges(walls,directions,image,np.ones((60,100),bool))[0])==4


def test_local_layer_masks_far_pixels_and_uncertain_pixels_and_reuses_mask():
    details={'local_bounds':[10,10,89,89], 'selected_local_pose':[[1,0,0],[0,1,0]],
             'uncertainty_poses':[[[1,0,0],[0,1,0]],[[1.08,0,-4],[0,1.08,-4]]]}
    registration=SimpleNamespace(validation={'projection_scope':'local','reliable_geometry':details},
                                 observation_bbox=(20,30,100,100))
    layer=np.full((180,180,4),255,np.uint8)
    clipped=clip_local_layer(layer,registration)
    assert not np.any(clipped[:30,:,3]) and not np.any(clipped[:,120:,3])
    assert clipped[80,70,3]==255
    assert clipped[40,30,3]==0 # inside box, but combined x/y movement exceeds 4 px
    assert np.array_equal(clipped,clip_local_layer(layer,registration))
    registration.validation['projection_scope']='full'
    assert clip_local_layer(layer,registration) is layer


@pytest.mark.parametrize('source,layout,floor,scope',[
    ('20260926_163415_222816_layout_19_2nd','layout_19','2nd','full'),
    ('20260926_174358_909693_layout_04_2nd','layout_04','2nd','local'),
    ('20260926_175121_699252_layout_04_1st','layout_04','1st','local')])
def test_reported_capture_recovery_and_history_scope(tmp_path,source,layout,floor,scope):
    root=Path(__file__).resolve().parents[1]
    path=root/'runtime/diagnostics/route_projection/app_runs'/source/'01_screenshot.png'
    if not path.exists():pytest.skip('Local diagnostic is not distributed')
    image=cv2.imdecode(np.fromfile(path,np.uint8),1)
    prepared=prepare_registration_template(layout,floor,
        root/f'assets/maps/a_bringer_of_doom/nightmare/clean/{layout}/{layout}_{floor}.png',tmp_path)
    result=register_screenshot(image,prepared)
    assert 'recovery_error' not in result.validation
    assert result.validation.get('projection_scope')==scope,result.rejection_reasons
    assert result.validation['reliable_geometry']['position_status']=='confirmed'
    if layout=='layout_04' and floor=='1st':
        assert .30<result.screenshot_to_clean[0,0]<.40
        assert (result.screenshot_to_clean@np.float32([530,640,1]))[1]>800
        assert result.validation['scale_recovery_trace']
    if scope=='local':
        assert not result.accepted # workflow requires independent route verification
        result.accepted=True # simulate route confirmation; history must still refuse
        assert not TemporalEvidence.empty(prepared).add_registration(result,prepared)
    else:
        assert result.accepted


def test_repeated_rectangle_remains_unconfirmed_with_reliable_walls():
    from test_credible_geometry import colored_scene
    from src.registration_core.credible_geometry import screenshot_evidence,review_geometry
    template,observation,truth,image=colored_scene()
    for name in ('image','room_mask','corridor_mask','map_mask','edge_mask','stair_mask','distance_transform'):
        value=getattr(template,name);setattr(template,name,np.concatenate((value,value),axis=1))
    other=truth.copy();other[0,2]+=460
    _,details=review_geometry(template,screenshot_evidence(image,observation,reliable=True),[truth,other],
        scale_min=.7,scale_max=1.4,viewport=(300,300),origin=(0,0))
    assert not details['accepted'] and not details['local_accepted']
    assert details['competitive_branches']>=1


def test_local_status_is_not_reported_as_full_confirmation():
    from src.route_projection_core import format_projection_metrics
    registration=SimpleNamespace(accepted=True,validation={'projection_scope':'local',
        'edge_median_screen_px':1.3,'route_corroborated':True},timings={})
    result=SimpleNamespace(accepted=True,registration=registration,overlay_validation=None,timings_ms={})
    assert '完整显示，缩放持续校准' in format_projection_metrics(result)
    assert '可靠墙组' in format_projection_metrics(result)


def test_failed_expanded_geometry_cannot_be_overridden_by_route_validation():
    from src.projection_workflow import _route_can_corroborate
    registration=SimpleNamespace(clean_to_screen=np.eye(2,3),observation_mask=np.ones((100,100),np.uint8),
        validation={'observed_structure_pixels':10000,'structure_precision':.99,'semantic_precision':.99,
                    'reliable_recovery_rejected':True})
    assert not _route_can_corroborate(registration)
