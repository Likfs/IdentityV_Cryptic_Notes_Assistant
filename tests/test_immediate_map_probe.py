"""Actual app handlers with deterministic captures/clock; no live game required."""
from dataclasses import replace
from pathlib import Path
import queue
from types import SimpleNamespace
import numpy as np
import pytest
from src import route_projection_app as app
from src.capture.region import CaptureRegion
from src.floor_indicator import FloorDetection, CapturedFloorSample


def floor(value='b1'):
    return FloorDetection(value,{'b1':.99,'1st':.1,'2nd':.1},.89,'b1',40,None)


def sample(**kwargs):
    return replace(CapturedFloorSample(floor(),np.zeros((78,402,3),np.uint8),[1864,64,402,78],
        [0,0,2560,1440],[2560,1440],100.,7,'session'),**kwargs)


class InlineThread:
    def __init__(self,*,target,**kwargs):self.target=target
    def start(self):self.target()


class FakeApp:
    def __init__(self):
        self.settings={'game_window_titles':['game'],'capture_reference_size':[2560,1440],
            'auto_project':True,'auto_floor':True,'capture_roi':[901,185,1180,1050],
            'route_versions':{},'opacity':.72,'map_capture_delay_ms':35}
        self.map_generation=7;self.session_id='session';self.closed=False;self.busy=False
        self.map_state=app.MapToggleState(is_open=True);self.map_probe_pending=True
        self.map_presence_started_at=100.;self.locked_layout='layout_16';self.settings_dialog=None
        self.results=queue.Queue();self.requests=[];self.scheduled=[];self.hidden=0
        self.profile=SimpleNamespace(key='a_bringer_of_doom/nightmare');self.floor_var=SimpleNamespace(get=lambda:'b1')
        self.catalog=object();self.model_cache={};self.prepared_cache={};self.asset_cache={}
        self.registration_priors={};self.registration_soft_priors={};self.registration_evidence={}
        self.template_prepare_lock=None
    def _log(self,*args):pass
    def _refresh_status(self):pass
    def _set_status(self,*args):pass
    def hide_overlay(self):self.hidden+=1
    def after(self,delay,callback):self.scheduled.append((delay,callback))
    def request_work(self,mode,*,auto):self.requests.append((mode,auto))


@pytest.fixture
def environment(monkeypatch):
    clock=[100.04]
    monkeypatch.setattr(app.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    monkeypatch.setattr(app,'save_map_presence_diagnostic',lambda *a,**k:None)
    monkeypatch.setattr(app,'save_workflow_result',lambda _:Path('test-memory-only'))
    monkeypatch.setattr(app,'capture_bgr',lambda roi:np.zeros((roi[3],roi[2],3),np.uint8))
    return FakeApp(),clock


@pytest.mark.parametrize('floor_success',[True,False])
def test_only_floor_can_open_map_without_capturing_mylocation(environment,monkeypatch,floor_success):
    window,_=environment;calls=[]
    detection=floor() if floor_success else floor(None)
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**kw:(detection,sample().image,sample().region))
    from src import map_presence
    monkeypatch.setattr(map_presence,'classify_map_presence',lambda _:pytest.fail('mylocation is disabled'))
    monkeypatch.setattr(app,'capture_bgr',lambda _:pytest.fail('no extra marker screenshot'))
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True)
    event=window.results.get_nowait()
    app.RouteProjectionApp._apply_map_presence_probe(window,*event[1:])
    if floor_success:
        assert window.requests==[('project',True)] and window.map_state.is_open
        assert window.opening_floor_sample is not None and not window.map_probe_pending
        assert not window.scheduled
    else:
        assert not window.requests and window.map_probe_pending
        assert window.scheduled[-1][0]==35
    assert event[4:7]==(None,None,None)
    assert 'marker_capture_ms' not in event[-1]


def test_floor_capture_error_retries_without_mylocation(environment,monkeypatch):
    window,_=environment
    def broken(*a,**kw):raise RuntimeError('floor capture unavailable')
    monkeypatch.setattr(app,'capture_floor_indicator',broken)
    monkeypatch.setattr(app,'capture_bgr',lambda _:pytest.fail('no fallback marker capture'))
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True)
    app.RouteProjectionApp._apply_map_presence_probe(window,*window.results.get_nowait()[1:])
    assert not window.requests and window.map_probe_pending
    assert window.scheduled[-1][0]==35


@pytest.mark.parametrize('delay_ms', [0, 35, 80])
def test_custom_delay_controls_retry_and_respects_deadline(environment,delay_ms):
    window,clock=environment
    window.settings['map_capture_delay_ms']=delay_ms
    app.RouteProjectionApp._apply_map_presence_probe(window,7,0,True,None,None,None,None,None,'session')
    assert window.scheduled[-1][0]==max(1,delay_ms)
    clock[0]=100.495
    app.RouteProjectionApp._apply_map_presence_probe(window,7,1,True,None,None,None,None,None,'session')
    assert 1<=window.scheduled[-1][0]<=5
    assert not window.requests


def test_zero_delay_can_capture_immediately(environment,monkeypatch):
    window,clock=environment;clock[0]=100.
    window.settings['map_capture_delay_ms']=0
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**k:(floor(),sample().image,sample().region))
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True,'session')
    assert not window.scheduled and not window.results.empty()


@pytest.mark.parametrize("delay_ms", [35, 60, 120])
def test_early_timer_does_not_capture_before_configured_delay(environment,monkeypatch,delay_ms):
    window,clock=environment;captures=[]
    window.settings["map_capture_delay_ms"]=delay_ms
    def capture(*args,**kwargs):
        captures.append(clock[0]);return floor(),sample().image,sample().region
    monkeypatch.setattr(app,'capture_floor_indicator',capture)
    clock[0]=100.010
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True,'session')
    assert not captures and window.results.empty()
    clock[0]=100.+(delay_ms-1)/1000
    window.scheduled[-1][1]()
    assert not captures
    clock[0]=100.+delay_ms/1000
    window.scheduled[-1][1]()
    assert captures==[100.+delay_ms/1000]
    app.RouteProjectionApp._apply_map_presence_probe(window,*window.results.get_nowait()[1:])
    assert window.requests==[('project',True)]


@pytest.mark.parametrize('cancel',['generation','session','shutdown'])
def test_initial_wait_can_be_cancelled(environment,monkeypatch,cancel):
    window,clock=environment;clock[0]=100.010
    app.RouteProjectionApp._start_map_presence_probe(window,7,0,True,'session')
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a,**k:pytest.fail('cancelled capture'))
    if cancel=='generation':window.map_generation+=1
    elif cancel=='session':window.session_id='new'
    else:window.closed=True
    clock[0]=100.020
    window.scheduled[-1][1]()
    assert window.results.empty() and not window.requests


def test_absent_markers_retry_until_deadline_not_attempt_count(environment):
    window,clock=environment;closed=app.MapPresence('closed',.1,(0,0))
    for attempt,elapsed in enumerate((.01,.05,.15,.3,.49)):
        clock[0]=100+elapsed
        app.RouteProjectionApp._apply_map_presence_probe(window,7,attempt,True,closed,None,None,None,floor_sample=sample(detection=floor(None)),probe_session='session')
        assert not window.requests and window.map_probe_pending
        assert 1<=window.scheduled[-1][0]<=35
    clock[0]=100.501
    app.RouteProjectionApp._apply_map_presence_probe(window,7,5,True,closed,None,None,None)
    assert not window.map_state.is_open and not window.map_probe_pending and not window.requests


@pytest.mark.parametrize('generation,session',[(6,'session'),(7,'old-session')])
def test_stale_probe_never_opens_or_projects(environment,generation,session):
    window,_=environment;window.map_state.is_open=False
    app.RouteProjectionApp._apply_map_presence_probe(window,generation,0,True,None,None,None,None,sample(),session)
    assert not window.map_state.is_open and not window.requests


def test_focus_loss_prevents_projection(environment,monkeypatch):
    window,_=environment
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:False)
    app.RouteProjectionApp._apply_map_presence_probe(window,7,0,True,None,None,None,None,sample(),'session')
    assert not window.requests


def test_scheduled_retry_does_not_survive_session_reset(environment,monkeypatch):
    window,_=environment
    app.RouteProjectionApp._apply_map_presence_probe(window,7,0,True,
        app.MapPresence('closed',.1,(0,0)),None,None,None,None,'session')
    window.session_id='new-session'
    monkeypatch.setattr(app,'capture_floor_indicator',lambda *a:pytest.fail('old retry must be cancelled'))
    window.scheduled[-1][1]()
    assert window.results.empty()


@pytest.mark.parametrize('changed',[
    {'generation':8},{'session_id':'new'},{'client':[1,0,2560,1440]},
    {'reference_size':[1920,1080]},{'now':100.101},{'now':99.99},
])
def test_floor_reuse_rejects_old_or_different_capture_context(changed):
    context=dict(client=[0,0,2560,1440],reference_size=[2560,1440],generation=7,session_id='session',now=100.02)
    assert sample().reusable(**context)
    context.update(changed)
    assert not sample().reusable(**context)


@pytest.mark.parametrize('after_floor',['b1','1st'])
def test_reused_floor_is_still_verified_after_map_capture(environment,monkeypatch,after_floor):
    window,_=environment;window.opening_floor_sample=sample();captures=[];processed=[]
    def capture(*a):captures.append('floor');return floor(after_floor),sample().image,sample().region
    monkeypatch.setattr(app,'capture_floor_indicator',capture)
    def process(image,**kwargs):
        processed.append(kwargs['floor']);return SimpleNamespace(timings_ms={})
    monkeypatch.setattr(app,'process_screenshot',process)
    app.RouteProjectionApp.request_work(window,'project',auto=True)
    events=[]
    while not window.results.empty():events.append(window.results.get())
    work=next(e for e in events if e[0]=='work')
    assert work[-1] is None and captures==['floor']
    result=work[-2]
    assert result.floor_detection['before_map_capture_reused_from_open_probe']
    if after_floor=='b1':assert processed==['b1']
    else:
        assert not processed and not result.accepted and result.floor=='unknown'
    assert window.opening_floor_sample is None


def test_expired_floor_is_recaptured_before_and_after(environment,monkeypatch):
    window,clock=environment;window.opening_floor_sample=sample();clock[0]=100.2;calls=[]
    def capture(*a):calls.append(1);return floor(),sample().image,sample().region
    monkeypatch.setattr(app,'capture_floor_indicator',capture)
    monkeypatch.setattr(app,'process_screenshot',lambda *a,**k:SimpleNamespace(timings_ms={}))
    app.RouteProjectionApp.request_work(window,'project',auto=True)
    events=[]
    while not window.results.empty():events.append(window.results.get())
    result=next(e for e in events if e[0]=='work')[-2]
    assert len(calls)==2 and not result.floor_detection['before_map_capture_reused_from_open_probe']
