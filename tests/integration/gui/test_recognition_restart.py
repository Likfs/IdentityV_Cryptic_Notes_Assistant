from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from src import route_projection_app as app
from src.capture.region import CaptureRegion
from tests.support.qt_wait import wait_for


def capture_setup(window,monkeypatch):
    monkeypatch.setattr(app,'foreground_game_client',lambda _:CaptureRegion(0,0,2560,1440))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda _:None)
    monkeypatch.setattr(app,'recognition_window_identity',lambda _:['game'])
    monkeypatch.setattr(window,'_layout_retriever',lambda:SimpleNamespace())
    captures=[]
    def capture(roi):
        captures.append(1)
        return np.full((30,30,3),100,np.uint8)
    monkeypatch.setattr(app,'capture_bgr',capture)
    return captures


@pytest.mark.parametrize('old_outcome',['result','error'])
def test_f8_restarts_capture_and_old_completion_cannot_clear_new_busy(window,monkeypatch,old_outcome):
    captures=capture_setup(window,monkeypatch)
    entered=[Event(),Event()];release=[Event(),Event()];finished=[Event(),Event()]
    calls=[]
    def process(image,**kw):
        index=len(calls);calls.append(kw);entered[index].set()
        try:
            assert release[index].wait(5)
            if index==0 and old_outcome=='error':raise RuntimeError('obsolete error')
            return SimpleNamespace(timings_ms={})
        finally:finished[index].set()
    monkeypatch.setattr(app,'process_screenshot',process)
    applied=[]
    def apply(*args,**kwargs):
        applied.append(args);window.busy=False
    monkeypatch.setattr(window,'_apply_result',apply)
    try:
        window.request_work('recognize',auto=False);wait_for(entered[0].is_set)
        first_session=window.session_id
        window.request_work('recognize',auto=False);wait_for(entered[1].is_set)
        assert len(captures)==2 and window.session_id!=first_session
        assert calls[0]['recognition_cancelled']() and not calls[1]['recognition_cancelled']()
        release[0].set();wait_for(finished[0].is_set)
        window._tick()
        assert window.busy and not applied
        release[1].set();wait_for(finished[1].is_set)
        wait_for(lambda:not window.results.empty());window._tick()
        assert len(applied)==1 and not window.busy
    finally:
        for event in release:event.set()


def test_recognition_during_projection_says_not_started_and_does_not_cancel_it(window,monkeypatch):
    capture_setup(window,monkeypatch)
    token=Event()
    window.busy=True
    window._active_work=dict(id='projection',mode='project',cancel=token)
    monkeypatch.setattr(app,'capture_bgr',lambda *a:pytest.fail('Rejected recognition captured'))
    window.request_work('recognize',auto=False)
    assert '正在配准' in window.status_var.get() and '本次识别未启动' in window.status_var.get()
    assert not token.is_set() and window.busy


def test_f8_discards_pending_confirmation_and_starts_new_capture(window,monkeypatch):
    captures=capture_setup(window,monkeypatch)
    window._pending_confirmation=(window.map_generation,object(),window.session_id,window.profile.key)
    done=Event()
    def process(*a,**kw):done.set();raise RuntimeError('test capture finished')
    monkeypatch.setattr(app,'process_screenshot',process)
    old_session=window.session_id
    window.request_work('recognize',auto=False)
    wait_for(done.is_set)
    assert captures and window._pending_confirmation is None and window.session_id!=old_session


def test_queued_old_result_and_progress_cannot_override_new_request(window,monkeypatch):
    window._active_work=dict(id='latest',mode='recognize',cancel=Event())
    window.busy=True
    window._set_status('正在识别布局…')
    window.results.put(('recognition_work','old',('work','recognize',False,window.map_generation,
                                                 [0,0,30,30],None,'obsolete error')))
    window._receive_work_event(('work_progress','old',window.session_id,window.map_generation,'旧提示'))
    window._tick()
    assert window.busy and window.status_var.get()=='正在识别布局…'


def test_invalidated_open_confirmation_closes_itself(qt_app):
    # Actual modal event loop, no native input or game interaction.
    assert app.messagebox.confirm_layout([dict(layout='layout_12',score=.9)],None,
                                          is_current=lambda:False) is None


def test_new_recognition_phase_uses_separate_capture_and_matching_messages(window,monkeypatch):
    window.busy=True
    window._active_work=dict(id='current',mode='recognize',cancel=Event())
    window._set_status('正在为布局识别截图…')
    window._receive_work_event(('work_progress','current',window.session_id,window.map_generation,'正在识别布局…'))
    assert window.status_var.get()=='正在识别布局…'
    window._active_work['mode']='project'
    window._receive_work_event(('work_progress','current',window.session_id,window.map_generation,'正在配准路线，匹配三层完整原图…'))
    assert '正在配准路线' in window.status_var.get()
