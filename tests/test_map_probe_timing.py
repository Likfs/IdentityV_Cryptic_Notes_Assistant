"""Timing boundaries, event-thread delivery and bounded diagnostic I/O."""
import json
import threading
from types import SimpleNamespace
import numpy as np
from src import route_projection_app as app
from src import route_projection_core as core
from src.capture.region import CaptureRegion
from src.map_probe_diagnostics import LatestDiagnosticWriter
from test_immediate_map_probe import FakeApp, sample, floor


def test_floor_capture_and_classification_have_separate_timings(monkeypatch):
    clock=[0.]
    monkeypatch.setattr(core.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(core,'ensure_region_on_screen',lambda _:None)
    def capture(roi):
        clock[0]+=.004
        return np.zeros((roi[3],roi[2],3),np.uint8)
    def classify(image,size):
        clock[0]+=.002
        return floor()
    monkeypatch.setattr(core,'capture_bgr',capture)
    monkeypatch.setattr(core,'classify_floor_indicator',classify)
    timings={}
    detection,image,region=core.capture_floor_indicator(CaptureRegion(0,0,2560,1440),[2560,1440],timings=timings)
    assert detection.accepted and image.shape[:2]==(78,402)
    assert timings['floor_capture_ms']==4 and timings['floor_classify_ms']==2


def test_rounds_preserve_actual_retry_and_delivery_wait_without_io(monkeypatch):
    window=FakeApp();clock=[100.01];saved=[];logs=[]
    window._map_probe_diagnostics=SimpleNamespace(submit=lambda image,**details:saved.append(details))
    window._log=lambda text:logs.append(text)
    monkeypatch.setattr(app.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(app,'game_window_is_foreground',lambda _:True)
    def synchronous_io(*_,**kw):raise AssertionError('GUI performed diagnostic I/O')
    monkeypatch.setattr(app,'save_map_presence_diagnostic',synchronous_io)
    first=dict(attempt=1,_posted_at=100.007,initial_dispatch_ms=1.,thread_start_ms=1.,
               floor_capture_ms=3.,floor_classify_ms=1.,delivery='qt_signal')
    app.RouteProjectionApp._apply_map_presence_probe(window,7,0,True,app.MapPresence('closed',.1,(0,0)),
        None,None,None,sample(detection=floor(None)),'session',first)
    assert not saved[-1]['timing']['complete'] and window.scheduled[-1][0]==35
    clock[0]=100.060
    second=dict(attempt=2,_posted_at=100.059,retry_wait_ms=35.,retry_requested_ms=30.,
                floor_capture_ms=4.,floor_classify_ms=1.,delivery='qt_signal')
    app.RouteProjectionApp._apply_map_presence_probe(window,7,1,True,None,None,None,None,sample(),'session',second)
    timing=saved[-1]['timing'];totals=timing['totals_ms']
    assert timing['complete'] and len(timing['rounds'])==2
    assert totals['retry_wait_ms']==35 and totals['floor_capture_ms']==7
    assert abs(totals['delivery_wait_ms']-4)<1e-8
    assert abs(sum(totals.values())-timing['total_ms'])<1e-8
    assert window.requests==[('project',True)] and any('地图校验分段' in s for s in logs)


def test_signal_runs_receiver_on_gui_thread_and_checks_close_keys_first(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PyQt6.QtCore import QObject,Qt
    from PyQt6.QtWidgets import QApplication
    qt=QApplication.instance() or QApplication([])
    calls=[];owner=threading.get_ident()
    class Receiver(QObject):
        closed=False
        _receive_map_probe=app.RouteProjectionApp._receive_map_probe
        def _handle_map_key(self):calls.append(('keys',threading.get_ident()))
        def _apply_map_presence_probe(self,*_):calls.append(('apply',threading.get_ident()))
    receiver=Receiver();bridge=app.MapProbeBridge(receiver)
    bridge.ready.connect(receiver._receive_map_probe,Qt.ConnectionType.QueuedConnection)
    worker=threading.Thread(target=lambda:bridge.ready.emit(('map_presence',{})))
    worker.start();worker.join(timeout=1)
    assert not calls  # Worker never invokes GUI code directly.
    qt.processEvents()
    assert calls==[('keys',owner),('apply',owner)]


def test_close_key_between_ticks_invalidates_signal_result():
    window=FakeApp();window._handle_map_key=lambda:setattr(window,'map_generation',8)
    window._apply_map_presence_probe=lambda *args:app.RouteProjectionApp._apply_map_presence_probe(window,*args)
    profile={'_posted_at':100.}
    app.RouteProjectionApp._receive_map_probe(window,('map_presence',7,0,True,None,None,None,None,sample(),'session',profile))
    assert not window.requests


def test_slow_writer_does_not_block_submission_and_keeps_latest():
    entered=threading.Event();release=threading.Event();done=threading.Event();written=[]
    def save(image,**details):
        if image==1:
            entered.set();assert release.wait(2)
        written.append(image)
        if image==3:done.set()
    writer=LatestDiagnosticWriter(save)
    try:
        writer.submit(1);assert entered.wait(1)
        writer.submit(2);writer.submit(3)
        assert written==[]
        release.set();assert done.wait(2)
        assert written==[1,3]
    finally:
        release.set();writer.close()


def test_diagnostic_records_complete_timings_and_marks_skipped_marker(tmp_path,monkeypatch):
    from src import map_presence
    monkeypatch.setattr(map_presence,'MAP_MARKER_REPORT_DIR',tmp_path)
    timing={'complete':True,'total_ms':12.,'rounds':[{'floor_capture_ms':4.}],'totals_ms':{'floor_capture_ms':4.}}
    path=map_presence.save_map_presence_diagnostic(None,detection=None,region=None,attempt=0,
        floor_sample=sample(),observed_state='open',timing=timing)
    data=json.loads(path.read_text(encoding='utf-8'))
    assert not data['image_captured'] and data['confirmed_by']=='floor_indicator'
    assert data['timing']==timing
    history=[json.loads(s) for s in (tmp_path/'timings.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(history)==1 and history[0]['diagnostic_write_ms']>=0
