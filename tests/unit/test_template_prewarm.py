from threading import Event, Thread
from src.registration.template_matching import prewarm, resources


def test_latest_request_coalesces_old_work_and_never_marks_old_session_ready(monkeypatch):
    entered,release,finished=Event(),Event(),Event()
    seen=[]
    def prepare(*key,cancelled):
        seen.append(key)
        if len(seen)==1:
            entered.set()
            assert release.wait(3)
            assert cancelled()
        else:
            assert not cancelled()
            finished.set()
        return True
    monkeypatch.setattr(prewarm.floors,'prewarm',prepare)
    warmer=prewarm.FloorPrewarmer()
    try:
        warmer.request(('s1','map/diff','layout_01','route_clean'))
        assert entered.wait(3)
        warmer.request(('s2','map/diff','layout_02','route_clean'))
        key=('s3','map/diff','layout_03','route_lhp_solo')
        warmer.request(key)
        thread=warmer._thread
        release.set()
        assert finished.wait(3)
        thread.join(3)
        assert not thread.is_alive()
        assert seen==[('map/diff','layout_01','route_clean'),key[-3:]]
        assert warmer.ready_key==key
    finally:
        release.set();warmer.close()


def test_close_cancels_active_and_pending_work_without_waiting(monkeypatch):
    entered,release=Event(),Event()
    seen=[]
    def prepare(*key,cancelled):
        seen.append(key);entered.set()
        assert release.wait(3)
        assert cancelled()
        return True
    monkeypatch.setattr(prewarm.floors,'prewarm',prepare)
    warmer=prewarm.FloorPrewarmer()
    warmer.request(('s','p','l','v'))
    assert entered.wait(3)
    warmer.request(('s2','p','l2','v'))
    thread=warmer._thread
    warmer.close()
    warmer.request(('s3','p','l3','v'))
    release.set();thread.join(3)
    assert not thread.is_alive() and len(seen)==1
    assert warmer.ready_key is None


def test_live_match_lock_takes_precedence_over_background_loading():
    entered,release=Event(),Event()
    def live():
        with resources.resource_lock:
            entered.set();release.wait(3)
    thread=Thread(target=live)
    thread.start()
    try:
        assert entered.wait(3)
        assert not prewarm.floors.prewarm('not-read','not-read','route_clean')
    finally:
        release.set();thread.join(3)


def test_resource_error_stops_background_retries_and_does_not_mark_ready(monkeypatch):
    entered,release=Event(),Event()
    def bad(*key,**kw):
        entered.set();assert release.wait(3)
        raise ValueError('changed fingerprint')
    monkeypatch.setattr(prewarm.floors,'prewarm',bad)
    warmer=prewarm.FloorPrewarmer()
    try:
        warmer.request(('s','p','l','v'));assert entered.wait(3)
        thread=warmer._thread
        release.set();thread.join(3)
        assert not thread.is_alive() and warmer.ready_key is None
        assert warmer.last_error=='changed fingerprint'
    finally:
        release.set();warmer.close()
