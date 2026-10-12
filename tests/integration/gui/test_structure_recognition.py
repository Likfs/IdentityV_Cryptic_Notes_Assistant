import threading
from types import SimpleNamespace
import numpy as np
import pytest
from src import route_projection_app as app
from src.projection_workflow import process_screenshot
from src.app_catalog import load_settings
from tests.support.layout_candidates import candidates


@pytest.mark.parametrize('backend',['pure_structure','icon_structure'])
def test_both_recognizers_always_require_human_confirmation(window,monkeypatch,backend):
    window.apply_settings({**window.settings,'layout_recognition_backend':backend,'skip_layout_confirmation':True},save=False)
    window.locked_layout=None
    retrieval=candidates();retrieval.backend=backend
    result=process_screenshot(np.zeros((80,80,3),np.uint8),catalog=window.catalog,profile=window.profile,
        opacity=1.,recognize_only=True,layout_recognition_backend=backend,
        layout_retriever=SimpleNamespace(recognize=lambda *a:retrieval))
    result.session_id=window.session_id
    prompts=[]
    monkeypatch.setattr(app.messagebox,'confirm_layout',lambda items,*a,**kw:(prompts.append(items) or 'layout_27'))
    monkeypatch.setattr(window.toast,'show_message',lambda *a,**kw:None)
    window._apply_result('recognize',False,window.map_generation,[0,0,80,80],result,None)
    assert window.locked_layout is None and result.registration is None
    window._confirm_recognition(window.map_generation,result)
    assert window.locked_layout=='layout_27' and len(prompts)==1


def test_switch_discards_candidates_retires_index_and_saves_choice(window,monkeypatch):
    window.apply_settings({**window.settings,'layout_recognition_backend':'pure_structure'},save=False)
    first=window._layout_retriever();closed=threading.Event()
    monkeypatch.setattr(first,'close',closed.set)
    generation=window.map_generation
    window._pending_confirmation=object();window.locked_layout='layout_02'
    window.open_settings();page=window.settings_page
    assert page.layout_recognition_combo.currentData()=='pure_structure'
    assert '缩到最小' in page.recognition_note.text()
    page.layout_recognition_combo.setCurrentIndex(page.layout_recognition_combo.findData('icon_structure'))
    assert page._persist_settings() and closed.wait(2)
    assert load_settings(window.catalog,window.settings_path)['layout_recognition_backend']=='icon_structure'
    assert window.map_generation>generation and window.locked_layout is None
    assert window._pending_confirmation is None and window._layout_retriever() is not first
    stale=SimpleNamespace(layout_recognition_backend='pure_structure')
    monkeypatch.setattr(window,'_set_preview',lambda *a:pytest.fail('Stale result displayed'))
    window._apply_result('recognize',False,window.map_generation,[0,0,80,80],stale,None)
    assert window.last_result is None


def test_capture_worker_routes_selected_recognizer(window,monkeypatch):
    from tests.support.app_fakes import InlineThread
    from src.capture.region import CaptureRegion
    window.settings['layout_recognition_backend']='pure_structure'
    retriever=SimpleNamespace(recognize=lambda *a:candidates())
    monkeypatch.setattr(window,'_layout_retriever',lambda:retriever)
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    monkeypatch.setattr(app,'foreground_game_client',lambda *a:None)
    monkeypatch.setattr(app,'game_capture_region',lambda *a:CaptureRegion(0,0,80,80))
    monkeypatch.setattr(app,'ensure_region_on_screen',lambda *a:None)
    monkeypatch.setattr(app,'capture_bgr',lambda *a:np.zeros((80,80,3),np.uint8))
    monkeypatch.setattr(app,'recognition_window_identity',lambda *a:(1,0,0,80,80))
    calls=[]
    original=app.process_screenshot
    def process(image,**kwargs):
        calls.append(kwargs)
        return original(image,**kwargs)
    monkeypatch.setattr(app,'process_screenshot',process)
    window.request_work('recognize',auto=False)
    assert len(calls)==1 and calls[0]['layout_recognition_backend']=='pure_structure'
    event=window.results.get_nowait()
    assert event[0]=='work' and event[-1] is None
    assert event[-2].layout_recognition_backend=='pure_structure'
    assert not event[-2].decision['accepted']
