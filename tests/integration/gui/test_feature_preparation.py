"""Feature preparation progress observes GUI identity and cooperative cancellation."""
from types import SimpleNamespace
from threading import Event

from src import route_projection_app as app
from tests.support.app_fakes import InlineThread

START_PREPARATION = app.RouteProjectionApp._start_layout_retrieval

def test_feature_progress_and_ready_ignore_old_token_and_old_session(window, tmp_path):
    backend='pure_structure';window.settings['layout_recognition_backend']=backend
    retriever=SimpleNamespace();token=object()
    window._retrieval_prepare_token=token
    window.layout_retrievers={(window.profile.key,backend):retriever}
    def progress(t=token,session=None):
        window.results.put(('retrieval_progress',window.profile.key,backend,retriever,t,
                            window.session_id if session is None else session,1,9))
        window._tick()
    progress()
    assert '1/9' in window.recognition_method_label.text()
    assert '1/9' in window.feature_status_label.text()
    window.show()
    app.QApplication.processEvents()
    assert window.grab().save(str(tmp_path/'features_progress.png'))
    window.recognition_method_label.setText('new state')
    progress(object());progress(session='previous-round')
    assert window.recognition_method_label.text()=='new state'
    window.results.put(('retrieval_index',window.profile.key,'a'*64,None,backend,retriever,object(),window.session_id))
    window._tick()
    assert window.recognition_method_label.text()=='new state'
    window.results.put(('retrieval_index',window.profile.key,'a'*64,None,backend,retriever,token,window.session_id))
    window._tick()
    assert '资源就绪' in window.recognition_method_label.text()
    assert window.feature_status_label.text()=='识别资源就绪'
    window.session_id='next-round'
    window._refresh_status()
    assert not window.feature_status_label.text()


def test_prepare_cancels_on_session_change_without_closing_reused_retriever(window, monkeypatch):
    backend='pure_structure';window.settings['layout_recognition_backend']=backend
    closed=Event();calls=[]
    def prepare(*,cancelled,progress):
        assert not cancelled()
        progress(1,2)
        window.session_id='new-session'
        calls.append(cancelled())
        app.check_cancelled(cancelled)
    retriever=SimpleNamespace(prepare=prepare,close=closed.set)
    window.layout_retrievers={(window.profile.key,backend):retriever}
    monkeypatch.setattr(window,'_layout_retriever',lambda:retriever)
    monkeypatch.setattr(app.threading,'Thread',InlineThread)
    START_PREPARATION(window)
    window.recognition_method_label.setText('new state')
    window._tick()
    assert calls==[True] and not closed.is_set()
    assert window.recognition_method_label.text()=='new state'
