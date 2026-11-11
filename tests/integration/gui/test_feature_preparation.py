"""Profile preparation gates every recognition entry and survives game sessions."""
from types import SimpleNamespace
from threading import Event

from src import route_projection_app as app, recognition_resources as resources
from src.recognition_control import check_cancelled
from tests.support.app_fakes import InlineThread
from tests.support.qt_wait import wait_for

START_PREPARATION = app.RouteProjectionApp._start_layout_retrieval


def install_resources(window, monkeypatch, tmp_path, factory):
    window.recognition_resources.close()
    monkeypatch.setattr(resources, 'create_layout_retriever', factory)
    window.recognition_resources = resources.RecognitionResources(tmp_path/'cache')
    window.layout_retrievers = window.recognition_resources.retrievers
    window._retrieval_prepare_key = None
    monkeypatch.setattr(window, '_start_layout_retrieval', lambda:START_PREPARATION(window))


def test_progress_disables_button_hotkey_and_direct_requests_until_ready(window, tmp_path, monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    calls = []
    def prepare(*, cancelled, progress):
        calls.append(True)
        progress(1, 9)
        entered.set()
        try:
            while not release.wait(.01):
                check_cancelled(cancelled)
            check_cancelled(cancelled)
            return 'verified'
        finally:
            finished.set()
    retriever = SimpleNamespace(prepare=prepare, close=lambda:None)
    install_resources(window, monkeypatch, tmp_path, lambda *a, **kw:retriever)
    try:
        START_PREPARATION(window)
        wait_for(entered.is_set)
        window._tick()
        assert not window.start_button.isEnabled()
        assert '1/9' in window.feature_status_label.text()
        assert window.feature_progress_bar.value() == 1
        window.start_button.click()
        window.hotkeys.events.put('recognize')
        window._tick()
        window.request_work('recognize', auto=False)
        assert not window.busy and window.last_result is None
        assert window.grab().save(str(tmp_path/'features_progress.png'))
        release.set()
        wait_for(lambda:window._recognition_ready())
        window._tick()
        assert window.start_button.isEnabled()
        assert window.feature_status_label.text() == '识别资源就绪'
        window.session_id = 'next-round'
        window._refresh_status()
        START_PREPARATION(window)
        assert window.start_button.isEnabled() and calls == [True]
        assert window._layout_retriever() is retriever
    finally:
        release.set()
        wait_for(finished.is_set)


def test_profile_switch_ignores_old_token_and_reuses_ready_index(window, monkeypatch, tmp_path):
    calls = []
    def factory(*args, **kwargs):
        def prepare(**kwargs):
            calls.append(str(args[1]))
            kwargs['progress'](1, 1)
            return 'ready'
        return SimpleNamespace(prepare=prepare, close=lambda:None)
    install_resources(window, monkeypatch, tmp_path, factory)
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    START_PREPARATION(window)
    window._tick()
    first_profile, first_token = window.profile.key, window._retrieval_prepare_token
    first = window._layout_retriever()
    window.difficulty_var.set('困难')
    window._profile_changed()
    window._tick()
    assert window.start_button.isEnabled() and len(calls) == 2
    window.results.put(('retrieval_progress', first_profile, 'pure_structure', first, first_token, 0, 9))
    window.results.put(('retrieval_index', first_profile, None, 'old error', 'pure_structure', first, first_token))
    window._tick()
    assert window.feature_status_label.text() == '识别资源就绪'
    window.difficulty_var.set('噩梦')
    window._profile_changed()
    window._tick()
    assert window._layout_retriever() is first
    assert window.start_button.isEnabled() and len(calls) == 2


def test_preparation_survives_new_session_and_failure_can_retry(window, monkeypatch, tmp_path):
    calls = []
    def prepare(*, cancelled, progress):
        calls.append(True)
        progress(1, 2)
        window.session_id = 'new-session'
        assert not cancelled()
        if len(calls) == 1:
            raise OSError('测试缓存写入失败')
        return 'ready'
    install_resources(window, monkeypatch, tmp_path,
        lambda *a, **kw:SimpleNamespace(prepare=prepare, close=lambda:None))
    monkeypatch.setattr(app.threading, 'Thread', InlineThread)
    START_PREPARATION(window)
    window._tick()
    assert not window.start_button.isEnabled() and window.feature_retry_button.isVisible()
    assert '失败' in window.feature_status_label.text()
    window.feature_retry_button.click()
    window._tick()
    assert window.start_button.isEnabled() and not window.feature_retry_button.isVisible()
    assert calls == [True, True]


def test_switch_cancels_unfinished_profile_without_enabling_wrong_profile(window, monkeypatch, tmp_path):
    first_entered, cancelled, second_ready = Event(), Event(), Event()
    profiles = []
    def factory(*args, **kwargs):
        first = not profiles
        profiles.append(args[1])
        def prepare(*, cancelled, progress):
            progress(1, 2)
            if first:
                first_entered.set()
                while True:
                    if cancelled():
                        # The old profile must stop while the new profile is still preparing.
                        cancel_observed.set()
                        check_cancelled(cancelled)
                    Event().wait(.01)
            while not second_ready.wait(.01):
                check_cancelled(cancelled)
            return 'second-ready'
        return SimpleNamespace(prepare=prepare, close=lambda:None)
    cancel_observed = cancelled
    install_resources(window, monkeypatch, tmp_path, factory)
    try:
        START_PREPARATION(window)
        wait_for(first_entered.is_set)
        old_profile, old_token, old_retriever = window.profile.key, window._retrieval_prepare_token, window._layout_retriever()
        window.difficulty_var.set('困难')
        window._profile_changed()
        wait_for(cancelled.is_set)
        window.results.put(('retrieval_index', old_profile, 'obsolete-ready', None,
                            'pure_structure', old_retriever, old_token))
        window._tick()
        assert not window.start_button.isEnabled()
        second_ready.set()
        wait_for(window._recognition_ready)
        window._tick()
        assert window.start_button.isEnabled() and len(profiles) == 2
    finally:
        second_ready.set()
        window._retrieval_prepare_cancel.set()
