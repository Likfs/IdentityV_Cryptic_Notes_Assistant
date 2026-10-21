import pytest
from PyQt6.QtWidgets import QLabel
from src.app_catalog import selected_template_variant
from src.route_projection_qt_ui import LayoutConfirmationDialog, EntrancePreview


def test_switch_to_hard_clears_old_context_and_only_lists_two_floors(window, tmp_path):
    window.settings['floor'] = 'b1'
    window.settings['reference_floor'] = 'b1'
    window.difficulty_var.set('困难')
    window._profile_changed()
    assert window.profile.key == 'a_bringer_of_doom/hard'
    assert window.locked_layout is None
    assert window.settings['floor'] == '2nd'
    import json
    assert json.loads(window.settings_path.read_text(encoding='utf-8'))['reference_floor'] == '2nd'
    assert [window.floor_combo.itemData(i) for i in range(window.floor_combo.count())] == ['2nd', '1st']
    assert selected_template_variant(window.settings, window.profile.key) == 'route_zhanshi_full'
    assert [window.route_combo.itemData(i) for i in range(window.route_combo.count())] == [
        'route_clean', 'route_zhanshi_full', 'route_zhanshi_full_clean']
    window.open_settings()
    assert '50%' in window.settings_page.recognition_note.text()
    assert '缩到最小再配准' in window.settings_page.recognition_note.text()
    window._leave_settings(window.settings_page)
    window.show()
    window.grab().save(str(tmp_path/'hard_window.png'))
    window.difficulty_var.set('噩梦')
    window._profile_changed()
    assert [window.floor_combo.itemData(i) for i in range(window.floor_combo.count())] == ['2nd', '1st', 'b1']
    assert window.route_combo.findData('route_zhanshi_full_clean') == -1


def test_hard_clean_version_switch_rematches_and_uses_new_reference(window, monkeypatch, tmp_path):
    from src.app_catalog import load_settings
    from tests.support.qt_wait import wait_for
    calls = []
    monkeypatch.setattr(window, 'request_work', lambda mode, **kw: calls.append((mode, kw)))
    window.difficulty_combo.setCurrentText('困难')
    window.locked_layout = 'layout_05'
    window.layout_var.set('layout_05')
    window.map_state.is_open = True
    window._update_assets()
    generation = window.map_generation
    window.last_result = object()
    window.route_combo.setCurrentIndex(window.route_combo.findData('route_zhanshi_full_clean'))
    assert window.map_generation == generation+1
    assert window.last_result is None and window.live_projection is None
    assert calls == [('project', {'auto': False})]
    assert load_settings(window.catalog, window.settings_path)['template_route_versions'][
        window.profile.key] == 'route_zhanshi_full_clean'
    seen = []
    monkeypatch.setattr(window.reference_overlay, 'show_map', lambda *args: seen.append(args))
    window.floor_combo.setCurrentIndex(window.floor_combo.findData('1st'))
    window.reference_visible = True
    window._refresh_reference_map()
    wait_for(lambda: bool(seen))
    assert seen[-1][0].shape[:2] == (3592, 4360)
    assert '展十全图路线底图版' in seen[-1][2]
    window.show()
    window.grab().save(str(tmp_path/'hard_clean_version.png'))


def test_missing_entrance_uses_labeled_clean_preview_and_requires_confirmation(qt_app, tmp_path):
    from PyQt6.QtGui import QImage, QColor
    path = tmp_path/'layout_01_2nd.png'
    image = QImage(300, 200, QImage.Format.Format_RGB32)
    image.fill(QColor('gray'))
    image.save(str(path))
    dialog = LayoutConfirmationDialog([dict(layout='layout_01', score=.8, preview_path=path)], None)
    assert dialog.result() != dialog.DialogCode.Accepted
    assert not dialog.findChild(EntrancePreview).source.isNull()
    assert any('入口图待补' in label.text() for label in dialog.findChildren(QLabel))
    dialog.reject()
    assert dialog.result() == dialog.DialogCode.Rejected
    dialog.deleteLater()


@pytest.mark.parametrize('locked_layout', [None, 'layout_07'])
def test_confirmation_previews_the_matched_floor_instead_of_always_second(window, monkeypatch, locked_layout):
    from src import route_projection_app as app
    from src.layout_retrieval import LayoutRetrievalResult, LayoutCandidate
    from src.projection_workflow import process_screenshot
    from types import SimpleNamespace
    import numpy as np
    window.difficulty_var.set('困难')
    window._profile_changed()
    window.locked_layout = locked_layout
    retrieval = LayoutRetrievalResult(backend='pure_structure',timings_ms={'total':1},candidates=[
        LayoutCandidate(1,'layout_03','layout_03/layout_03_1st.png','a',.97,.97,0,0,'候选',{'floor':'1st'}),
        LayoutCandidate(2,'layout_27','layout_27/layout_27_2nd.png','b',.96,.96,0,0,'候选',{'floor':'2nd'})])
    result=process_screenshot(np.zeros((80,80,3),np.uint8),catalog=window.catalog,profile=window.profile,
        opacity=1,recognize_only=True,layout_recognition_backend='pure_structure',
        layout_retriever=SimpleNamespace(recognize=lambda *a:retrieval))
    result.session_id=window.session_id
    prompts=[]
    statuses=[]
    monkeypatch.setattr(window, '_set_status', lambda text, *args: statuses.append(text))
    monkeypatch.setattr(app.messagebox,'confirm_layout',lambda items,*a,**kw:prompts.append(items))
    window._pending_confirmation=(window.map_generation,result,window.session_id,window.profile.key)
    window._confirm_recognition(window.map_generation,result)
    assert [item['preview_path'].name for item in prompts[0]] == ['layout_03_1st.png','layout_27_2nd.png']
    assert [item['preview_floor'] for item in prompts[0]] == ['1st','2nd']
    assert window.locked_layout == locked_layout
    assert ('仍锁定 layout_07' if locked_layout else '布局尚未锁定') in statuses[-1]
    assert '已取消锁定' not in statuses[-1]


def test_close_scores_explain_local_ambiguity_without_blocking_manual_choice(qt_app, tmp_path):
    dialog=LayoutConfirmationDialog([
        dict(layout='layout_03',score=.970,preview_floor='1st',recognition_zoom='50_percent',preview_path=tmp_path/'first.png'),
        dict(layout='layout_27',score=.966,preview_floor='1st',preview_path=tmp_path/'second.png')],None)
    texts=[label.text() for label in dialog.findChildren(QLabel)]
    assert any('分数接近' in text for text in texts)
    assert any('50%' in text and '缩到最小再配准' in text for text in texts)
    assert any('一楼底图' in text for text in texts)
    assert not any('二楼底图' in text for text in texts)
    assert dialog.confirm_button.isEnabled()
    dialog.show()
    qt_app.processEvents()
    dialog.grab().save(str(tmp_path/'ambiguous_floor_confirmation.png'))
    dialog.reject()
    dialog.deleteLater()
