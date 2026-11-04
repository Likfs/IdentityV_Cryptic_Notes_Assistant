from pathlib import Path
from threading import Event, get_ident

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src import asset_sync
from src.startup import StartupDialog, synchronize_with_progress


def test_startup_progress_runs_in_background_and_keeps_event_loop_responsive(qt_app, tmp_path, monkeypatch):
    main_thread = get_ident()
    release = Event()
    observed = []
    quit_on_close = qt_app.quitOnLastWindowClosed()

    def synchronize(root, *, progress):
        assert get_ident() != main_thread and root == tmp_path
        progress('校验素材 · 测试地图/normal/layout_01.png', 3, 10)
        assert release.wait(3)
        return ['同步完成']

    def inspect():
        dialog = next((w for w in QApplication.topLevelWidgets() if isinstance(w, StartupDialog) and w.isVisible()), None)
        if dialog is not None and dialog.progress_bar.value() == 3:
            observed.append(dialog.stage_label.text())
            assert dialog.progress_bar.maximum() == 10
            assert dialog.grab().save(str(tmp_path/'startup_progress.png'))
            release.set()
        else:
            QTimer.singleShot(10, inspect)

    monkeypatch.setattr(asset_sync, 'synchronize', synchronize)
    QTimer.singleShot(10, inspect)
    assert synchronize_with_progress(tmp_path) == ['同步完成']
    assert observed and '测试地图' in observed[0]
    assert qt_app.quitOnLastWindowClosed() == quit_on_close


@pytest.mark.parametrize('failure', [False, True])
def test_startup_can_cancel_or_display_worker_error(qt_app, tmp_path, monkeypatch, failure):
    observed = []
    def synchronize(root, *, progress):
        if failure:
            raise ValueError('测试素材缺少有效 PNG')
        for index in range(300):
            progress('校验素材', index, 300)
            Event().wait(.01)
        pytest.fail('Cancellation was ignored')

    def close_when_ready():
        dialog = next((w for w in QApplication.topLevelWidgets() if isinstance(w, StartupDialog) and w.isVisible()), None)
        if dialog is None or (failure and dialog.title_label.text() != '启动失败'):
            QTimer.singleShot(10, close_when_ready)
            return
        observed.append(dialog.stage_label.text())
        dialog.close()

    monkeypatch.setattr(asset_sync, 'synchronize', synchronize)
    QTimer.singleShot(10, close_when_ready)
    if failure:
        with pytest.raises(ValueError, match='测试素材缺少有效 PNG'):
            synchronize_with_progress(tmp_path)
        assert observed == ['测试素材缺少有效 PNG']
    else:
        assert synchronize_with_progress(tmp_path) is None
        assert observed


def test_cancel_real_parallel_audit_waits_for_workers_without_publishing(qt_app, tmp_path, monkeypatch):
    import numpy as np
    from src import asset_audit
    from tests.support.aligned_runtime import difficulty, publish, write_png

    root = tmp_path/'parallel'
    definition = difficulty('hard', '困难', ('1st',), {'clean': '底图', 'route': '路线'})
    for index in range(1, 5):
        layout = f'layout_{index:02d}'
        for version in definition['asset_versions']:
            write_png(root/version['source_root']/layout/f'{layout}_1st.png', np.full((40, 60, 3), 100, np.uint8))
    publish(root, dict(schema_version=2, maps=[dict(id='a_bringer_of_doom', name='地图', difficulties=[definition])]))
    original = {name: (root/name).read_bytes() for name in asset_sync.CONFIG_FILES}
    release = Event()
    started = []
    read = asset_audit._read
    def blocked(path):
        started.append(path)
        assert release.wait(3)
        return read(path)
    monkeypatch.setattr(asset_audit, '_read', blocked)
    observed = []
    def cancel():
        dialog = next((w for w in QApplication.topLevelWidgets() if isinstance(w, StartupDialog) and w.isVisible()), None)
        if dialog is None or len(started) < 4:
            QTimer.singleShot(10, cancel)
            return
        observed.append(dialog.stage_label.text())
        dialog.close()
        release.set()
    QTimer.singleShot(10, cancel)
    assert synchronize_with_progress(root) is None
    assert observed and len(started) == 4
    assert {name: (root/name).read_bytes() for name in asset_sync.CONFIG_FILES} == original
    assert not (root/'runtime/cache/asset_audit/png_validation.json').exists()


def test_empty_catalog_starts_with_disabled_actions_and_preserves_personal_settings(qt_app, tmp_path, monkeypatch):
    import json
    from src import __main__ as entry
    from src import app_catalog, route_projection_app
    from src.startup import EmptyAssetsWindow
    from tests.support.aligned_runtime import activate, publish

    root = publish(tmp_path/'package', dict(schema_version=2, maps=[]))
    activate(root, monkeypatch)
    personal = tmp_path/'personal.json'
    raw = json.dumps(dict(map_id='removed_map', difficulty_id='nightmare', opacity=.61,
                          template_route_versions={'removed_map/nightmare': '中文路线'})).encode()
    personal.write_bytes(raw)
    monkeypatch.setattr(entry, 'PROJECT_ROOT', root)
    monkeypatch.setattr(entry, 'load_settings', lambda catalog: app_catalog.load_settings(catalog, personal))
    monkeypatch.setattr(entry.sys, 'argv', ['src', '--smoke'])
    monkeypatch.setattr(route_projection_app, 'RouteProjectionApp', lambda: pytest.fail('Empty startup built a profile window'))
    observed = []
    def inspect():
        window = next((w for w in QApplication.topLevelWidgets() if isinstance(w, EmptyAssetsWindow) and w.isVisible()), None)
        if window is None:
            QTimer.singleShot(10, inspect)
            return
        observed.append((window.title_label.text(), window.recognize_button.isEnabled(),
                         window.project_button.isEnabled(), window.grab().save(str(tmp_path/'empty_assets.png'))))
        window.close()
    QTimer.singleShot(10, inspect)
    entry.main()
    assert observed == [('暂无地图素材', False, False, True)]
    assert personal.read_bytes() == raw
