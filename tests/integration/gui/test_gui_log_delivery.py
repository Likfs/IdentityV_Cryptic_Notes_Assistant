import pytest
from src import route_projection_app as app


def flush(window):
    # The redesigned workspace defers painting while the log drawer is folded.
    window.log_section.setExpanded(True)
    window._log_last_flush_at = 0.
    window._flush_logs()


def test_new_logs_are_batched_and_incremental_plain_text(window, monkeypatch):
    window._clear_logs()
    window._log('first <b>literal</b>')
    assert window.log.toPlainText() == ''
    flush(window)
    assert '<b>literal</b>' in window.log.toPlainText()
    monkeypatch.setattr(window.log, 'setPlainText', lambda *_: pytest.fail('history rebuilt on append'))
    window._log('second\nthird')
    assert 'second' not in window.log.toPlainText()
    window._flush_logs()  # Throttled while a batch is pending.
    assert 'second' not in window.log.toPlainText()
    flush(window)
    assert window.log.toPlainText().endswith('second\nthird')
    assert window.log.toPlainText().count('literal') == 1
    assert window._gui_activity['log_refresh']['calls'] >= 2


@pytest.mark.parametrize('blocked', ['hidden', 'minimized', 'busy', 'map_probe'])
def test_deferred_log_catches_up_with_bounded_tail_and_full_export(window, monkeypatch, tmp_path, blocked):
    window._clear_logs()
    if blocked == 'hidden': window.log.hide()
    if blocked == 'minimized': monkeypatch.setattr(window, 'isMinimized', lambda: True)
    if blocked == 'busy': window.busy = True
    if blocked == 'map_probe': window.projection_delay_pending = True
    for index in range(2500):
        window._log(f'entry-{index:04d}\ncontinuation-{index:04d}')
    flush(window)
    assert window.log.toPlainText() == '' and window._log_dirty
    window.log.show()
    monkeypatch.setattr(window, 'isMinimized', lambda: False)
    window.busy = window.projection_delay_pending = False
    flush(window)
    text = window.log.toPlainText()
    assert window.log.document().blockCount() == app.LOG_VISIBLE_BLOCK_LIMIT
    assert 'entry-0000' not in text and 'entry-2499' in text
    assert len(window.log_lines) == 2500
    path = tmp_path / 'full.txt'
    monkeypatch.setattr(app.QFileDialog, 'getSaveFileName', lambda *_: (str(path), ''))
    window._export_logs()
    exported = path.read_text(encoding='utf-8')
    assert 'entry-0000' in exported and 'entry-2499' in exported


def test_filter_rebuilds_recent_matching_history_and_clear_keeps_lifetime_count(window):
    window._clear_logs()
    window._log('很早的失败')
    for index in range(2100): window._log(f'normal-{index}')
    flush(window)
    assert '很早的失败' not in window.log.toPlainText()
    window.log_filter.setCurrentText('错误')
    flush(window)
    assert '很早的失败' in window.log.toPlainText()
    window._log('normal-unmatched')
    window._log('新的错误')
    flush(window)
    assert 'normal' not in window.log.toPlainText()
    assert '新的错误' in window.log.toPlainText()
    window.log_filter.setCurrentText('警告')
    flush(window)
    assert window.log.toPlainText() == ''
    lifetime = window._gui_diagnostic_snapshot()['log_entries_since_start']
    window._clear_logs()
    window.log_filter.setCurrentText('全部日志')
    window._log('after-clear')
    flush(window)
    assert 'normal' not in window.log.toPlainText()
    assert window.log.toPlainText().endswith('after-clear')
    snapshot = window._gui_diagnostic_snapshot()
    assert snapshot['log_entries'] == 1
    assert snapshot['log_entries_since_start'] == lifetime + 1
