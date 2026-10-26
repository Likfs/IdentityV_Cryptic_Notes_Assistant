import pytest
from PyQt6.QtWidgets import QRadioButton
from src import route_projection_app as app


def activate(window, text):
    window.layout_combo.setCurrentText(text)
    window.layout_combo.textActivated.emit(text)


def test_automatic_layout_display_never_opens_manual_confirmation(window, monkeypatch):
    monkeypatch.setattr(app.messagebox, 'askyesno', lambda *a, **k: pytest.fail('automatic update prompted'))
    assert not window.findChildren(QRadioButton)
    window.layout_var.set('layout_12')
    window._update_assets()
    assert window.layout_combo.currentText() == 'layout_12'


@pytest.mark.parametrize('confirmed', [True, False])
def test_manual_choice_requires_confirmation_and_cancel_restores_lock(window, monkeypatch, confirmed):
    prompts=[]
    monkeypatch.setattr(app.messagebox, 'askyesno', lambda *a, **k: prompts.append((a,k)) or confirmed)
    generation=window.map_generation;session=window.session_id
    activate(window, 'layout_12')
    assert len(prompts)==1 and 'layout_12' in prompts[0][0][1]
    assert prompts[0][1]['entrance_path'].name=='layout_12.png'
    expected='layout_12' if confirmed else 'layout_11'
    assert window.locked_layout==window.layout_var.get()==window.layout_combo.currentText()==expected
    assert window.map_generation==generation+1 and window.session_id!=session
    assert window.last_result is None and window.capture_panel._pixmap is None
    assert window.live_projection is None


def test_automatic_choice_releases_manual_lock_without_confirmation(window, monkeypatch):
    monkeypatch.setattr(app.messagebox,'askyesno',lambda *a,**k:pytest.fail('automatic option prompted'))
    activate(window, app.AUTO)
    assert window.locked_layout is None and window.layout_var.get()==app.AUTO
    assert window.layout_combo.currentText()==app.AUTO


def test_cancel_from_unlocked_state_restores_automatic(window, monkeypatch):
    window.clear_layout_lock();window._update_assets()
    monkeypatch.setattr(app.messagebox,'askyesno',lambda *a,**k:False)
    activate(window,'layout_12')
    assert window.locked_layout is None and window.layout_combo.currentText()==app.AUTO


def test_map_change_during_manual_dialog_prevents_late_lock(window, monkeypatch):
    def confirm(*a, **k):
        window.map_generation+=1
        return True
    monkeypatch.setattr(app.messagebox,'askyesno',confirm)
    activate(window,'layout_12')
    assert window.locked_layout=='layout_11' and window.layout_combo.currentText()=='layout_11'
