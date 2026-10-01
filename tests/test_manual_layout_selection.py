"""Manual selection requires confirmation; automatic updates never prompt."""
import pytest
from PyQt6.QtWidgets import QRadioButton
from src import route_projection_app as app
from src.app_catalog import MapCatalog, load_settings
from test_reference_map_ui import window


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
    prompts=[];preloaded=[]
    monkeypatch.setattr(app.messagebox, 'askyesno', lambda *a, **k: prompts.append((a,k)) or confirmed)
    monkeypatch.setattr(window, '_preload_locked_layout', preloaded.append)
    generation=window.map_generation;session=window.session_id
    activate(window, 'layout_12')
    assert len(prompts)==1 and 'layout_12' in prompts[0][0][1]
    assert prompts[0][1]['entrance_path'].name=='layout_12.png'
    expected='layout_12' if confirmed else 'layout_11'
    assert window.locked_layout==window.layout_var.get()==window.layout_combo.currentText()==expected
    assert window.map_generation==generation+1 and window.session_id!=session
    assert preloaded==(['layout_12'] if confirmed else [])
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
    monkeypatch.setattr(window,'_preload_locked_layout',lambda *a:pytest.fail('stale selection preloaded'))
    activate(window,'layout_12')
    assert window.locked_layout=='layout_11' and window.layout_combo.currentText()=='layout_11'


def test_previous_backend_preference_loads_with_canonical_id(tmp_path):
    path=tmp_path/'settings.json'
    path.write_text('{"registration_backend":"legacy"}',encoding='utf-8')
    assert load_settings(MapCatalog(),path)['registration_backend']=='template'
