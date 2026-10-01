"""Production CPU retention, GPU failure semantics and backend preferences."""
from dataclasses import replace
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.registration.multiscale_structure import runtime
from src.registration.runtime_registration import RuntimeRegistrationResult
from src.app_catalog import MapCatalog, load_settings

ROOT = Path(__file__).resolve().parents[1]


def result(accepted=True):
    return RuntimeRegistrationResult(accepted, [] if accepted else ['ambiguous'], None, None, None, None,
        {'registration_algorithm':'multiscale_structure','route_validation_enabled':False,'trusted_history_eligible':False},
        {'total':1.},None,None,None)


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(runtime,'_gpu_register',runtime._gpu_register)
    monkeypatch.setattr(runtime,'_initialized',True)
    monkeypatch.setattr(runtime,'_status',dict(effective_backend='cuda',device='test GPU',fallback_reason=None,initialization_ms=0.))
    monkeypatch.setattr(runtime,'_gpu_prepared',lambda prepared:prepared)
    return monkeypatch


def test_manual_cpu_never_uses_gpu_even_after_gpu_ready(isolated):
    isolated.setattr(runtime,'_gpu_register',lambda *a:pytest.fail('CPU preference used GPU'))
    isolated.setattr(runtime,'cpu_register',lambda *a:result())
    actual=runtime.register_screenshot(None,None,backend='cpu')
    assert actual.validation['registration_backend']['effective_backend']=='cpu'
    assert actual.validation['registration_backend']['fallback_reason'] is None


def test_gpu_rejection_does_not_retry_cpu_or_override_policy(isolated):
    isolated.setattr(runtime,'_gpu_register',lambda *a:result(False))
    isolated.setattr(runtime,'cpu_register',lambda *a:pytest.fail('Rejection retried on CPU'))
    actual=runtime.register_screenshot(None,None)
    assert not actual.accepted and actual.rejection_reasons==['ambiguous']
    assert actual.validation['registration_backend']['effective_backend']=='cuda'
    assert not actual.validation['trusted_history_eligible']


def test_runtime_fault_falls_back_to_original_cpu_and_sticks(isolated):
    calls=[]
    def failing(*args):
        calls.append('gpu');raise RuntimeError('out of memory')
    def cpu(*args):
        calls.append('cpu');return result()
    isolated.setattr(runtime,'_gpu_register',failing)
    isolated.setattr(runtime,'cpu_register',cpu)
    for _ in range(2):
        actual=runtime.register_screenshot(None,None)
        assert actual.validation['registration_backend']['effective_backend']=='cpu'
        assert 'out of memory' in actual.validation['registration_backend']['fallback_reason']
    assert calls==['gpu','cpu','cpu']


def test_cuda_unavailable_initialization_falls_back_without_loading_gpu(isolated):
    isolated.setattr(runtime,'_initialized',False)
    isolated.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    isolated.setattr(runtime,'cpu_register',lambda *a:result())
    actual=runtime.register_screenshot(None,None)
    assert actual.validation['registration_backend']['effective_backend']=='cpu'
    assert actual.validation['registration_backend']['fallback_reason']=='CUDA 不可用'


def test_incompatible_opencv_stops_before_gpu_import(isolated):
    isolated.setattr(runtime,'_initialized',False)
    isolated.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    isolated.setattr(runtime,'_sampling_supported',lambda:False)
    assert runtime.initialize()['effective_backend']=='cpu'
    assert '插值' in runtime.initialize()['fallback_reason']


@pytest.mark.parametrize('saved,expected',[({},'template'),({'registration_backend':'cpu'},'cpu'),({'registration_backend':'auto'},'auto'),({'registration_backend':'bad'},'template')])
def test_settings_preserve_cpu_and_upgrade_missing_preference(tmp_path,saved,expected):
    path=tmp_path/'settings.json';path.write_text(json.dumps(saved),encoding='utf8')
    assert load_settings(MapCatalog(),path)['registration_backend']==expected


def test_cpu_execution_needs_neither_torch_nor_temporary_modules():
    code='''
import sys,importlib.abc,tempfile
class BlockGPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith(('torch.','tmp.','src.registration.multiscale_structure.gpu')):
   raise AssertionError('CPU imported GPU or experiment: '+fullname)
sys.meta_path.insert(0,BlockGPU())
from src.registration.multiscale_structure.runtime import register_screenshot,initialize,prewarm_template
from src.registration.runtime_registration import prepare_registration_template
from src.app_catalog import MapCatalog
from pathlib import Path
import numpy as np
c=MapCatalog();p=c.profile('a_bringer_of_doom','nightmare')
with tempfile.TemporaryDirectory() as folder:
 t=prepare_registration_template('layout_05','2nd',c.clean_template(p,'layout_05','2nd'),Path(folder))
 r=register_screenshot(np.zeros((400,500,3),np.uint8),t,backend='cpu')
 assert not r.accepted and r.validation['registration_backend']['effective_backend']=='cpu'
 assert initialize('cpu')['effective_backend']=='cpu'
 assert prewarm_template(t,'cpu')['effective_backend']=='cpu'
'''
    process=subprocess.run([sys.executable,'-X','utf8','-B','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    assert process.returncode==0,process.stdout+process.stderr


def test_settings_ui_saves_cpu_and_invalidates_inflight_generation(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PyQt6.QtWidgets import QApplication
    from src import route_projection_app as app
    class NoHotkeys:
        events=None;error=None
        def __init__(self,*args):pass
        def start(self):pass
        def stop(self):pass
    monkeypatch.setattr(app,'GlobalActionHotkeys',NoHotkeys)
    monkeypatch.setattr(app.RouteProjectionApp,'_start_model_load',lambda self:None)
    monkeypatch.setattr(app.RouteProjectionApp,'_start_registration_backend',lambda self:None)
    qt=QApplication.instance() or QApplication([])
    path=tmp_path/'settings.json';window=app.RouteProjectionApp(settings_path=path)
    try:
        window.open_settings();page=window.settings_dialog
        generation=window.map_generation
        assert page.registration_backend_combo.currentData()=='template'
        page.registration_backend_combo.setCurrentIndex(page.registration_backend_combo.findData('cpu'))
        assert page._persist_settings()
        assert window.settings['registration_backend']=='cpu'
        assert load_settings(MapCatalog(),path)['registration_backend']=='cpu'
        assert window.map_generation>generation
    finally:
        window.close()
