import queue
from types import SimpleNamespace
from src import route_projection_core as app


class InlineThread:
    def __init__(self,*,target,**kwargs):self.target=target
    def start(self):self.target()


class FakeApp:
    def __init__(self):
        self.settings={'game_window_titles':['game'],'capture_reference_size':[2560,1440],
            'registration_backend':'template','layout_recognition_backend':'icon_structure','auto_project':True,'auto_floor':False,'capture_roi':[901,185,1180,1050],
            'template_route_versions':{},'opacity':.72,'map_capture_delay_ms':35}
        self.map_generation=7;self.session_id='session';self.closed=False;self.busy=False
        self.map_state=app.MapToggleState(is_open=True);self.projection_delay_pending=True
        self.projection_triggered_at=100.;self.locked_layout='layout_16';self.settings_page=None
        self.results=queue.Queue();self.requests=[];self.scheduled=[];self.hidden=0
        self.profile=SimpleNamespace(key='a_bringer_of_doom/nightmare')
        self.catalog=object()
        self.capture_policy=SimpleNamespace(begin_capture=lambda _token: None, end_capture=lambda _token: None)
    def _log(self,*args):pass
    def _refresh_status(self):pass
    def _gui_diagnostic_snapshot(self):return {'log_entries': 0, 'activity': {}}
    def _set_status(self,*args):pass
    def hide_overlay(self):self.hidden+=1
    def after(self,delay,callback):self.scheduled.append((delay,callback))
    def request_work(self,mode,*,auto):self.requests.append((mode,auto))
