"""Physical monitor dimensions for the original 2560 x 1440 convention."""
import ctypes
from src.capture.monitors import MonitorInfo, MonitorEnumProc, monitor_api


def factor_for_size(width,height):
    ratio=min(width/2560.,height/1440.)
    return 1./ratio if ratio>0 else 1.

def screen_info(name=''):
    try:
        from src.capture.region import enable_physical_pixel_coordinates
        enable_physical_pixel_coordinates()
        user=monitor_api()
        monitors=[]
        @MonitorEnumProc
        def visit(handle,dc,rect,param):
            item=MonitorInfo();item.cbSize=ctypes.sizeof(MonitorInfo)
            if user.GetMonitorInfoW(handle,ctypes.byref(item)):
                monitors.append(dict(screen_name=item.szDevice,width=item.rcMonitor.right-item.rcMonitor.left,
                    height=item.rcMonitor.bottom-item.rcMonitor.top,primary=bool(item.dwFlags&1)))
            return True
        user.EnumDisplayMonitors(None,None,visit,0)
        selected=next((m for m in monitors if m['screen_name']==name),None) if name else next((m for m in monitors if m['primary']),None)
        if selected:return dict(ok=True,**selected,factor=factor_for_size(selected['width'],selected['height']))
    except (AttributeError,OSError):pass
    return dict(ok=False,screen_name=name,width=0,height=0,factor=1.)
