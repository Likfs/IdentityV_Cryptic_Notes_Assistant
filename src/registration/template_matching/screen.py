"""Physical monitor dimensions for the original 2560 x 1440 convention."""
import ctypes
from ctypes import wintypes

def factor_for_size(width,height):
    ratio=min(width/2560.,height/1440.)
    return 1./ratio if ratio>0 else 1.

def screen_info(name=''):
    try:
        from src.capture.region import enable_physical_pixel_coordinates
        enable_physical_pixel_coordinates()
        class Info(ctypes.Structure):
            _fields_=[('size',wintypes.DWORD),('monitor',wintypes.RECT),('work',wintypes.RECT),
                      ('flags',wintypes.DWORD),('device',wintypes.WCHAR*32)]
        user=ctypes.windll.user32
        user.GetMonitorInfoW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Info)]
        monitors=[]
        callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HANDLE,wintypes.HDC,ctypes.POINTER(wintypes.RECT),wintypes.LPARAM)
        @callback_type
        def visit(handle,dc,rect,param):
            item=Info();item.size=ctypes.sizeof(Info)
            if user.GetMonitorInfoW(handle,ctypes.byref(item)):
                monitors.append(dict(screen_name=item.device,width=item.monitor.right-item.monitor.left,
                    height=item.monitor.bottom-item.monitor.top,primary=bool(item.flags&1)))
            return True
        user.EnumDisplayMonitors(None,None,visit,0)
        selected=next((m for m in monitors if m['screen_name']==name),None) if name else next((m for m in monitors if m['primary']),None)
        if selected:return dict(ok=True,**selected,factor=factor_for_size(selected['width'],selected['height']))
    except (AttributeError,OSError):pass
    return dict(ok=False,screen_name=name,width=0,height=0,factor=1.)
