import os
import time


def record(source, down, vk=ord('G'), *, title='IdentityV', pid=None, age=0.):
    source._record(vk, down, (123, title, os.getpid()+1 if pid is None else pid),
                   detected_at=time.perf_counter()-age)


def install(window, source, *, template=False):
    if template:
        window.settings['registration_backend'] = 'template'
    window.map_keys = source
    source._notify = lambda: window._map_key_bridge.ready.emit(None)
