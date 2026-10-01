"""Saved-scale 定尺度模板匹配 adapter; live projection uses direct original-image coordinates."""
from threading import RLock
from time import perf_counter
import hashlib
import json
import cv2
import numpy as np
from src.app_catalog import PROJECT_ROOT
from src.registration.runtime_registration import RuntimeRegistrationResult
from . import matcher
from .screen import screen_info

_lock=RLock()
_config_stamp=None
_config=None
_verified={}

def initialize():
    return dict(effective_backend='template',device=None,fallback_reason=None,initialization_ms=0.,
        requested_backend='template',version='fixed_scale_template',cpu_double_verification=False)

def _fingerprint(path,expected):
    stat=path.stat();stamp=(str(path),stat.st_mtime_ns,stat.st_size,expected)
    if stamp not in _verified:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'尺度对应资源已改变，请重新导入：{path.name}')
        _verified[stamp]=True

def _load_entry(prepared):
    global _config,_config_stamp
    path=PROJECT_ROOT/'configs/template_matching.json'
    stat=path.stat();stamp=(str(path),stat.st_mtime_ns,stat.st_size)
    if stamp!=_config_stamp:
        _config=json.loads(path.read_text(encoding='utf-8'));_config_stamp=stamp
        _verified.clear();matcher._map_cache.clear()
    key=prepared.path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    entry=_config['entries'].get(key)
    if entry is None:raise ValueError('当前地图尚无已核实的完整原图及尺度，请先导入标定')
    source=PROJECT_ROOT/entry['source_image']
    _fingerprint(source,entry['source_sha256'])
    _fingerprint(prepared.path,entry['clean_sha256'])
    _fingerprint(PROJECT_ROOT/entry['calibration'],entry['calibration_sha256'])
    return entry,source,_config

def register_screenshot(screenshot,prepared):
    started=perf_counter()
    validation=dict(registration_algorithm='fixed_scale_template',route_validation_enabled=False,
        trusted_history_eligible=False,projection_scope='full',
        acceptance_policy='template_candidate_without_geometry_validation',registration_backend=initialize())
    timings=dict(method='fixed-scale-template',calibration=0.,matching=0.)
    try:
        with _lock:
            entry,path,config=_load_entry(prepared)
            screen=screen_info(config.get('capture',{}).get('screen_name',''))
            saved_scale=float(entry['saved_scale']);scale=saved_scale*screen['factor']
            if not np.isfinite(scale) or scale<=0:raise ValueError('原图尚未保存有效尺度')
            mode=int(config['perf_mode'])
            if not 0<=mode<len(matcher.PERF_MODES):raise ValueError('性能档位无效')
            ds=matcher.PERF_MODES[mode]['ds'];matcher.TEMPLATE_DS=ds
            matching_start=perf_counter()
            match=matcher._match_template(screenshot,str(path),scale)
            timings['matching']=(perf_counter()-matching_start)*1000
        if not np.isfinite(match['score']) or not np.isfinite(match['location']).all():
            raise ValueError('定尺度模板匹配未返回有限分数或坐标')
        x,y=match['location']
        screenshot_to_original=np.float64([[scale,0,x],[0,scale,y],[0,0,1]])
        clean_to_original=np.asarray(entry['clean_to_original'],np.float64)
        if clean_to_original.shape!=(3,3) or not np.isfinite(clean_to_original).all():
            raise ValueError('底图到完整原图标定矩阵无效')
        # Preserve the full offline homography. 定尺度模板匹配 itself remains scale+translation.
        clean_to_screen=np.linalg.inv(screenshot_to_original)@clean_to_original
        screenshot_to_clean=np.linalg.inv(clean_to_screen)
        aligned=cv2.warpPerspective(prepared.raw_image,clean_to_screen,(screenshot.shape[1],screenshot.shape[0]))
        preview=cv2.addWeighted(screenshot,.6,aligned,.4,0)
        validation['template']=dict(scale=scale,saved_scale=saved_scale,screen=screen,ds=ds,perf_mode=mode,
            score=float(match['score']),location=[int(x),int(y)],target='original_source_image',
            source_reference=entry['source_reference'],scale_policy='saved_original_scale_with_screen_factor',
            source_sha256=entry['source_sha256'],screenshot_to_original=screenshot_to_original.tolist(),
            clean_to_original=clean_to_original.tolist())
        accepted,reasons=True,[]
    except (OSError,ValueError,KeyError,cv2.error,np.linalg.LinAlgError) as exc:
        accepted,reasons=False,[f'定尺度模板匹配 运行失败：{exc}']
        clean_to_screen=screenshot_to_clean=preview=None
    timings['total']=(perf_counter()-started)*1000
    return RuntimeRegistrationResult(accepted=accepted,rejection_reasons=reasons,clean_to_screen=clean_to_screen,
        screenshot_to_clean=screenshot_to_clean,observation_bbox=None,match=None,validation=validation,timings=timings,
        observation_image=None,observation_mask=None,alignment_preview=preview)
