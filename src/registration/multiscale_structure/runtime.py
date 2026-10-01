"""Multiscale structure registration with lazy GPU loading and CPU fallback.

CPU is independently selectable without importing the GPU implementation.
GPU faults disable CUDA for this process; a rejected pose is not a GPU fault.
"""
from dataclasses import fields, replace
from threading import RLock
from time import perf_counter
import cv2
import numpy as np
from .cpu.runtime import register_screenshot as cpu_register
from src.registration.runtime_registration import RuntimeRegistrationResult

_lock = RLock()
_initialized = False
_gpu_register = None
_status = dict(effective_backend='cpu', device=None, fallback_reason=None, initialization_ms=0.)


def _validate_backend(backend):
    if backend not in ('auto', 'cpu', 'template'):
        raise ValueError('配准后端必须为 auto、cpu 或 template')


def _sampling_supported():
    field = np.tile(np.arange(8, dtype=np.float32), (3, 1))
    x = np.array([[1.01, 1.02, 1.03, 1.04, 1.05, 1.06]], np.float32)
    y = np.ones_like(x)
    continuous = cv2.remap(np.stack([field]*3, axis=2), x, y, cv2.INTER_LINEAR)[:, :, 0]
    quantized = cv2.remap(np.stack([field]*2, axis=2), x, y, cv2.INTER_LINEAR)[:, :, 0]
    return np.array_equal(continuous, x) and np.array_equal(quantized, np.rint(x*32)/32)


def _disable(reason):
    global _gpu_register
    _gpu_register = None
    _status.update(effective_backend='cpu', fallback_reason=reason)


def initialize(backend='auto'):
    global _initialized, _gpu_register
    _validate_backend(backend)
    if backend == 'template':
        from src.registration.template_matching.runtime import initialize as template_initialize
        return template_initialize()
    if backend == 'cpu':
        return dict(effective_backend='cpu', device=None, fallback_reason=None, initialization_ms=0.)
    with _lock:
        if not _initialized:
            start = perf_counter()
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA 不可用')
                if not _sampling_supported():
                    raise RuntimeError('当前 OpenCV 插值路径不兼容 GPU 配准')
                torch.ones(1024, device='cuda').square().sum().item()
                from src.registration.multiscale_structure.gpu.acceptance.runtime import register_screenshot as gpu_register
                from src.registration.multiscale_structure.gpu.core.cuda_color import ColorMixture
                ColorMixture(np.zeros((320, 320, 3), np.float32)).evaluate(
                    np.array([30, 80, 120], np.float32), clip=True)
                torch.cuda.synchronize()
                _gpu_register = gpu_register
                _status.update(effective_backend='cuda', device=torch.cuda.get_device_name(0), fallback_reason=None)
            except (ImportError, OSError, RuntimeError) as exc:
                _disable(str(exc))
            _status['initialization_ms'] = (perf_counter()-start)*1000
            _initialized = True
        return dict(_status)


def _gpu_prepared(prepared):
    # Dataclass replacement drops dynamic CPU caches. Source arrays are only
    # read; lazy GPU fields live on a separate processed-template owner.
    value = getattr(prepared, '_gpu_structure_prepared', None)
    if value is None:
        value = replace(prepared, processed=replace(prepared.processed))
        prepared._gpu_structure_prepared = value
    return value


def prewarm_template(prepared, backend='auto'):
    _validate_backend(backend)
    if backend == 'template':
        return initialize('template')
    if backend == 'cpu':
        return initialize('cpu')
    with _lock:
        initialize()
        if _gpu_register is not None:
            try:
                from src.registration.multiscale_structure.gpu.search_core.template import prepared_adapter
                from src.registration.multiscale_structure.gpu.core.cuda_scoring import _template_fields
                from src.registration.multiscale_structure.gpu.core.credible_geometry import _fields
                template, template_lock = prepared_adapter(_gpu_prepared(prepared))
                with template_lock:
                    _template_fields(template.processed)
                    _fields(template.processed, True)
                import torch
                torch.cuda.synchronize()
            except (ImportError, OSError, RuntimeError) as exc:
                _disable(f'模板 GPU 预热失败：{exc}')
        return dict(_status)


def register_screenshot(screenshot, prepared, prior_screenshot_to_clean=None,
                        temporal_evidence=None, soft_pose_hint=None, *, backend='auto'):
    _validate_backend(backend)
    if backend == 'template':
        from src.registration.template_matching.runtime import register_screenshot as baseline_register
        return baseline_register(screenshot, prepared)
    start = perf_counter()
    if backend == 'cpu':
        result = cpu_register(screenshot, prepared)
        info = initialize('cpu')
    else:
        # Serialize prewarming and registration, including evidence branches.
        # An explicit CPU selection bypasses this GPU lock.
        with _lock:
            initialize()
            result = None
            if _gpu_register is not None:
                try:
                    result = _gpu_register(screenshot, _gpu_prepared(prepared))
                except (ImportError, OSError, RuntimeError) as exc:
                    _disable(f'GPU 配准失败：{exc}')
            if result is None:
                result = cpu_register(screenshot, prepared)
            info = dict(_status)
    if not isinstance(result, RuntimeRegistrationResult):
        result = RuntimeRegistrationResult(**{f.name: getattr(result, f.name) for f in fields(RuntimeRegistrationResult)})
    info.update(requested_backend=backend, implementation='multiscale_structure_gpu' if info['effective_backend']=='cuda' else 'multiscale_structure_cpu',
                cpu_double_verification=False)
    return replace(result, validation={**result.validation, 'registration_backend': info},
                   timings={**result.timings, 'total': (perf_counter()-start)*1000})
