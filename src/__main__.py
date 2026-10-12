from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import traceback
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import LOG_ROOT
from src.version import APP_NAME, APP_VERSION
from src.app_catalog import FLOOR_NAMES, MapCatalog, load_settings
from src.layout_retrieval import RESOURCE_ROOT
from src.projection_workflow import process_screenshot, save_workflow_result


def _read_image(path: Path):
    image = cv2.imdecode(__import__("numpy").fromfile(str(path), dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取截图：{path}")
    return image


def _acquire_app_mutex():
    if sys.platform != "win32":
        return None, False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    name = "Local\\IdentityVRouteProjection." + hashlib.sha256(
        str(PROJECT_ROOT.resolve()).casefold().encode("utf-8")
    ).hexdigest()[:20]
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        raise OSError(ctypes.get_last_error(), "无法创建程序实例锁")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        ctypes.windll.user32.MessageBoxW(
            None, "路线投影程序已经打开。请使用现有窗口；若刚更新程序，请先关闭旧窗口再重新启动。",
            "路线投影", 0x40,
        )
        return None, True
    return handle, False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}",
                        help="显示程序版本并退出")
    parser.add_argument("--check", action="store_true", help="检查地图、难度与模板资源")
    parser.add_argument("--project-once", type=Path, help="使用本地截图运行一次配准投影")
    parser.add_argument("--layout", help="指定布局，例如 layout_05；必须指定已人工确认的布局")
    parser.add_argument("--route-version", help="显示版本：route_clean / route_lhp_solo / route_zhanshi_fast；模板模式另支持 route_lhp_solo_clean / route_zhanshi_fast_clean")
    parser.add_argument("--output", type=Path, help="一次性运行的诊断输出目录")
    parser.add_argument("--registration-backend", choices=("template",),
                        help="配准方式：template 定尺度模板匹配")
    parser.add_argument("--smoke", action="store_true", help="启动窗口并在 2 秒后关闭，用于界面烟雾检查")
    args = parser.parse_args()
    catalog = MapCatalog()
    settings = load_settings(catalog)
    profile = catalog.profile(settings["map_id"], settings["difficulty_id"])

    if args.check:
        from src.registration.template_matching.direct import check_original_resources
        summary = []
        for item in catalog.profiles.values():
            layouts = catalog.layouts(item)
            status = {floor: 0 for floor in FLOOR_NAMES}
            review = []
            versions = {}
            for layout in layouts:
                for available in catalog.variants(item, layout):
                    version = versions.setdefault(available.variant_id, {
                        "name": available.label, "layouts": 0, "pass_floors": 0, "review": [],
                    })
                    version["layouts"] += 1
                    for floor, state in available.floor_status.items():
                        if state == "PASS":
                            version["pass_floors"] += 1
                        else:
                            version["review"].append(f"{layout}/{floor}: {state}")
                variant = catalog.selected_variant(item, layout, settings["route_versions"])
                if variant:
                    for floor, state in variant.floor_status.items():
                        if state == "PASS":
                            status[floor] += 1
                        else:
                            review.append(f"{layout}/{floor}: {state}")
            summary.append({
                "map": item.map_name, "difficulty": item.difficulty_name,
                "layouts": len(layouts),
                "pure_structure_floors": sum((item.clean_root / layout / f'{layout}_{floor}.png').is_file()
                                             for layout in layouts for floor in ('2nd', '1st', 'b1')),
                "icon_structure_layouts": sum((item.clean_root / layout / f'{layout}_2nd.png').is_file() for layout in layouts),
                "pass_floors": status, "review": review,
                "versions": versions,
                "template_originals": check_original_resources(item.key, layouts),
            })
        print(json.dumps({"application_version": APP_VERSION,
                          "layout_recognition_backend": settings.get('layout_recognition_backend', 'icon_structure'),
                          "icon_templates_exist": all((RESOURCE_ROOT / 'icons' / f'{name}.png').is_file()
                                                      for name in ('query', 'clean_09', 'clean_15')),
                          "map_key": settings["map_key"], "press_to_toggle": True,
                          "game_window_titles": settings["game_window_titles"],
                          "capture_roi": settings["capture_roi"],
                          "auto_floor": settings["auto_floor"],
                          "profiles": summary},
                         ensure_ascii=False, indent=2))
        return

    if args.project_once:
        from src.app_catalog import selected_template_variant, TEMPLATE_VERSION_NAMES
        backend = args.registration_backend or settings['registration_backend']
        template_variant = selected_template_variant(settings, profile.key)
        recognition_backend = settings.get('layout_recognition_backend', 'icon_structure')
        if args.layout is None:
            parser.error('布局识别需要人工确认，请通过 --layout 指定已确认的布局后离线投影')
        if args.route_version:
            if args.route_version not in TEMPLATE_VERSION_NAMES:
                parser.error('--route-version 必须为：' + '、'.join(TEMPLATE_VERSION_NAMES))
            template_variant = args.route_version
        screenshot = _read_image(args.project_once)
        result = process_screenshot(
            screenshot, catalog=catalog, profile=profile,
            layout_override=args.layout,
            opacity=settings["opacity"],
            registration_backend=backend, template_variant_id=template_variant,
            layout_recognition_backend=recognition_backend)
        output = save_workflow_result(result, args.output)
        print(json.dumps({"accepted": result.accepted, "layout": result.selected_layout,
                          "variant": result.variant_id, "reasons": result.reasons,
                          "timings_ms": result.timings_ms, "output": str(output)},
                         ensure_ascii=False, indent=2))
        if not result.accepted:
            raise SystemExit(2)
        return

    mutex, already_running = (None, False) if args.smoke else _acquire_app_mutex()
    if already_running:
        return
    try:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
        from src.capture.region import enable_physical_pixel_coordinates
        from src.route_projection_app import RouteProjectionApp
        enable_physical_pixel_coordinates()
        qt_app = QApplication.instance() or QApplication(sys.argv)
        qt_app.setApplicationName(APP_NAME)
        qt_app.setApplicationVersion(APP_VERSION)
        app = RouteProjectionApp()
        if args.smoke:
            QTimer.singleShot(2000, app.close)
        qt_app.exec()
    finally:
        if mutex is not None:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(mutex))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log = LOG_ROOT / 'app_error.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        if "pythonw" in Path(sys.executable).name.lower():
            try:
                from PyQt6.QtWidgets import QApplication, QMessageBox
                qt_app = QApplication.instance() or QApplication(sys.argv)
                QMessageBox.critical(None, "路线投影启动失败", f"详情已写入：\n{log}")
            except Exception:
                pass
        raise
