"""Build a portable Windows application from an explicit runtime asset manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.version import APP_NAME, APP_VERSION, WINDOWS_VERSION

NAME = "IdentityVRouteAssistant"


def windows_version_resource() -> str:
    """PyInstaller's Windows resource, using the same version as the application."""
    entries = {
        "FileDescription": APP_NAME,
        "FileVersion": APP_VERSION,
        "InternalName": NAME,
        "OriginalFilename": f"{NAME}.exe",
        "ProductName": APP_NAME,
        "ProductVersion": APP_VERSION,
    }
    strings = ",\n".join(f"        StringStruct({key!r}, {value!r})" for key, value in entries.items())
    return (
        "VSVersionInfo(\n"
        f"  ffi=FixedFileInfo(filevers={WINDOWS_VERSION!r}, prodvers={WINDOWS_VERSION!r},\n"
        "    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
        "  kids=[StringFileInfo([StringTable('080404B0', [\n"
        f"{strings}\n"
        "  ])]), VarFileInfo([VarStruct('Translation', [2052, 1200])])])\n"
    )


def maintenance_files() -> list[Path]:
    from src.registration.template_matching.resources import verify_module_provenance
    verify_module_provenance(ROOT / 'src/registration/template_matching')
    from tools.assets.imports.export_aligned_assets import generate
    data, _ = generate(ROOT)
    return sorted(ROOT/name for name in data['payload']['files'] if name != 'configs/scale_config.json')


def runtime_files() -> list[Path]:
    from src.registration.template_matching.resources import verify_module_provenance
    from src.runtime_resources import runtime_files as published_files
    verify_module_provenance(ROOT / 'src/registration/template_matching')
    return published_files(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建完整功能 Windows EXE 文件夹发行版")
    parser.add_argument("--output", type=Path, help="新建的发行目录；已有目录不会覆盖")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("必须在 Windows 上构建 Windows EXE")
    try:
        build_version = version("pyinstaller")
    except Exception:
        parser.error("请先安装 tools/release/requirements-build.txt")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "workspace/releases" / f"windows_{stamp}").resolve()
    output.relative_to(ROOT / "workspace")
    if output.exists():
        parser.error(f"发行目录已存在，请选择新目录：{output}")
    from tools.assets.imports.export_runtime_resources import generate
    if generate() != json.loads((ROOT / "configs/runtime_resources.json").read_text(encoding="utf-8")):
        raise ValueError("运行资源清单已过期，请先重新导出")
    files = runtime_files()
    work = ROOT / "workspace/tmp/windows_exe" / output.name
    if work.exists():
        parser.error(f"构建目录已存在，请选择不同的发行目录名称：{work}")
    stage = work / "resources"
    stage.mkdir(parents=True)
    manifest = []
    for source in files:
        relative = source.relative_to(ROOT)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({"path": relative.as_posix(), "bytes": source.stat().st_size,
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    launcher = work / "launch_app.py"
    launcher.write_text(
        'import multiprocessing\nimport runpy\n\n'
        'if __name__ == "__main__":\n'
        '    multiprocessing.freeze_support()\n'
        '    runpy.run_module("src", run_name="__main__")\n', encoding="utf-8")
    output.mkdir(parents=True)
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    version_file = work / "windows_version.txt"
    version_file.write_text(windows_version_resource(), encoding="utf-8")
    command = [sys.executable, "-m", "PyInstaller", "--onedir", "--console",
               "--hide-console", "hide-early", "--contents-directory", ".", "--noupx",
               "--name", NAME, "--distpath", str(output), "--workpath", str(work / "analysis"),
               "--specpath", str(work), "--paths", str(ROOT), "--collect-submodules", "src",
               "--version-file", str(version_file)]
    for package in ("tkinter", "pytest", "tools", "tests", "IPython", "matplotlib",
                    "pandas", "scipy", "torch", "retired_features", "torchvision", "torchaudio", "tensorboard"):
        command.extend(["--exclude-module", package])
    for directory in ("assets", "configs", "src"):
        command.extend(["--add-data", f"{stage / directory}:{directory}"])
    command.append(str(launcher))
    # PyInstaller also searches PATH for DLLs. Host tools (e.g. Poppler) can
    # provide an incompatible icuuc.dll with the same name as Windows ICU.
    build_env = dict(os.environ)
    windows_root = Path(os.environ["SystemRoot"])
    build_env["PATH"] = os.pathsep.join(map(str, (
        Path(sys.executable).parent, Path(sys.base_prefix),
        windows_root / "System32", windows_root,
    )))
    build_env["PYTHONUTF8"] = "1"
    for key in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        build_env.pop(key, None)
    print(f"发布资源：{len(files)} 个文件；输出：{output}", flush=True)
    with (output / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=build_env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    app = output / NAME
    for document in ("README.md", "项目目录与文件说明.md", "新增布局教程.md"):
        shutil.copy2(ROOT / document, app / document)
    (output / "build_info.json").write_text(json.dumps({
        "built_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "application_version": APP_VERSION, "windows_version": WINDOWS_VERSION,
        "python": sys.version, "pyinstaller": build_version, "recognition": "pure_structure", "recognition_options": ["pure_structure", "icon_structure"], "registration": "template",
        "format": "onedir", "resource_files": len(manifest),
        "executable": f"{NAME}/{NAME}.exe", "command": command,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"构建完成：{app / (NAME + '.exe')}", flush=True)


if __name__ == "__main__":
    main()
