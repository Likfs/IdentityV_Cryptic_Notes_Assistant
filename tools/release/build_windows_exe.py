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


def runtime_files() -> list[Path]:
    from src.registration.template_matching.resources import verify_module_provenance
    verify_module_provenance(ROOT / 'src/registration/template_matching')
    files: set[Path] = set()

    def include(value: str | Path) -> None:
        path = (ROOT / value).resolve()
        path.relative_to(ROOT)  # Do not accidentally ship files outside this project.
        if not path.is_file():
            raise FileNotFoundError(f"缺少运行资源：{path}")
        files.add(path)

    for name in ("map_catalog.json", "app_defaults.json", "template_matching.json"):
        include(Path("configs") / name)
    catalog = json.loads((ROOT / "configs/map_catalog.json").read_text(encoding="utf-8-sig"))
    for map_item in catalog["maps"]:
        for profile in map_item["difficulties"]:
            clean_root = ROOT / profile["clean_root"]
            layouts = sorted(p for p in clean_root.glob("layout_*") if p.is_dir())
            if not layouts:
                raise ValueError(f"没有可发布布局：{clean_root}")
            for layout in layouts:
                for floor in ("2nd", "1st", "b1"):
                    images = list(layout.glob(f"*_{floor}.png"))
                    if len(images) != 1:
                        raise ValueError(f"底图数量错误：{layout.name}/{floor}")
                    include(images[0])
                if profile.get("entrance_root"):
                    include(Path(profile["entrance_root"]) / f"{layout.name}.png")
                calibrations = list((ROOT / profile["calibration_root"]).glob(
                    f"route_*/{layout.name}/calibration.json"))
                if not calibrations:
                    raise ValueError(f"布局缺少路线标定：{layout.name}")
                for calibration in calibrations:
                    include(calibration)
    config = json.loads((ROOT / "configs/template_matching.json").read_text(encoding="utf-8"))
    for entry in config["entries"].values():
        include(entry["source_image"])
        for field, fingerprint in (("source_image", "source_sha256"),
                                   ("calibration", "calibration_sha256")):
            path = (ROOT / entry[field]).resolve()
            include(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry[fingerprint]:
                raise ValueError(f"定尺度模板匹配指纹不一致：{path}")
    from src.registration.template_matching.direct import _source
    for profile_key, layouts in config.get('original_variants', {}).items():
        for layout, variants in layouts.items():
            for variant_id in variants:
                entry, source, _ = _source(profile_key, layout, variant_id)
                if 'source_parts' in entry:
                    for part in entry['source_parts']:
                        include(part['path'])
                else:
                    include(source)
                for candidate in entry['candidates']:
                    for dependency in candidate.get('dependencies', []):
                        include(dependency['path'])
    include("assets/app_ui/home_wallpaper.jpg")
    include("assets/app_ui/icons/check.svg")
    for name in ('query', 'clean_09', 'clean_15'):
        include(f'assets/layout_retrieval/icons/{name}.png')
    include("src/registration/template_matching/provenance.json")
    return sorted(files)


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
