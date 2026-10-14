"""Standard-library bootstrap for source checkouts and linked Git worktrees."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEPENDENCY_PROBE = (
    'import cv2, numpy, PIL, mss; from PyQt6 import QtWidgets; '
    'import sys; '
    'exec("import comtypes" if sys.platform == "win32" else "pass")'
)


def python_candidates(root):
    """Resolve worktree metadata directly, without requiring Git on PATH."""
    candidates = [root / '.venv/Scripts/python.exe']
    marker = root / '.git'
    if marker.is_file():
        text = marker.read_text(encoding='utf-8-sig').strip()
        if text.startswith('gitdir:'):
            git_dir = (root / text.split(':', 1)[1].strip()).resolve()
            common_file = git_dir / 'commondir'
            if common_file.is_file():
                common = (git_dir / common_file.read_text(encoding='utf-8').strip()).resolve()
                candidates.append(common.parent / '.venv/Scripts/python.exe')
    candidates.append(Path(sys.executable))
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def probe_python(executable, root):
    try:
        result = subprocess.run([str(executable), '-c', DEPENDENCY_PROBE], cwd=root,
            capture_output=True, encoding='utf-8', errors='replace', timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return None if result.returncode == 0 else result.stderr.strip() or f'exit={result.returncode}'
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def select_python(root):
    errors = []
    for executable in python_candidates(root):
        error = probe_python(executable, root)
        if error is None:
            return executable
        errors.append(f'{executable}\n{error}')
    raise RuntimeError('No Python environment with all application dependencies was found.\n'
                       + '\n\n'.join(errors)
                       + '\nSee README.md: install requirements.txt into this checkout\'s .venv.')


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)
    log = ROOT / 'runtime/logs/launcher.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        executable = select_python(ROOT)
        details = dict(workspace=str(ROOT), python=str(executable))
        log.write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if args == ['--launcher-check']:
            print(json.dumps(details, ensure_ascii=False))
            return 0
        if args:
            return subprocess.run([str(executable), '-m', 'src', *args], cwd=ROOT).returncode
        # The working directory stays on this checkout even when the interpreter is shared.
        pythonw = executable.with_name('pythonw.exe')
        child = subprocess.Popen([str(pythonw if pythonw.is_file() else executable), '-m', 'src'],
            cwd=ROOT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        with log.open('a', encoding='utf-8') as stream:
            stream.write(f'pid={child.pid}\n')
        return 0
    except (OSError, RuntimeError) as exc:
        message = f'{datetime.now().isoformat(timespec="seconds")}\n{exc}\n'
        log.write_text(message, encoding='utf-8')
        print(message, file=sys.stderr)
        print(f'Startup log: {log}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
