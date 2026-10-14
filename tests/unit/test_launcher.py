from pathlib import Path
import sys

import pytest

import launch


def linked_checkout(tmp_path):
    root = tmp_path/'worktree'
    primary = tmp_path/'primary'
    metadata = primary/'.git/worktrees/example'
    metadata.mkdir(parents=True)
    root.mkdir()
    (root/'.git').write_text(f'gitdir: {metadata.as_posix()}\n', encoding='utf-8')
    (metadata/'commondir').write_text('../..\n', encoding='utf-8')
    interpreter = primary/'.venv/Scripts/python.exe'
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    return root, interpreter


def test_worktree_without_local_venv_reuses_primary_environment(tmp_path, monkeypatch):
    root, shared = linked_checkout(tmp_path)
    calls = []
    def probe(executable, cwd):
        calls.append((executable, cwd))
        return None if executable == shared else 'missing cv2'
    monkeypatch.setattr(launch, 'probe_python', probe)
    assert launch.select_python(root) == shared
    assert calls == [(shared, root)]  # Resolve modules from this checkout, not the primary one.


def test_local_environment_takes_priority_and_bad_candidates_are_checked(tmp_path, monkeypatch):
    root, shared = linked_checkout(tmp_path)
    local = root/'.venv/Scripts/python.exe'
    local.parent.mkdir(parents=True)
    local.touch()
    monkeypatch.setattr(launch, 'probe_python', lambda executable, root: None)
    assert launch.select_python(root) == local
    monkeypatch.setattr(launch, 'probe_python', lambda executable, root: 'missing cv2' if executable == local else None)
    assert launch.select_python(root) == shared


def test_missing_dependencies_fail_with_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'probe_python', lambda *a: 'ModuleNotFoundError: cv2')
    with pytest.raises(RuntimeError, match='requirements.txt'):
        launch.select_python(tmp_path)


def test_actual_runtime_interpreter_passes_probe():
    assert launch.probe_python(Path(sys.executable), launch.ROOT) is None
