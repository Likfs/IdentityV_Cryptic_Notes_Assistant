import sys

import pytest

from tests.support.subprocess_checks import run_probe


@pytest.mark.parametrize('mode', ['check', 'recognize', 'project', 'gui'])
def test_live_entry_runs_with_nonruntime_imports_blocked(tmp_path, mode):
    run_probe(sys.executable, mode, tmp_path/mode)
