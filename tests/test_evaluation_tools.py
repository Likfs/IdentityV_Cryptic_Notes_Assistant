"""Current-backend reports remain auditable after developer-tool cleanup."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.evaluation import audit_pose_quality as audit
from tools.evaluation import benchmark_first_registration as replay
from tools.evaluation import benchmark_first_synthetic as synthetic


@pytest.mark.parametrize("module", [replay, synthetic])
@pytest.mark.parametrize("args", [["--mode", "optimized"], ["--no-geometry"], ["--candidate-pool"], ["--config", "old.json"]])
def test_old_options_cannot_silently_change_current_benchmark(module, args, capsys):
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2
    assert "--backend cpu" in capsys.readouterr().err


def test_empty_and_missing_replay_inputs_fail_without_success_report(tmp_path):
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "out.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        replay.main(["--manifest", str(manifest), "--output", str(report)])
    assert error.value.code == 2 and not report.exists()
    manifest.write_text(json.dumps([dict(source="missing", image=str(tmp_path / "missing.png"),
                                        layout="layout_01", floor="b1", sha256="missing")]), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        replay.main(["--manifest", str(manifest), "--output", str(report)])
    assert error.value.code == 2 and not report.exists()


def test_empty_synthetic_selection_fails(tmp_path):
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        synthetic.main(["--limit", "1", "--case", "unknown/b1/narrow", "--output", str(out)])
    assert error.value.code == 2 and not out.exists()


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_replay_forwards_backend_counts_invalid_pose_and_ignores_implicit_baseline(tmp_path, monkeypatch, backend):
    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.zeros((20, 24, 3), np.uint8))
    item = dict(source="fixed", image=str(image), layout="layout_05", floor="2nd",
                sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([item]), encoding="utf-8")
    (tmp_path / "baseline.json").write_text("invalid JSON; must not be read", encoding="utf-8")
    monkeypatch.setattr(replay, "OUTPUT", tmp_path)
    monkeypatch.setattr(replay, "prepare_registration_template", lambda *_args, **_kwargs: object())
    calls = []
    def process(_image, **kwargs):
        calls.append(kwargs)
        reg = SimpleNamespace(clean_to_screen=None, timings={"total": 30},
                              validation={"registration_backend": dict(effective_backend="cpu", fallback_reason="test fallback" if backend == "auto" else None)})
        return SimpleNamespace(registration=reg, accepted=True, reasons=[], timings_ms={"registration": 30}, overlay_validation={})
    monkeypatch.setattr(replay, "process_screenshot", process)
    out = tmp_path / "out.json"
    assert replay.main(["--manifest", str(manifest), "--backend", backend, "--output", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert calls[0]["registration_backend"] == backend
    assert not any(name in calls[0] for name in ("registration_priors", "registration_evidence", "registration_soft_priors"))
    assert result["comparison_file"] is None and result["history_used"] is False
    assert result["summary"]["accepted_invalid_pose"] == 1
    assert result["effective_backends"] == {"cpu": 1}
    assert result["fallback_reasons"] == (["test fallback"] if backend == "auto" else [])
    queue = audit.real_review_queue(result)
    assert queue["rows"][0]["screenshot"] == str(image)
    assert "accepted_invalid_pose" in queue["rows"][0]["review_reasons"]
    # Explicit identical-input reference is permitted, mismatched input is not.
    reference = tmp_path / "reference.json"
    reference.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    assert replay.main(["--manifest", str(manifest), "--reference", str(reference), "--output", str(out)]) == 0
    result["rows"][0]["sha256"] = "changed"
    reference.write_text(json.dumps(result), encoding="utf-8")
    bad_output = tmp_path / "bad.json"
    with pytest.raises(SystemExit) as error:
        replay.main(["--manifest", str(manifest), "--reference", str(reference), "--output", str(bad_output)])
    assert error.value.code == 2 and not bad_output.exists()


def test_known_transform_and_invalid_pose_are_not_zero_error():
    truth = [[1, 0, 0], [0, 1, 0]]
    shifted = [[1, 0, 30], [0, 1, 0]]
    assert replay.pose_delta(shifted, truth, (80, 100, 3)) == pytest.approx(30)
    for invalid in (None, [[0, 0, 0], [0, 0, 0]], [[1, 0, float("nan")], [0, 1, 0]]):
        assert replay.pose_delta(invalid, truth, (80, 100, 3)) is None
    outcomes = [dict(accepted=True, p95_screen_error_px=None, clean_to_screen=None, registration_ms=10),
                dict(accepted=True, p95_screen_error_px=30, clean_to_screen=shifted, registration_ms=20),
                dict(accepted=False, p95_screen_error_px=None, clean_to_screen=None, registration_ms=100)]
    summary = synthetic.summarize(outcomes)
    assert summary["accepted_invalid_pose"] == summary["accepted_over_10px"] == summary["rejected"] == 1
    assert summary["accepted_correct"] == 0
    assert summary["all_timing"]["p50_ms"] == 20
    assert summary["accepted_timing"]["p50_ms"] == 15


def synthetic_report(backend=None, accepted=True, error=0):
    return {**({"backend": backend} if backend else {}), "rows": [dict(layout="layout_01", floor="b1", level="spread",
             reference_clean_to_screen=[[1, 0, 0], [0, 1, 0]],
             outcomes={backend or "optimized": dict(accepted=accepted, p95_screen_error_px=error)})]}


def test_quality_audit_supports_legacy_and_new_backend_reports():
    before = synthetic_report()
    after = synthetic_report("auto", error=None)
    result = audit.compare_synthetic(before, after)
    assert result["new_incorrect_acceptances"] == result["lost_correct_acceptances"] == 1
    assert result["before_backend"] == "optimized" and result["after_backend"] == "auto"
    cpu = synthetic_report("cpu")
    assert audit.compare_synthetic(cpu, synthetic_report("auto"))["new_incorrect_acceptances"] == 0
    for invalid in ({"rows": []}, {"rows": cpu["rows"] * 2}):
        with pytest.raises(ValueError):
            audit.compare_synthetic(cpu, invalid)
    with pytest.raises(ValueError):
        audit.real_review_queue({"rows": []})
    after["rows"][0]["reference_clean_to_screen"][0][2] = 1
    with pytest.raises(ValueError):
        audit.compare_synthetic(before, after)


def test_reference_set_cannot_be_partially_compared():
    row = dict(source="one", sha256="x", layout="layout_01", floor="b1")
    with pytest.raises(ValueError):
        replay.validate_reference([row], {"rows": [row, dict(row, source="two")]})
