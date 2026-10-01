from pathlib import Path

from src.layout_classifier import summarize_prediction
from tools.demos.layout_recognition_demo import RecognitionHotkey, layout_display_name, model_coverage_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_layout_display_name() -> None:
    assert layout_display_name("layout_05") == "布局 05"
    assert layout_display_name("custom") == "custom"


def test_model_coverage_shows_empty_and_newer_untrained_layouts(tmp_path: Path) -> None:
    roi = tmp_path / 'workspace/datasets/layout_classification/images/a_bringer_of_doom/nightmare'
    (roi / "layout_03").mkdir(parents=True)
    (roi / "layout_11").mkdir()
    (roi / 'layout_11/sample.png').touch()
    checkpoint = {
        "map_id": "a_bringer_of_doom",
        "difficulty": "nightmare",
        "labels": ["layout_01", "layout_02", "layout_04"],
    }
    text = model_coverage_text(checkpoint, tmp_path)
    assert "3 类" in text
    assert "layout_03（无样本）" in text
    assert "layout_11（未训练）" in text


def test_prediction_summary_accepts_confident_explored_layout() -> None:
    checkpoint = {
        "confidence_threshold": 0.70,
        "margin_threshold": 0.20,
        "minimum_explored_ratio": 0.10,
    }
    result = summarize_prediction(
        [("layout_03", 0.86), ("layout_02", 0.08), ("layout_01", 0.06)],
        checkpoint,
        0.15,
    )
    assert result["accepted"] is True
    assert result["result"] == "layout_03"


def test_prediction_summary_rejects_ambiguous_opening() -> None:
    checkpoint = {
        "confidence_threshold": 0.70,
        "margin_threshold": 0.20,
        "minimum_explored_ratio": 0.10,
    }
    result = summarize_prediction(
        [("layout_02", 0.90), ("layout_01", 0.04), ("layout_03", 0.03)],
        checkpoint,
        0.03,
    )
    assert result["accepted"] is False
    assert result["result"] == "unknown"


def test_demo_uses_f9_global_hotkey() -> None:
    assert RecognitionHotkey.LABEL == "F9"
    assert RecognitionHotkey.VK_F9 == 0x78


def test_windows_batch_launchers_use_crlf() -> None:
    for path in PROJECT_ROOT.glob("*.bat"):
        data = path.read_bytes()
        assert b"\r\n" in data, path.name
        assert b"\n" not in data.replace(b"\r\n", b""), path.name
