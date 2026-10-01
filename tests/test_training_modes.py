import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.training import train_layout_classifier as trainer
from tools.training.train_layout_classifier import output_paths, prepare_pre_resized_samples
from src.layout_classifier import LayoutSample, _resize
from tools.demos.layout_recognition_demo import available_model_options


def test_training_modes_write_different_model_and_report_paths(tmp_path: Path) -> None:
    original = output_paths("original", tmp_path)
    pre_resized = output_paths("pre_resized", tmp_path)
    assert original[0].name == "layout_classifier.pt"
    assert pre_resized[0].name == "layout_classifier_pre_resized.pt"
    assert original[1] != pre_resized[1]
    assert original[2] != pre_resized[2]


def test_pre_resized_cache_matches_source_and_invalidates_changes(tmp_path: Path) -> None:
    source = tmp_path / 'layout_01/sample.png'
    source.parent.mkdir()
    Image.new("RGB", (40, 30), (20, 80, 140)).save(source)
    samples = [LayoutSample("layout_01", source)]
    cache_dir = tmp_path / "cache"

    first, created = prepare_pre_resized_samples(samples, cache_dir, (16, 14))
    assert created == 1
    with Image.open(source) as original, Image.open(first[0].path) as cached:
        assert np.array_equal(np.asarray(_resize(original, (16, 14))), np.asarray(cached))

    second, created = prepare_pre_resized_samples(samples, cache_dir, (16, 14))
    assert created == 0
    assert second[0].path == first[0].path

    previous = source.stat()
    Image.new("RGB", (40, 30), (180, 40, 10)).save(source)
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    third, created = prepare_pre_resized_samples(samples, cache_dir, (16, 14))
    assert created == 1
    assert third[0].path != first[0].path


def test_demo_lists_each_mode_and_custom_model(tmp_path: Path) -> None:
    original = tmp_path / "layout_classifier.pt"
    pre_resized = tmp_path / "layout_classifier_pre_resized.pt"
    custom = tmp_path / "experiment.pt"
    for path in (original, pre_resized, custom):
        path.touch()

    options = available_model_options(custom, original, pre_resized)
    assert options["原图增强（正式模型）"] == original.resolve()
    assert options["预缩放后增强"] == pre_resized.resolve()
    assert options["自定义：experiment.pt"] == custom.resolve()


def test_demo_uses_labeled_experiment_until_pre_resized_model_exists(tmp_path: Path) -> None:
    original = tmp_path / "layout_classifier.pt"
    pre_resized = tmp_path / "layout_classifier_pre_resized.pt"
    experiment_root = tmp_path / 'workspace/reports'
    experiment = experiment_root / 'resize_cache_ab_01/pre_resized_best.pt'
    experiment.parent.mkdir(parents=True)
    original.touch()
    experiment.touch()

    options = available_model_options(original, original, pre_resized, experiment_root)
    assert options["预缩放实验模型（仅验证）"] == experiment.resolve()
    pre_resized.touch()
    options = available_model_options(original, original, pre_resized, experiment_root)
    assert options["预缩放后增强"] == pre_resized.resolve()
    assert "预缩放实验模型（仅验证）" not in options


def test_each_training_mode_saves_a_separate_checkpoint(tmp_path: Path, monkeypatch) -> None:
    roi = tmp_path / 'workspace/datasets/layout_classification/images/a_bringer_of_doom/nightmare'
    manifests = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare'
    manifests.mkdir(parents=True)
    for number, color in ((1, (25, 65, 100)), (2, (150, 180, 200))):
        label = f"layout_{number:02d}"
        folder = roi / label
        folder.mkdir(parents=True)
        lines = []
        for index in range(16):
            image = folder / f"frame_{index:03d}.png"
            Image.new("RGB", (32, 28), color).save(image)
            lines.append(json.dumps({"image": str(image), "label": label}))
        (manifests / f"{label}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    for mode in ("original", "pre_resized"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_layout_classifier.py", "--preprocess", mode, "--device", "cpu",
                "--epochs", "1", "--batch-size", "8", "--workers", "0", "--max-layout", "2",
            ],
        )
        assert trainer.main() == 0
        final, validation, report = trainer.output_paths(mode, tmp_path)
        assert final.is_file() and validation.is_file() and report.is_file()
        checkpoint = torch.load(final, map_location="cpu", weights_only=False)
        assert checkpoint["preprocess"] == mode
        assert checkpoint["labels"] == ["layout_01", "layout_02"]

    assert trainer.output_paths("original", tmp_path)[0] != trainer.output_paths("pre_resized", tmp_path)[0]
