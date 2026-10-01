from pathlib import Path

from src.layout_classifier import LayoutSample, split_chronologically


def test_chronological_split_keeps_late_frames_for_validation() -> None:
    samples = [LayoutSample("layout_01", Path(f"frame_{index:03d}.png")) for index in range(100)]
    train, val, skipped = split_chronologically(
        samples, val_ratio=0.2, warmup_ratio=0.1, purge_frames=3
    )

    assert [sample.path.name for sample in val] == [f"frame_{index:03d}.png" for index in range(82, 100)]
    assert max(sample.path.name for sample in train) < min(sample.path.name for sample in val)
    assert len(train) + len(val) + len(skipped) == len(samples)
