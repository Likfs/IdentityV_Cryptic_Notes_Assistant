from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset


DEFAULT_IMAGE_SIZE = (160, 144)


@dataclass(frozen=True)
class LayoutSample:
    label: str
    path: Path


def read_layout_samples(project_root: Path, map_id: str, difficulty: str) -> list[LayoutSample]:
    manifest_dir = project_root / 'workspace/manifests' / map_id / difficulty
    samples: list[LayoutSample] = []
    for manifest in sorted(manifest_dir.glob("layout_*.jsonl")):
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("label", "")).strip()
            image_text = str(row.get("image", "")).strip()
            if not label or not image_text:
                raise ValueError(f"Invalid record: {manifest}:{line_number}")
            image_path = Path(image_text)
            if not image_path.is_absolute():
                image_path = project_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Image in JSONL does not exist: {image_path}")
            samples.append(LayoutSample(label=label, path=image_path))
    return samples


def split_chronologically(
    samples: Iterable[LayoutSample],
    val_ratio: float = 0.2,
    warmup_ratio: float = 0.12,
    purge_frames: int = 3,
) -> tuple[list[LayoutSample], list[LayoutSample], list[LayoutSample]]:
    """Split by time and discard ambiguous opening/adjacent video frames."""
    grouped: dict[str, list[LayoutSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    train: list[LayoutSample] = []
    val: list[LayoutSample] = []
    skipped: list[LayoutSample] = []
    for label, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.path.name)
        warmup_n = int(round(len(ordered) * warmup_ratio))
        useful = ordered[warmup_n:]
        skipped.extend(ordered[:warmup_n])
        if len(useful) < 10:
            raise ValueError(f"Not enough useful samples for {label}: {len(useful)}")
        val_n = max(1, int(round(len(useful) * val_ratio)))
        split_at = len(useful) - val_n
        train_end = max(1, split_at - max(0, purge_frames))
        train.extend(useful[:train_end])
        skipped.extend(useful[train_end:split_at])
        val.extend(useful[split_at:])
    return train, val, skipped


class LayoutCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
            block(64, 96),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 4 * 4, 160),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _resize(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
    if rng.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.8)))
    return image


def image_to_tensor(
    image: Image.Image,
    image_size: tuple[int, int],
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    resized = _resize(image, image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1)).copy()
    mean_array = np.asarray(list(mean), dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(list(std), dtype=np.float32).reshape(3, 1, 1)
    array = (array - mean_array) / np.maximum(std_array, 1e-6)
    return torch.from_numpy(array)


class LayoutDataset(Dataset):
    def __init__(
        self,
        samples: list[LayoutSample],
        label_to_index: dict[str, int],
        image_size: tuple[int, int],
        mean: list[float],
        std: list[float],
        augment: bool = False,
        seed: int = 20260918,
    ) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        if self.augment:
            image = _augment(image, self.rng)
        return image_to_tensor(image, self.image_size, self.mean, self.std), self.label_to_index[sample.label]


def compute_channel_stats(
    samples: Iterable[LayoutSample], image_size: tuple[int, int]
) -> tuple[list[float], list[float]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for sample in samples:
        with Image.open(sample.path) as source:
            image = np.asarray(_resize(source, image_size), dtype=np.float32) / 255.0
        pixels = image.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_square_sum += (pixels * pixels).sum(axis=0)
        pixel_count += pixels.shape[0]
    mean = channel_sum / max(pixel_count, 1)
    variance = channel_square_sum / max(pixel_count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(float).tolist(), std.astype(float).tolist()


def load_classifier(checkpoint_path: Path, device: str | None = None):
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    labels = [str(label) for label in checkpoint["labels"]]
    model = LayoutCNN(len(labels)).to(selected_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, selected_device


def predict_image(
    model: nn.Module,
    checkpoint: dict,
    device: torch.device,
    image: Image.Image,
) -> list[tuple[str, float]]:
    tensor = image_to_tensor(
        image,
        tuple(checkpoint["image_size"]),
        checkpoint["mean"],
        checkpoint["std"],
    ).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.to(device)), dim=1).squeeze(0).cpu().tolist()
    pairs = zip(checkpoint["labels"], probabilities)
    return sorted(((str(label), float(score)) for label, score in pairs), key=lambda item: item[1], reverse=True)


def explored_ratio(image: Image.Image, brightness_threshold: int = 65) -> float:
    gray = np.asarray(image.convert("L"))
    return float((gray > brightness_threshold).mean())


def summarize_prediction(
    ranking: list[tuple[str, float]],
    checkpoint: dict,
    current_explored_ratio: float,
) -> dict:
    if not ranking:
        raise ValueError("Prediction ranking must not be empty")
    best_label, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0.0
    confidence_threshold = float(checkpoint.get("confidence_threshold", 0.70))
    margin_threshold = float(checkpoint.get("margin_threshold", 0.20))
    minimum_explored_ratio = float(checkpoint.get("minimum_explored_ratio", 0.10))
    margin = best_score - second_score
    accepted = (
        current_explored_ratio >= minimum_explored_ratio
        and best_score >= confidence_threshold
        and margin >= margin_threshold
    )
    return {
        "result": best_label if accepted else "unknown",
        "accepted": accepted,
        "confidence": best_score,
        "margin": margin,
        "explored_ratio": current_explored_ratio,
        "minimum_explored_ratio": minimum_explored_ratio,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "ranking": [{"label": label, "probability": score} for label, score in ranking],
    }
