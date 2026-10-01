from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layout_classifier import (
    DEFAULT_IMAGE_SIZE,
    LayoutCNN,
    LayoutDataset,
    LayoutSample,
    compute_channel_stats,
    read_layout_samples,
    split_chronologically,
    _resize,
)


def output_paths(preprocess: str, project_root: Path = PROJECT_ROOT) -> tuple[Path, Path, Path]:
    if preprocess not in {"original", "pre_resized"}:
        raise ValueError(f"Unknown preprocessing mode: {preprocess}")
    suffix = "" if preprocess == "original" else "_pre_resized"
    return (
        project_root / 'workspace/checkpoints' / f"layout_classifier{suffix}.pt",
        project_root / 'workspace/checkpoints' / f"layout_classifier{suffix}_validation.pt",
        project_root / 'workspace/reports/classification' / f"layout_classifier{suffix}_training.json",
    )


def prepare_pre_resized_samples(
    samples: list[LayoutSample], cache_dir: Path, image_size: tuple[int, int]
) -> tuple[list[LayoutSample], int]:
    """Reuse lossless thumbnails only while the source path, size and mtime match."""
    cached: list[LayoutSample] = []
    created = 0
    for sample in samples:
        source = sample.path.resolve()
        stat = source.stat()
        identity = f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{image_size}"
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        target = cache_dir / sample.label / f"{source.stem}_{fingerprint}.png"
        valid = False
        if target.is_file():
            try:
                with Image.open(target) as image:
                    valid = image.size == image_size and image.mode == "RGB"
                    image.verify()
            except (OSError, ValueError):
                valid = False
        if not valid:
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                _resize(image, image_size).save(target, format="PNG")
            created += 1
        cached.append(LayoutSample(sample.label, target))
    return cached, created


def evaluate(model, loader, labels, device) -> dict:
    model.eval()
    confusion = [[0 for _ in labels] for _ in labels]
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            logits = model(inputs)
            losses.append(float(criterion(logits, targets).cpu()))
            predicted = logits.argmax(dim=1)
            for truth, pred in zip(targets.cpu().tolist(), predicted.cpu().tolist()):
                confusion[truth][pred] += 1
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(len(labels)))
    per_label = {
        label: confusion[i][i] / max(sum(confusion[i]), 1) for i, label in enumerate(labels)
    }
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": correct / max(total, 1),
        "per_label_accuracy": per_label,
        "confusion_matrix": confusion,
    }


def train_epochs(model, loader, device, epochs, learning_rate, class_weights=None, val_loader=None, labels=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    best_state = None
    best_metrics = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for inputs, targets in loader:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else 0.0
        if val_loader is not None:
            metrics = evaluate(model, val_loader, labels, device)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            print(
                f"epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.3f}",
                flush=True,
            )
            score = (metrics["accuracy"], -metrics["loss"])
            previous = (-1.0, float("-inf")) if best_metrics is None else (
                best_metrics["accuracy"], -best_metrics["loss"]
            )
            if score > previous:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metrics = metrics
                best_epoch = epoch
        else:
            print(f"final epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f}", flush=True)
    return best_state, best_metrics, best_epoch, history


def class_weight_tensor(samples: list[LayoutSample], labels: list[str], device):
    counts = Counter(sample.label for sample in samples)
    maximum = max(counts.values())
    weights = [float(np.sqrt(maximum / counts[label])) for label in labels]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(path: Path, model, labels, image_size, mean, std, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "layout_cnn_v1",
            "model_state": model.state_dict(),
            "labels": labels,
            "image_size": list(image_size),
            "mean": mean,
            "std": std,
            "confidence_threshold": 0.75,
            "margin_threshold": 0.20,
            "minimum_explored_ratio": 0.10,
            **extra,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the random-layout classifier.")
    parser.add_argument("--map-id", default="a_bringer_of_doom")
    parser.add_argument("--difficulty", default="nightmare")
    parser.add_argument(
        "--preprocess",
        choices=("original", "pre_resized"),
        default="original",
        help="Enhance full-size originals or cached 160x144 thumbnails; each mode saves a separate model",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--warmup-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel image-loading workers; defaults to 4 on CUDA and 0 on CPU",
    )
    parser.add_argument(
        "--max-layout",
        type=int,
        default=None,
        help="Only train layout_01 through layout_NN; omit to train every archived layout",
    )
    parser.add_argument(
        "--allow-missing-layouts",
        action="store_true",
        help="Skip empty layout numbers within --max-layout without renumbering the remaining labels",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    all_samples = read_layout_samples(PROJECT_ROOT, args.map_id, args.difficulty)
    missing_labels: list[str] = []
    if args.max_layout is not None:
        if args.max_layout < 2:
            raise ValueError("--max-layout must be at least 2")
        allowed_labels = {f"layout_{number:02d}" for number in range(1, args.max_layout + 1)}
        all_samples = [sample for sample in all_samples if sample.label in allowed_labels]
    labels = sorted({sample.label for sample in all_samples})
    if args.max_layout is not None:
        missing_labels = sorted(allowed_labels - set(labels))
        if missing_labels and not args.allow_missing_layouts:
            raise RuntimeError(
                f"Requested layouts have no samples: {missing_labels}; "
                "pass --allow-missing-layouts to train the populated layouts"
            )
    if len(labels) < 2:
        raise RuntimeError(f"At least two layouts are required; found {labels}")
    train_samples, val_samples, skipped_samples = split_chronologically(
        all_samples, val_ratio=args.val_ratio, warmup_ratio=args.warmup_ratio
    )
    useful_samples = sorted(train_samples + val_samples, key=lambda sample: (sample.label, sample.path.name))
    label_to_index = {label: index for index, label in enumerate(labels)}
    image_size = DEFAULT_IMAGE_SIZE
    cache_created = 0
    cache_seconds = 0.0
    if args.preprocess == "pre_resized":
        cache_dir = PROJECT_ROOT / 'workspace/cache/pre_resized' / args.map_id / args.difficulty
        cache_started = time.perf_counter()
        cached_samples, cache_created = prepare_pre_resized_samples(useful_samples, cache_dir, image_size)
        cache_seconds = time.perf_counter() - cache_started
        cached_by_source = {
            original.path: cached
            for original, cached in zip(useful_samples, cached_samples)
        }
        train_samples = [cached_by_source[sample.path] for sample in train_samples]
        val_samples = [cached_by_source[sample.path] for sample in val_samples]
        useful_samples = cached_samples
    mean, std = compute_channel_stats(train_samples, image_size)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but the installed PyTorch cannot access an NVIDIA GPU")
        torch.backends.cudnn.benchmark = True

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    workers = args.workers if args.workers is not None else (4 if device.type == "cuda" else 0)
    if workers < 0:
        raise ValueError("--workers cannot be negative")
    loader_options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    print(
        f"device={device} ({device_name}); workers={workers}; "
        f"preprocess={args.preprocess}; labels={labels}; total={len(all_samples)}",
        flush=True,
    )
    if args.preprocess == "pre_resized":
        print(
            f"resized cache: created={cache_created}; total={len(useful_samples)}; "
            f"prepare_seconds={cache_seconds:.1f}; path={cache_dir}",
            flush=True,
        )
    if missing_labels:
        print(f"empty layout numbers excluded: {missing_labels}", flush=True)
    print(
        f"train={len(train_samples)}; val={len(val_samples)}; "
        f"ambiguous/purged={len(skipped_samples)}; mean={mean}; std={std}",
        flush=True,
    )
    train_ds = LayoutDataset(train_samples, label_to_index, image_size, mean, std, augment=True, seed=args.seed)
    val_ds = LayoutDataset(val_samples, label_to_index, image_size, mean, std, augment=False, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = LayoutCNN(len(labels)).to(device)
    best_state, best_metrics, best_epoch, history = train_epochs(
        model,
        train_loader,
        device,
        args.epochs,
        args.lr,
        class_weight_tensor(train_samples, labels, device),
        val_loader,
        labels,
    )
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    final_path, validation_path, report_path = output_paths(args.preprocess, PROJECT_ROOT)
    save_checkpoint(
        validation_path,
        model,
        labels,
        image_size,
        mean,
        std,
        {
            "map_id": args.map_id,
            "difficulty": args.difficulty,
            "preprocess": args.preprocess,
            "training_kind": "chronological_holdout",
            "best_epoch": best_epoch,
            "validation": best_metrics,
        },
    )

    full_ds = LayoutDataset(useful_samples, label_to_index, image_size, mean, std, augment=True, seed=args.seed + 2)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, **loader_options)
    fine_tune_epochs = max(3, min(6, (best_epoch + 2) // 3))
    train_epochs(
        model,
        full_loader,
        device,
        fine_tune_epochs,
        args.lr * 0.20,
        class_weight_tensor(useful_samples, labels, device),
    )
    save_checkpoint(
        final_path,
        model,
        labels,
        image_size,
        mean,
        std,
        {
            "map_id": args.map_id,
            "difficulty": args.difficulty,
            "preprocess": args.preprocess,
            "training_kind": "all_useful_samples",
            "sample_count": len(useful_samples),
            "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "validation_checkpoint": validation_path.name,
            "validation": best_metrics,
        },
    )

    report = {
        "map_id": args.map_id,
        "difficulty": args.difficulty,
        "preprocess": args.preprocess,
        "cache_created": cache_created,
        "cache_prepare_seconds": cache_seconds,
        "labels": labels,
        "max_layout": args.max_layout,
        "excluded_empty_layouts": missing_labels,
        "all_sample_count": len(all_samples),
        "training_sample_count": len(train_samples),
        "validation_sample_count": len(val_samples),
        "ambiguous_or_purged_count": len(skipped_samples),
        "useful_final_sample_count": len(useful_samples),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "final_checkpoint": str(final_path.relative_to(PROJECT_ROOT).as_posix()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved final model: {final_path}", flush=True)
    print(f"saved report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
