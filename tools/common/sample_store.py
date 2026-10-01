"""Dataset storage and JSONL maintenance, independent of capture and UI."""
from __future__ import annotations
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from PIL import Image
from src.capture.region import CaptureRegion
from tools.common.paths import PROJECT_ROOT, TOOL_CONFIG_ROOT

DEFAULT_LABELS_PATH = TOOL_CONFIG_ROOT / "labels.json"
DEFAULT_ROI_ROOT = PROJECT_ROOT / "workspace/datasets/layout_classification/images"
PENDING_LAYOUT_ID = "pending"
PENDING_LAYOUT_NAME = "待确认布局"

@dataclass(frozen=True)
class Category:
    id: str
    name: str

    @classmethod
    def from_value(cls, value: object, category_type: str) -> "Category":
        if not isinstance(value, dict):
            raise ValueError(f"{category_type}分类项必须是对象")
        category_id = str(value.get("id", "")).strip()
        name = str(value.get("name", "")).strip()
        if not category_id or not name:
            raise ValueError(f"{category_type}分类项必须同时包含非空 id 和 name")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", category_id):
            raise ValueError(f"{category_type}分类 id 只能包含小写字母、数字、下划线和连字符：{category_id}")
        return cls(category_id, name)

    @property
    def display(self) -> str:
        return f"{self.name} ({self.id})"


@dataclass(frozen=True)
class LayoutCategory(Category):
    map_id: str
    difficulty_id: str

    @classmethod
    def from_value(cls, value: object) -> "LayoutCategory":
        base = Category.from_value(value, "布局")
        assert isinstance(value, dict)
        map_id = str(value.get("map_id", "")).strip()
        difficulty_id = str(value.get("difficulty_id", "")).strip()
        if not map_id or not difficulty_id:
            raise ValueError(f"布局 {base.id} 必须指定 map_id 和 difficulty_id")
        return cls(base.id, base.name, map_id, difficulty_id)


def load_category_config(path: Path = DEFAULT_LABELS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("分类配置的根节点必须是对象")

    maps = [Category.from_value(item, "地图") for item in raw.get("maps", [])]
    difficulties = [Category.from_value(item, "难度") for item in raw.get("difficulties", [])]
    if not maps:
        raise ValueError("分类配置至少需要一张地图")
    if not difficulties:
        raise ValueError("分类配置至少需要一个难度")

    def validate_unique(categories: list[Category], category_type: str) -> None:
        ids = [item.id for item in categories]
        names = [item.name for item in categories]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{category_type}分类 id 不能重复")
        if len(names) != len(set(names)):
            raise ValueError(f"{category_type}分类名称不能重复")

    validate_unique(maps, "地图")
    validate_unique(difficulties, "难度")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults 必须是对象")
    default_map_id = str(defaults.get("map_id", maps[0].id)).strip()
    default_difficulty_id = str(defaults.get("difficulty_id", difficulties[0].id)).strip()
    if default_map_id not in {item.id for item in maps}:
        raise ValueError(f"默认地图不存在：{default_map_id}")
    if default_difficulty_id not in {item.id for item in difficulties}:
        raise ValueError(f"默认难度不存在：{default_difficulty_id}")
    return {
        "schema_version": int(raw.get("schema_version", 1)),
        "maps": maps,
        "difficulties": difficulties,
        "default_map_id": default_map_id,
        "default_difficulty_id": default_difficulty_id,
    }


def safe_label(value: str) -> str:
    """Return a filesystem-safe label while preserving Chinese characters."""
    label = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value.strip())
    label = label.strip(" .")
    return label[:80] or "unlabeled"


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


class SampleWriter:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        roi_root: Optional[Path] = None,
        manifest_root: Optional[Path] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.roi_root = (roi_root or self.project_root / 'workspace/datasets/layout_classification/images').resolve()
        self.manifest_root = (manifest_root or self.project_root / 'workspace/manifests').resolve()

    def known_layout_ids(self, map_id: str, difficulty_id: str) -> list[str]:
        pattern = re.compile(r"^layout_(\d+)$")
        found: set[str] = set()
        context_dir = self.roi_root / map_id / difficulty_id
        if context_dir.exists():
            for item in context_dir.iterdir():
                if item.is_dir() and pattern.fullmatch(item.name):
                    found.add(item.name)
        return sorted(found, key=lambda value: int(value.rsplit("_", 1)[1]))

    def layout_sample_count(self, map_id: str, difficulty_id: str, layout_id: str) -> int:
        layout_dir = self.roi_root / map_id / difficulty_id / layout_id
        return sum(1 for _path in layout_dir.glob("*.png")) if layout_dir.exists() else 0

    def next_layout(self, map_category: Category, difficulty_category: Category) -> LayoutCategory:
        known = self.known_layout_ids(map_category.id, difficulty_category.id)
        next_number = max((int(value.rsplit("_", 1)[1]) for value in known), default=0) + 1
        layout_id = f"layout_{next_number:02d}"
        return LayoutCategory(
            layout_id,
            f"布局 {next_number:02d}",
            map_category.id,
            difficulty_category.id,
        )

    def pending_sample_ids(self, map_id: str, difficulty_id: str) -> list[str]:
        pending_dir = self.roi_root / map_id / difficulty_id / PENDING_LAYOUT_ID
        if not pending_dir.exists():
            return []
        return [path.stem for path in sorted(pending_dir.glob("*.png"))]

    def save(
        self,
        image: Image.Image,
        *,
        region: CaptureRegion,
        map_category: Category,
        difficulty_category: Category,
        layout_category: LayoutCategory,
    ) -> dict[str, Any]:
        now = datetime.now().astimezone()
        sample_id = f"{now:%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
        if layout_category.map_id != map_category.id or layout_category.difficulty_id != difficulty_category.id:
            raise ValueError("所选布局不属于当前地图和难度")
        output_dir = (
            self.roi_root
            / safe_label(map_category.id)
            / safe_label(difficulty_category.id)
            / safe_label(layout_category.id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{sample_id}.png"
        temp_path = output_dir / f".{uuid.uuid4().hex[:16]}.tmp"
        image.save(temp_path, format="PNG")
        os.replace(temp_path, image_path)

        try:
            relative_image = image_path.relative_to(self.project_root).as_posix()
        except ValueError:
            relative_image = image_path.as_posix()
        return {
            "sample_id": sample_id,
            "layout_id": layout_category.id,
            "roi_image_path": relative_image,
            "label_status": "pending" if layout_category.id == PENDING_LAYOUT_ID else "confirmed",
        }

    def _layout_manifest_path(self, map_id: str, difficulty_id: str, layout_id: str) -> Path:
        return self.manifest_root / map_id / difficulty_id / f"{layout_id}.jsonl"

    def _write_layout_manifest(
        self,
        map_category: Category,
        difficulty_category: Category,
        layout_category: LayoutCategory,
    ) -> Path:
        layout_dir = self.roi_root / map_category.id / difficulty_category.id / layout_category.id
        images = sorted(layout_dir.glob("*.png")) if layout_dir.exists() else []
        manifest_path = self._layout_manifest_path(
            map_category.id, difficulty_category.id, layout_category.id
        )
        if not images:
            manifest_path.unlink(missing_ok=True)
            return manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = manifest_path.with_name(f".{uuid.uuid4().hex[:16]}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            for image_path in images:
                record = {
                    "image": image_path.relative_to(self.project_root).as_posix(),
                    "label": layout_category.id,
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, manifest_path)
        return manifest_path

    def assign_pending_layout(
        self,
        sample_ids: list[str],
        *,
        map_category: Category,
        difficulty_category: Category,
        layout_category: LayoutCategory,
    ) -> list[dict[str, Any]]:
        if not sample_ids:
            return []
        if layout_category.id == PENDING_LAYOUT_ID:
            raise ValueError("必须选择一个已确认布局")
        if layout_category.map_id != map_category.id or layout_category.difficulty_id != difficulty_category.id:
            raise ValueError("所选布局不属于当前地图和难度")
        pending_dir = self.roi_root / map_category.id / difficulty_category.id / PENDING_LAYOUT_ID
        target_dir = self.roi_root / map_category.id / difficulty_category.id / layout_category.id
        moves: list[tuple[Path, Path]] = []
        for sample_id in sample_ids:
            source = pending_dir / f"{sample_id}.png"
            target = target_dir / source.name
            if not source.exists():
                # The intended workflow allows users to manually remove dirty
                # pending PNGs before confirming the layout. Missing files are
                # therefore skipped instead of aborting the whole batch.
                continue
            if target.exists():
                raise FileExistsError(f"目标布局已存在同名图片：{target}")
            moves.append((source, target))

        if not moves:
            try:
                pending_dir.rmdir()
            except OSError:
                pass
            return []

        moved: list[tuple[Path, Path]] = []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for source, target in moves:
                os.replace(source, target)
                moved.append((target, source))
            self._write_layout_manifest(map_category, difficulty_category, layout_category)
        except Exception:
            for target, source in reversed(moved):
                source.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not source.exists():
                    os.replace(target, source)
            raise

        try:
            pending_dir.rmdir()
        except OSError:
            pass
        return [
            {
                "sample_id": source.stem,
                "layout_id": layout_category.id,
                "roi_image_path": target.relative_to(self.project_root).as_posix(),
            }
            for source, target in moves
        ]

    def delete_sample(self, sample_id: str, *, image_path: Optional[Path] = None) -> dict[str, Any]:
        """Delete a PNG permanently and rebuild its index; keep no discarded samples."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", sample_id):
            raise ValueError("无效的样本编号")
        matches = list(self.roi_root.rglob(f"{sample_id}.png"))
        if len(matches) != 1:
            raise ValueError(f"样本图片不存在或不唯一：{sample_id}")
        source = matches[0].resolve()
        relative_parts = source.relative_to(self.roi_root).parts
        if len(relative_parts) != 4:
            raise ValueError(f"样本图片目录结构无效：{source}")
        if image_path is not None and source != image_path.resolve():
            raise ValueError("指定图片不是样本目录中的目标文件")
        map_id, difficulty_id, layout_id, _filename = relative_parts
        # Only retain bytes in memory during the transaction, in case index I/O fails.
        original_bytes = source.read_bytes()
        source.unlink()
        try:
            if layout_id != PENDING_LAYOUT_ID:
                self._write_layout_manifest(
                    Category(map_id, map_id), Category(difficulty_id, difficulty_id),
                    LayoutCategory(layout_id, layout_id, map_id, difficulty_id),
                )
        except Exception:
            source.write_bytes(original_bytes)
            raise
        return {"sample_id": sample_id, "deleted": True}

    def rebuild_all_layout_manifests(self) -> dict[str, int]:
        pattern = re.compile(r"^layout_(\d+)$")
        expected_manifests: set[Path] = set()
        layout_count = 0
        image_count = 0

        if self.roi_root.exists():
            for map_dir in sorted(path for path in self.roi_root.iterdir() if path.is_dir()):
                for difficulty_dir in sorted(path for path in map_dir.iterdir() if path.is_dir()):
                    for layout_dir in sorted(path for path in difficulty_dir.iterdir() if path.is_dir()):
                        if not pattern.fullmatch(layout_dir.name):
                            continue
                        images = list(layout_dir.glob("*.png"))
                        manifest_path = self._layout_manifest_path(
                            map_dir.name, difficulty_dir.name, layout_dir.name
                        )
                        if images:
                            self._write_layout_manifest(
                                Category(map_dir.name, map_dir.name),
                                Category(difficulty_dir.name, difficulty_dir.name),
                                LayoutCategory(
                                    layout_dir.name,
                                    layout_dir.name,
                                    map_dir.name,
                                    difficulty_dir.name,
                                ),
                            )
                            expected_manifests.add(manifest_path.resolve())
                            layout_count += 1
                            image_count += len(images)
                        else:
                            manifest_path.unlink(missing_ok=True)

        stale_removed = 0
        if self.manifest_root.exists():
            for manifest_path in self.manifest_root.rglob("layout_*.jsonl"):
                if manifest_path.resolve() not in expected_manifests:
                    manifest_path.unlink(missing_ok=True)
                    stale_removed += 1
        return {
            "layouts": layout_count,
            "images": image_count,
            "stale_removed": stale_removed,
        }


