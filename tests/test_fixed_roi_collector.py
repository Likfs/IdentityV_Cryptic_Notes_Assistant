from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.collector.fixed_roi_collector import (  # noqa: E402
    VOICE_FALLBACK_TONES,
    VOICE_PROMPTS,
    Category,
    CaptureRegion,
    LayoutCategory,
    SampleWriter,
    border_rectangles,
    image_sha256,
    load_capture_config,
    load_category_config,
    safe_label,
    validate_region_inside_screen,
)


def test_default_config_uses_requested_region() -> None:
    config = load_capture_config(PROJECT_ROOT / 'tools/configs/capture_defaults.json')
    assert config["capture_roi"] == [901, 185, 1180, 1050]
    assert config["voice_prompts_enabled"] is True


def test_voice_prompts_are_distinct_chinese_phrases() -> None:
    required = {
        "app_started",
        "sampling_started",
        "sampling_paused",
        "sampling_resumed",
        "sampling_stopped",
        "single_saved",
        "layout_confirmed",
        "sample_deleted",
        "jsonl_rebuilt",
        "operation_failed",
    }
    assert required <= VOICE_PROMPTS.keys()
    assert len(set(VOICE_PROMPTS.values())) == len(VOICE_PROMPTS)
    assert all(any("\u4e00" <= char <= "\u9fff" for char in text) for text in VOICE_PROMPTS.values())
    assert VOICE_FALLBACK_TONES.keys() == VOICE_PROMPTS.keys()
    assert len(set(VOICE_FALLBACK_TONES.values())) == len(VOICE_FALLBACK_TONES)


def test_default_context_is_doom_nightmare() -> None:
    categories = load_category_config(PROJECT_ROOT / 'tools/configs/labels.json')
    assert categories["default_map_id"] == "a_bringer_of_doom"
    assert categories["default_difficulty_id"] == "nightmare"
    assert categories["maps"][0].name == "厄运之女"
    assert categories["difficulties"][0].name == "噩梦"


def test_region_validation_accepts_requested_region() -> None:
    validate_region_inside_screen(
        CaptureRegion(901, 185, 1180, 1050),
        CaptureRegion(0, 0, 2560, 1440),
    )


def test_region_validation_rejects_outside_region() -> None:
    try:
        validate_region_inside_screen(CaptureRegion(901, 185, 1180, 1050), CaptureRegion(0, 0, 1920, 1080))
    except ValueError as exc:
        assert "超出虚拟桌面" in str(exc)
    else:
        raise AssertionError("out-of-bounds region was accepted")


def test_border_stays_outside_capture_pixels() -> None:
    region = CaptureRegion(901, 185, 1180, 1050)
    assert border_rectangles(region, 3) == (
        (898, 182, 1186, 3),
        (898, 1235, 1186, 3),
        (898, 185, 3, 1050),
        (2081, 185, 3, 1050),
    )


def test_sample_writer_saves_pending_png_without_metadata(tmp_path: Path) -> None:
    image = Image.new("RGB", (12, 8), (10, 20, 30))
    writer = SampleWriter(tmp_path)
    record = writer.save(
        image,
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=Category("a_bringer_of_doom", "厄运之女"),
        difficulty_category=Category("nightmare", "噩梦"),
        layout_category=LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare"),
    )

    image_path = tmp_path / record["roi_image_path"]
    assert image_path.exists()
    assert image_path.parent.name == "pending"
    assert image_path.parent.parent.name == "nightmare"
    assert image_path.parent.parent.parent.name == "a_bringer_of_doom"
    assert record["layout_id"] == "pending"
    assert record["label_status"] == "pending"
    assert not (tmp_path / 'workspace/manifests').exists()


def test_sample_writer_rejects_layout_from_another_context(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    try:
        writer.save(
            Image.new("RGB", (12, 8)),
            region=CaptureRegion(901, 185, 1180, 1050),
            map_category=Category("a_bringer_of_doom", "厄运之女"),
            difficulty_category=Category("nightmare", "噩梦"),
            layout_category=LayoutCategory("layout_01", "布局 01", "another_map", "nightmare"),
        )
    except ValueError as exc:
        assert "不属于当前地图和难度" in str(exc)
    else:
        raise AssertionError("cross-context layout was accepted")


def test_assigns_pending_samples_to_confirmed_layout(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    pending = writer.save(
        Image.new("RGB", (12, 8), (1, 2, 3)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare"),
    )
    assert pending["label_status"] == "pending"

    updated = writer.assign_pending_layout(
        [pending["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("layout_09", "布局 09", "a_bringer_of_doom", "nightmare"),
    )

    assert len(updated) == 1
    manifest = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_09.jsonl'
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["label"] == "layout_09"
    assert set(record) == {"image", "label"}
    assert (tmp_path / record["image"]).exists()


def test_supplements_an_existing_layout_and_rebuilds_its_manifest(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    pending_layout = LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare")
    target_layout = LayoutCategory("layout_03", "布局 03", "a_bringer_of_doom", "nightmare")

    first = writer.save(
        Image.new("RGB", (12, 8), (1, 2, 3)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=pending_layout,
    )
    writer.assign_pending_layout(
        [first["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=target_layout,
    )
    supplement = writer.save(
        Image.new("RGB", (12, 8), (4, 5, 6)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=pending_layout,
    )
    updated = writer.assign_pending_layout(
        [supplement["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=target_layout,
    )

    manifest = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_03.jsonl'
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(updated) == 1
    assert writer.layout_sample_count("a_bringer_of_doom", "nightmare", "layout_03") == 2
    assert len(records) == 2
    assert {record["label"] for record in records} == {"layout_03"}


def test_archive_skips_pending_images_deleted_manually(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    pending_layout = LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare")
    kept = writer.save(
        Image.new("RGB", (12, 8), (1, 2, 3)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=pending_layout,
    )
    deleted = writer.save(
        Image.new("RGB", (12, 8), (4, 5, 6)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=pending_layout,
    )
    (tmp_path / deleted["roi_image_path"]).unlink()

    updated = writer.assign_pending_layout(
        [kept["sample_id"], deleted["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("layout_01", "布局 01", "a_bringer_of_doom", "nightmare"),
    )

    assert [item["sample_id"] for item in updated] == [kept["sample_id"]]
    manifest = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_01.jsonl'
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert records == [{"image": updated[0]["roi_image_path"], "label": "layout_01"}]


def test_archive_all_deleted_pending_images_does_not_consume_layout_number(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    pending = writer.save(
        Image.new("RGB", (12, 8)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare"),
    )
    (tmp_path / pending["roi_image_path"]).unlink()

    updated = writer.assign_pending_layout(
        [pending["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=writer.next_layout(map_category, difficulty_category),
    )

    assert updated == []
    assert writer.next_layout(map_category, difficulty_category).id == "layout_01"
    assert not (tmp_path / 'workspace/manifests').exists()


def test_layout_numbers_grow_without_a_fixed_limit(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    assert writer.next_layout(map_category, difficulty_category).id == "layout_01"
    writer.save(
        Image.new("RGB", (12, 8)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("layout_99", "布局 99", "a_bringer_of_doom", "nightmare"),
    )
    assert writer.next_layout(map_category, difficulty_category).id == "layout_100"


def test_delete_sample_removes_image_and_rebuilds_layout_jsonl(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    record = writer.save(
        Image.new("RGB", (12, 8), (9, 8, 7)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=LayoutCategory("pending", "待确认布局", "a_bringer_of_doom", "nightmare"),
    )
    layout = LayoutCategory("layout_01", "布局 01", "a_bringer_of_doom", "nightmare")
    writer.assign_pending_layout(
        [record["sample_id"]],
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=layout,
    )
    original = tmp_path / 'workspace/datasets/layout_classification/images/a_bringer_of_doom/nightmare/layout_01' / f"{record['sample_id']}.png"

    result = writer.delete_sample(record["sample_id"])

    assert not original.exists()
    assert result == {"sample_id": record["sample_id"], "deleted": True}
    assert list(tmp_path.rglob("*.png")) == []
    assert not (
        tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_01.jsonl'
    ).exists()
    assert not (tmp_path / 'workspace/rejected').exists()


def test_rebuild_all_layout_jsonl_after_manual_file_deletion(tmp_path: Path) -> None:
    writer = SampleWriter(tmp_path)
    map_category = Category("a_bringer_of_doom", "厄运之女")
    difficulty_category = Category("nightmare", "噩梦")
    layout = LayoutCategory("layout_01", "布局 01", "a_bringer_of_doom", "nightmare")
    first = writer.save(
        Image.new("RGB", (12, 8), (1, 1, 1)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=layout,
    )
    second = writer.save(
        Image.new("RGB", (12, 8), (2, 2, 2)),
        region=CaptureRegion(901, 185, 1180, 1050),
        map_category=map_category,
        difficulty_category=difficulty_category,
        layout_category=layout,
    )
    stale = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_99.jsonl'
    stale.parent.mkdir(parents=True)
    stale.write_text('{"image":"missing.png","label":"layout_99"}\n', encoding="utf-8")
    (tmp_path / first["roi_image_path"]).unlink()

    result = writer.rebuild_all_layout_manifests()

    manifest = tmp_path / 'workspace/manifests/a_bringer_of_doom/nightmare/layout_01.jsonl'
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert result == {"layouts": 1, "images": 1, "stale_removed": 1}
    assert records == [{"image": second["roi_image_path"], "label": "layout_01"}]
    assert not stale.exists()


def test_failed_index_update_restores_sample_without_disk_backup(tmp_path, monkeypatch):
    writer = SampleWriter(tmp_path)
    record = writer.save(
        Image.new("RGB", (12, 8)), region=CaptureRegion(0, 0, 12, 8),
        map_category=Category("map", "地图"), difficulty_category=Category("nightmare", "噩梦"),
        layout_category=LayoutCategory("layout_01", "布局 1", "map", "nightmare"),
    )
    writer.rebuild_all_layout_manifests()
    original = tmp_path / record["roi_image_path"]
    data = original.read_bytes()
    manifest = next(writer.manifest_root.rglob("*.jsonl"))
    old_index = manifest.read_bytes()
    def fail(*_args):
        raise OSError("模拟索引写入失败")
    monkeypatch.setattr(writer, "_write_layout_manifest", fail)
    with pytest.raises(OSError):
        writer.delete_sample(record["sample_id"])
    assert original.read_bytes() == data
    assert manifest.read_bytes() == old_index
    assert list(tmp_path.rglob("*.png")) == [original]


def test_delete_validates_id_and_requested_file(tmp_path):
    writer = SampleWriter(tmp_path)
    record = writer.save(
        Image.new("RGB", (12, 8)), region=CaptureRegion(0, 0, 12, 8),
        map_category=Category("map", "地图"), difficulty_category=Category("nightmare", "噩梦"),
        layout_category=LayoutCategory("pending", "待确认", "map", "nightmare"),
    )
    for invalid in ("*", "../sample", "foo/bar"):
        with pytest.raises(ValueError):
            writer.delete_sample(invalid)
    with pytest.raises(ValueError, match="指定图片"):
        writer.delete_sample(record["sample_id"], image_path=tmp_path / "wrong.png")
    assert (tmp_path / record["roi_image_path"]).exists()
    writer.delete_sample(record["sample_id"])
    assert not list(tmp_path.rglob("*.png"))
