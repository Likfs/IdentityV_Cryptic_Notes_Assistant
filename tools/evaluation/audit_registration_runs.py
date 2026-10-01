"""Summarize saved game captures and build a small visual-review queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit(runs: Path, day: str) -> dict:
    rows = []
    hashes = set()
    duplicates = 0
    for path in sorted(runs.glob(f"{day}*/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            screenshot = path.parent / "01_screenshot.png"
            if screenshot.is_file():
                digest = hashlib.sha256(screenshot.read_bytes()).hexdigest()
                duplicates += digest in hashes
                hashes.add(digest)
            rows.append((path.parent.name, record))
        except (ValueError, OSError):
            continue
    modes = Counter()
    groups = defaultdict(list)
    failure_types = Counter()
    route_overrides = []
    history_conflicts = []
    registrations_ms = defaultdict(list)
    for name, record in rows:
        registration = record.get("registration") or {}
        validation = registration.get("validation") or {}
        overlay = record.get("overlay_validation") or {}
        timing = registration.get("timings_ms") or {}
        mode = timing.get("method", "none")
        modes[mode] += 1
        groups[f"{record.get('selected_layout')}/{record.get('floor')}"].append(record)
        if not record.get("accepted"):
            if not registration:
                failure_types["no_registration"] += 1
            elif not registration.get("accepted"):
                failure_types["registration_rejected"] += 1
            elif not overlay.get("accepted"):
                failure_types["route_rejected"] += 1
            else:
                failure_types["other"] += 1
        if record.get("accepted") and isinstance(timing.get("total"), (int, float)):
            registrations_ms[mode].append(timing["total"])
        edge = validation.get("edge_median_error_px")
        if validation.get("route_corroborated"):
            route_overrides.append({
                "run": name, "edge_median_px": edge,
                "route_distance_p95_px": overlay.get("distance_p95_px"),
                "route_coverage": overlay.get("structure_coverage"),
            })
        conflict = validation.get("history_conflict_rate")
        if conflict is not None and conflict > 0.20:
            history_conflicts.append({
                "run": name, "rate": conflict,
                "compared_pixels": validation.get("history_compared_pixels"),
                "structure_precision": validation.get("structure_precision"),
                "semantic_precision": validation.get("semantic_precision"),
                "edge_median_px": edge,
            })
    route_overrides.sort(key=lambda item: item["edge_median_px"] or 0, reverse=True)
    history_conflicts.sort(key=lambda item: (
        (item["structure_precision"] or 0) >= 0.9,
        (item["semantic_precision"] or 0) >= 0.8,
        -(item["edge_median_px"] or 999), item["rate"],
    ), reverse=True)
    return {
        "day": day,
        "summary": {
            "runs": len(rows), "unique_screenshots": len(hashes), "duplicate_screenshots": duplicates,
            "accepted": sum(bool(record.get("accepted")) for _, record in rows),
            "failure_types": dict(failure_types),
            "layout_floor_count": sum(not name.endswith("/unknown") for name in groups),
            "modes": dict(modes),
            "route_overrides": len(route_overrides),
            "route_overrides_edge_over_10px": sum(
                (item["edge_median_px"] or 0) > 10 for item in route_overrides
            ),
            "history_conflicts_over_20_percent": len(history_conflicts),
            "accepted_registration_timing_ms": {
                mode: {"count": len(values), "median": round(float(statistics.median(values)), 1),
                       "p95": round(float(sorted(values)[int((len(values) - 1) * 0.95)]), 1)}
                for mode, values in registrations_ms.items()
            },
        },
        "groups": {name: {"runs": len(values), "accepted": sum(bool(row.get("accepted")) for row in values)}
                   for name, values in sorted(groups.items())},
        "review_queue": {
            "route_overrides_high_edge": route_overrides[:20],
            "history_conflicts": history_conflicts[:30],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="20260925")
    parser.add_argument("--runs", type=Path, default=ROOT / 'runtime/diagnostics/route_projection/app_runs')
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.runs, args.day)
    if not payload["summary"]["runs"]:
        parser.error("诊断目录或日期下没有有效运行记录，不能生成空的成功报告")
    output = (args.output or ROOT / 'workspace/reports/registration' / f"audit_{args.day}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
