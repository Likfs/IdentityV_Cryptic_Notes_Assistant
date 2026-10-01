"""Compare known-pose outcomes and prepare an explicitly unlabelled real review queue."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def accepted_wrong(outcome: dict) -> bool:
    error = outcome.get("p95_screen_error_px")
    return bool(outcome["accepted"] and (error is None or not np.isfinite(error) or error > 10))


def accepted_correct(outcome: dict) -> bool:
    return bool(outcome["accepted"] and not accepted_wrong(outcome))


def selected_outcome(report, row):
    key = report.get("backend", "optimized")
    if key not in row.get("outcomes", {}):
        raise ValueError(f"报告缺少所选后端结果：{key}")
    return row["outcomes"][key]


def compare_synthetic(before: dict, after: dict) -> dict:
    key = lambda row: (row["layout"], row["floor"], row["level"])
    previous = {key(row): row for row in before["rows"]}
    current = {key(row): row for row in after["rows"]}
    if (not previous or not current or len(previous) != len(before["rows"]) or
            len(current) != len(after["rows"]) or previous.keys() != current.keys()):
        raise ValueError("Synthetic case sets differ; refusing a partial comparison")
    changes, review = [], []
    newly_wrong = lost_correct = corrected = newly_correct = 0
    for identity, row in current.items():
        original = previous[identity]
        if not np.allclose(original["reference_clean_to_screen"], row["reference_clean_to_screen"]):
            raise ValueError(f"Reference changed: {identity}")
        for field in ("template_sha256", "screenshot_sha256"):
            if field in original and field in row and original[field] != row[field]:
                raise ValueError(f"Synthetic input changed: {identity}/{field}")
        a, b = selected_outcome(before, original), selected_outcome(after, row)
        newly_wrong += accepted_wrong(b) and not accepted_wrong(a)
        lost_correct += accepted_correct(a) and not accepted_correct(b)
        corrected += accepted_wrong(a) and not accepted_wrong(b)
        newly_correct += accepted_correct(b) and not accepted_correct(a)
        record = {"case": "/".join(identity), "before": a, "after": b,
                  "reference_clean_to_screen": row["reference_clean_to_screen"]}
        if a["accepted"] != b["accepted"] or accepted_wrong(a) != accepted_wrong(b):
            changes.append(record)
        if accepted_wrong(a) or accepted_wrong(b):
            review.append(record)
    return {"cases": len(current), "new_incorrect_acceptances": newly_wrong,
            "before_backend": before.get("backend", "optimized"),
            "after_backend": after.get("backend", "optimized"),
            "lost_correct_acceptances": lost_correct, "incorrect_acceptances_removed": corrected,
            "new_correct_acceptances": newly_correct, "changes": changes,
            "known_pose_review": review}


def real_review_queue(replay: dict) -> dict:
    if not replay.get("rows"):
        raise ValueError("回放报告为空，不能生成有效复核队列")
    rows, sessions = [], {}
    for row in replay["rows"]:
        session = row.get("session_id")
        # Never invent a game/session id for old logs. Layout fallback is marked.
        group = f"session:{session}" if session else f"layout:{row['layout']}"
        fold = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5
        split = "holdout" if fold == 0 else "development"
        sessions[group] = split
        regional = row.get("validation", {}).get("regional_evidence", {})
        reasons = []
        if not row["accepted"]:
            reasons.append("rejected_without_ground_truth")
        if row["accepted"] and row.get("pose_valid") is False:
            reasons.append("accepted_invalid_pose")
        if regional.get("contradictory_regions", 0) >= 2:
            reasons.append("regional_contradiction")
        if regional.get("scale_profile_span", 0) >= .04:
            reasons.append("scale_ambiguity")
        if row.get("reference_accepted") is not None and row["reference_accepted"] != row["accepted"]:
            reasons.append("acceptance_changed")
        if (row.get("p95_screen_delta_vs_reference", row.get("p95_screen_delta_vs_baseline")) or 0) > 10:
            reasons.append("pose_changed")
        if reasons:
            rows.append({"source": row["source"], "layout": row["layout"], "floor": row["floor"],
                         "screenshot": str(ROOT / row["image"]) if row.get("image") else
                         str(ROOT / 'runtime/diagnostics/route_projection/app_runs' / row["source"] / "01_screenshot.png"),
                         "screenshot_sha256": row["sha256"], "template_sha256": row["template_sha256"],
                         "session_id": session, "split_group": group, "split": split,
                         "split_kind": "session" if session else "layout_fallback_not_independent_game",
                         "review_reasons": reasons, "human_verified": False,
                         "correspondences_screen_to_clean": [], "reviewer": None,
                         "ground_truth_clean_to_screen": None})
    return {"human_labelled_cases": 0, "session_group_count": sum(k.startswith("session:") for k in sessions),
            "missing_session_cases": sum(not row.get("session_id") for row in replay["rows"]),
            "note": "Review queue only. Archived poses and acceptance are not truth. Add spatially separated point pairs before evaluating real accuracy.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    try:
        result = compare_synthetic(read(args.before), read(args.after))
        if args.real:
            result["real_review"] = real_review_queue(read(args.real))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
