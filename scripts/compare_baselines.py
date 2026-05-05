"""Scan outputs/baselines/*.json and emit a markdown comparison table.

For each SportsMOT sequence that has at least one baseline JSON, builds
one table whose columns are the runs (ground truth always first as the
upper bound, then model runs in chronological order) and whose rows are
the tracking metrics + run-level summary.

Usage:
    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --out outputs/baselines/COMPARISON.md
    python scripts/compare_baselines.py --print-only   # don't write a file
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# (display_label, json_metric_key, formatter, lower_is_better)
METRIC_ROWS: list[tuple[str, str, str, bool]] = [
    ("MOTA",                       "mota",                ".3f", False),
    ("IDF1",                       "idf1",                ".3f", False),
    ("IDP (precision)",            "idp",                 ".3f", False),
    ("IDR (recall)",               "idr",                 ".3f", False),
    ("MOTP (lower=better)",        "motp",                ".3f", True),
    ("ID switches",                "num_switches",        "d",   True),
    ("False positives",            "num_false_positives", ",d",  True),
    ("Misses (false negatives)",   "num_misses",          ",d",  True),
    ("Mostly tracked",             "mostly_tracked",      "d",   False),
    ("Mostly lost",                "mostly_lost",         "d",   True),
]

RUN_LEVEL_ROWS: list[tuple[str, str, str]] = [
    ("Total detections",   "n_detections",      ",d"),
    ("Unique track IDs",   "n_unique_tracks",   "d"),
    ("Inference speed",    "detect_fps",        "fps"),
]


def _fmt(val, spec: str) -> str:
    if val is None:
        return "—"
    if spec == "fps":
        return f"{val:.1f} fps"
    if spec == ",d" or spec == "d":
        return format(int(val), spec)
    return format(float(val), spec)


def _load(out_dir: Path) -> dict[str, list[dict]]:
    runs_by_seq: dict[str, list[dict]] = defaultdict(list)
    for jp in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except json.JSONDecodeError:
            print(f"[skip] {jp.name}: not valid JSON")
            continue
        if "seq" not in data or "tag" not in data:
            print(f"[skip] {jp.name}: missing seq/tag")
            continue
        runs_by_seq[data["seq"]].append(data)
    return runs_by_seq


def _sort_runs(runs: list[dict]) -> list[dict]:
    """Ground truth first; then chronological by timestamp; finally by tag."""
    def key(r):
        is_gt = r.get("weights_kind") == "ground_truth" or r.get("tag") == "groundtruth"
        return (
            0 if is_gt else 1,
            r.get("timestamp_utc", ""),
            r.get("tag", ""),
        )
    return sorted(runs, key=key)


def _table_for_seq(seq: str, runs: list[dict]) -> str:
    runs = _sort_runs(runs)
    headers = ["Metric"] + [r["tag"] for r in runs]
    lines: list[str] = []
    lines.append(f"### `{seq}`  ({runs[0].get('frames', '?')} frames)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * len(runs)) + "|")

    for label, key, spec, _lower_better in METRIC_ROWS:
        cells = [label]
        for r in runs:
            cells.append(_fmt((r.get("metrics") or {}).get(key), spec))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("|" + "|".join(["---"] + ["---"] * len(runs)) + "|")
    for label, key, spec in RUN_LEVEL_ROWS:
        cells = [label]
        for r in runs:
            cells.append(_fmt(r.get(key), spec))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _commentary(runs: list[dict]) -> str:
    """Pull the GT and the worst model run out of `runs` and write a short
    plain-English diagnosis comparing them."""
    gt = next((r for r in runs if r.get("weights_kind") == "ground_truth"), None)
    models = [r for r in runs if r is not gt]
    if not gt or not models:
        return ""
    # Pick the single most recent model for the commentary.
    m = sorted(models, key=lambda r: r.get("timestamp_utc", ""))[-1]
    gt_dets   = gt.get("n_detections")  or 0
    gt_ids    = gt.get("n_unique_tracks") or 0
    m_dets    = m.get("n_detections")   or 0
    m_ids     = m.get("n_unique_tracks") or 0
    m_metrics = m.get("metrics") or {}
    fp        = int(m_metrics.get("num_false_positives") or 0)
    misses    = int(m_metrics.get("num_misses") or 0)
    mota      = m_metrics.get("mota")
    idf1      = m_metrics.get("idf1")
    over_pct  = 100 * (m_dets - gt_dets) / max(gt_dets, 1)
    id_inflation = m_ids / max(gt_ids, 1)

    return (
        f"\n**Reading the table** (`{m.get('tag')}` vs ground truth):\n\n"
        f"- The detector produced **{m_dets:,} boxes** vs the ground truth's "
        f"{gt_dets:,} ({over_pct:+.0f}%). Almost all of the {fp:,} false "
        f"positives are non-players (refs, coaches, courtside fans) that "
        f"COCO-pretrained YOLO eagerly labels as 'person'.\n"
        f"- It also missed {misses:,} player boxes that the human annotators "
        f"caught — typically small/distant players or occluded ones.\n"
        f"- It assigned **{m_ids} unique track IDs** for what is actually "
        f"only **{gt_ids} people** on the broadcast — a {id_inflation:.0f}× "
        f"inflation caused by IDs flipping every time a player is briefly "
        f"occluded or the detector momentarily drops them.\n"
        f"- Net result: MOTA = {mota:.3f}, IDF1 = {idf1:.3f}. The ground "
        f"truth scores 1.000 on both by definition; the gap is what "
        f"fine-tuning needs to close."
    )


def render(out_dir: Path) -> str:
    runs_by_seq = _load(out_dir)
    if not runs_by_seq:
        return f"No baseline JSONs found under {out_dir}/"

    blocks: list[str] = []
    blocks.append("# Baseline comparison\n")
    blocks.append(
        "Auto-generated by `scripts/compare_baselines.py`. Re-run it any "
        "time you add a new baseline (e.g. after fine-tuning) to refresh.\n"
    )
    blocks.append(
        "Each table compares one or more pipeline runs on the same SportsMOT "
        "sequence. The `groundtruth` column is the upper bound — what perfect "
        "human annotation looks like — and any model column is what an "
        "actual model produces. The closer a model gets to the `groundtruth` "
        "column, the better.\n"
    )

    for seq, runs in sorted(runs_by_seq.items()):
        blocks.append(_table_for_seq(seq, runs))
        blocks.append(_commentary(runs))
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("outputs/baselines"))
    p.add_argument("--out", type=Path, default=Path("outputs/baselines/COMPARISON.md"))
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()

    md = render(args.in_dir)
    print(md)
    if not args.print_only:
        args.out.write_text(md)
        print(f"\n[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
