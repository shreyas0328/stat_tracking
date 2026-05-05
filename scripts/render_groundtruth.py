"""Render SportsMOT ground-truth annotations as an annotated MP4, and save
the matching `_groundtruth.json` with "perfect" metrics.

This produces the reference "what hand-annotation looks like" baseline that
sits alongside model baselines in outputs/baselines/. Compared to a model
run, the GT trivially scores MOTA=1.0 / IDF1=1.0 / 0 false positives /
0 misses / 0 ID switches -- it's the upper bound that all model runs are
trying to approach.

Usage:
    python scripts/render_groundtruth.py
    python scripts/render_groundtruth.py --seq v_BgwzTUxJaeU_c008
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEQ = "v_BgwzTUxJaeU_c008"


def _find_seq_dir(data_root: Path, seq: str) -> Path | None:
    for split in ("train", "val", "test"):
        candidate = data_root / split / seq
        if (candidate / "img1").is_dir():
            return candidate
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--data-root", type=Path, default=Path("data/sportsmot_raw"))
    p.add_argument("--out", type=Path, default=Path("outputs/baselines"))
    p.add_argument("--tag", default="groundtruth")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    seq_dir = _find_seq_dir(args.data_root, args.seq)
    if seq_dir is None:
        print(f"[error] could not find {args.seq} under {args.data_root}/")
        return 1
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        print(f"[error] no gt.txt at {gt_path} (test split has no labels)")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    json_path  = args.out / f"{args.seq}_{args.tag}.json"
    video_path = args.out / f"{args.seq}_{args.tag}.mp4"

    # gt.txt is MOT-Challenge format with 9 columns:
    #   frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis
    # The renderer's _load_predictions expects the first 7 to be named
    # frame/track_id/x/y/w/h/conf, so rename to match.
    gt_df = pd.read_csv(gt_path, header=None)
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis"]
    gt_df = gt_df.iloc[:, : len(cols)]
    gt_df.columns = cols[: gt_df.shape[1]]

    img_dir = seq_dir / "img1"
    n_frames = len(list(img_dir.glob("*.jpg")))
    n_boxes = len(gt_df)
    n_unique = int(gt_df["track_id"].nunique())

    print(f"[gt] sequence       : {args.seq}")
    print(f"[gt] frames         : {n_frames}")
    print(f"[gt] gt boxes       : {n_boxes:,}")
    print(f"[gt] unique people  : {n_unique}  (vs model's 158 inflated tracks)")

    if not args.no_video:
        from src.viz.overlay import render_video
        print(f"[gt] rendering -> {video_path} ...")
        render_video(str(img_dir), gt_df, str(video_path), fps=25.0)

    # GT vs GT through motmetrics: a sanity check that should yield perfect
    # numbers. We ALSO want this in the same JSON shape as the model
    # baselines so the comparison script can read it uniformly.
    from src.eval.trackeval_wrapper import evaluate_sequence
    summary = evaluate_sequence(gt_path, gt_path).iloc[0].to_dict()
    metrics = {k: (float(v) if hasattr(v, "item") else v) for k, v in summary.items()}

    record = {
        "tag": args.tag,
        "weights": None,
        "weights_kind": "ground_truth",   # marker so the comparator knows this row is the upper bound
        "seq": args.seq,
        "frames": n_frames,
        "imgsz": None,
        "conf": None,
        "class_filter": None,
        "n_detections": n_boxes,
        "n_unique_tracks": n_unique,
        "detect_seconds": 0.0,
        "detect_fps": None,
        "metrics": metrics,
        "predictions_path": str(gt_path),
        "video_path": None if args.no_video else str(video_path),
        "ground_truth_path": str(gt_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(record, indent=2))
    print(f"[gt] wrote          : {json_path}")
    print()
    print(f"[gt] sanity-check metrics (gt vs gt; should all be perfect):")
    for k in ("mota", "idf1", "num_switches", "num_false_positives", "num_misses"):
        print(f"        {k:>22}: {metrics.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
