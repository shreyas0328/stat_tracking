"""End-to-end baseline: detect + track + render + evaluate, all in one command.

Default behavior reproduces the pretrained YOLO11n + BoT-SORT baseline on
one SportsMOT basketball val sequence. Outputs land under outputs/baselines/
so you can compare runs over time (e.g. pretrained vs. fine-tuned).

Usage:
    # First time on a fresh checkout: also auto-downloads the sample seq
    python scripts/run_baseline.py

    # Specific sequence (must already be downloaded under data/sportsmot_raw/)
    python scripts/run_baseline.py --seq v_BgwzTUxJaeU_c008

    # Compare your fine-tuned model later
    python scripts/run_baseline.py \\
        --weights runs/detect/train/weights/best.pt \\
        --tag finetuned_yolo11s \\
        --no-class-filter

Outputs (per run, under outputs/baselines/):
    {seq}_{tag}.txt       MOT-format predictions
    {seq}_{tag}.mp4       annotated video
    {seq}_{tag}.json      tracking metrics + run metadata
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEQ = "v_BgwzTUxJaeU_c008"   # the seq scripts/download_sportsmot_sample.py grabs by default


def _find_seq_dir(data_root: Path, seq: str) -> Path | None:
    """Look for a seq under any of train/val/test."""
    for split in ("train", "val", "test"):
        candidate = data_root / split / seq
        if (candidate / "img1").is_dir():
            return candidate
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", default=DEFAULT_SEQ, help="SportsMOT sequence name (e.g. v_BgwzTUxJaeU_c008)")
    p.add_argument("--data-root", type=Path, default=Path("data/sportsmot_raw"))
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO checkpoint")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--device", default=None, help="torch device: cpu, mps, or cuda id")
    p.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Path to a tracker YAML, or one of Ultralytics' built-ins "
             "('botsort.yaml', 'bytetrack.yaml'). Use "
             "'configs/botsort_persistent.yaml' for the basketball-tuned "
             "config with ReID enabled and longer track buffer.",
    )
    p.add_argument(
        "--no-class-filter",
        action="store_true",
        help="Don't filter to class 0. Use for fine-tuned single-class models.",
    )
    p.add_argument(
        "--tag",
        default="pretrained_yolo11n",
        help="Short label for this run; used in output filenames.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip the annotated MP4 to save time/disk on quick reruns",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/baselines"),
        help="Where to put predictions / video / metrics",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Locate (or auto-download) the sequence.
    seq_dir = _find_seq_dir(args.data_root, args.seq)
    if seq_dir is None:
        print(f"[setup] {args.seq} not on disk, attempting to download...")
        cmd = [sys.executable, str(ROOT / "scripts" / "download_sportsmot_sample.py"), "--num", "1"]
        # If user asked for a different seq than the default downloader picks,
        # we can't easily force it without parsing the tar -- just download
        # whatever is first and continue.
        rc = subprocess.call(cmd)
        if rc != 0:
            print("[setup] download failed. See message above.")
            return rc
        seq_dir = _find_seq_dir(args.data_root, args.seq)
        if seq_dir is None:
            # The downloader picked a different seq than the user asked for.
            # Find any basketball seq it did download.
            for split_dir in (args.data_root / "val", args.data_root / "train"):
                if split_dir.exists():
                    cands = [p for p in split_dir.iterdir() if (p / "img1").is_dir()]
                    if cands:
                        seq_dir = cands[0]
                        args.seq = seq_dir.name
                        print(f"[setup] using {args.seq} (downloaded by default).")
                        break
        if seq_dir is None:
            print("[setup] could not resolve a sequence after download.")
            return 2

    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    print(f"[setup] sequence: {args.seq}")
    print(f"[setup] frames  : {len(list(img_dir.glob('*.jpg')))}")
    print(f"[setup] gt      : {'yes' if gt_path.exists() else 'NO (cannot evaluate)'}")
    print()

    # 2. Detect + track.
    pred_path  = args.out / f"{args.seq}_{args.tag}.txt"
    video_path = args.out / f"{args.seq}_{args.tag}.mp4"
    json_path  = args.out / f"{args.seq}_{args.tag}.json"

    classes = None if args.no_class_filter else [0]
    # If user gave a relative path to a custom tracker config, resolve it to
    # an absolute path; Ultralytics' loader looks for built-in names first
    # then falls back to the path on disk.
    tracker_arg = args.tracker
    if tracker_arg not in ("botsort.yaml", "bytetrack.yaml"):
        candidate = Path(tracker_arg)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        if candidate.exists():
            tracker_arg = str(candidate)
        else:
            print(f"[warn] tracker config not found at {candidate}; "
                  f"falling back to Ultralytics built-in {args.tracker!r}")
    print(f"[detect] running {args.weights} + tracker={tracker_arg} on "
          f"{len(list(img_dir.glob('*.jpg')))} frames...")
    t0 = time.time()
    from src.tracking.botsort_runner import track_video
    df = track_video(
        weights=args.weights,
        source=str(img_dir),
        tracker=tracker_arg,
        classes=classes,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        out_mot=str(pred_path),
        save_video=False,
    )
    detect_dt = time.time() - t0
    n_frames  = int(df["frame"].max()) if not df.empty else 0
    n_dets    = len(df)
    n_tracks  = int(df["track_id"].nunique()) if not df.empty else 0
    fps       = n_frames / max(detect_dt, 1e-6)
    print(f"[detect] {n_dets:,} detections | {n_tracks} unique track IDs | "
          f"{detect_dt:.1f}s ({fps:.1f} fps)")

    # 3. Render annotated MP4 (optional).
    if not args.no_video:
        print(f"[render] writing annotated MP4 to {video_path}...")
        from src.viz.overlay import render_video
        render_video(str(img_dir), df, str(video_path), fps=25.0)

    # 4. Evaluate against ground truth.
    metrics: dict | None = None
    if gt_path.exists():
        print(f"[eval]   computing MOTA / IDF1 vs {gt_path}...")
        from src.eval.trackeval_wrapper import evaluate_sequence
        summary = evaluate_sequence(gt_path, pred_path)
        # summary is a one-row DataFrame; convert to a flat dict.
        row = summary.iloc[0].to_dict()
        metrics = {k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()}
    else:
        print("[eval]   no ground truth available; skipping metrics.")

    # 5. Save run metadata + metrics as JSON for later comparison.
    record = {
        "tag": args.tag,
        "weights": args.weights,
        "tracker": tracker_arg,
        "seq": args.seq,
        "frames": n_frames,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "class_filter": classes,
        "n_detections": n_dets,
        "n_unique_tracks": n_tracks,
        "detect_seconds": round(detect_dt, 2),
        "detect_fps": round(fps, 2),
        "metrics": metrics,
        "predictions_path": str(pred_path),
        "video_path": None if args.no_video else str(video_path),
        "ground_truth_path": str(gt_path) if gt_path.exists() else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(record, indent=2))

    # 6. Pretty-print a summary.
    print()
    print("=" * 64)
    print(f"BASELINE: {args.tag}  on  {args.seq}")
    print("=" * 64)
    print(f"  Frames               : {n_frames:,}")
    print(f"  Detections (pred/gt) : {n_dets:,}", end="")
    if gt_path.exists():
        gt_count = sum(1 for _ in gt_path.open())
        print(f" / {gt_count:,}")
    else:
        print()
    print(f"  Unique track IDs     : {n_tracks}")
    print(f"  Inference time       : {detect_dt:.1f}s ({fps:.1f} fps)")
    if metrics is not None:
        print()
        print("  Tracking metrics (motmetrics):")
        for k in ("mota", "idf1", "idp", "idr", "motp",
                  "num_switches", "num_false_positives", "num_misses",
                  "mostly_tracked", "mostly_lost"):
            if k in metrics:
                v = metrics[k]
                fmt = f"{v:.4f}" if isinstance(v, float) and abs(v) < 100 else f"{int(v)}"
                print(f"    {k:>22}: {fmt}")
    print()
    print(f"Saved:")
    print(f"  predictions  -> {pred_path}")
    if not args.no_video:
        print(f"  annotated mp4-> {video_path}")
    print(f"  metrics json -> {json_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
