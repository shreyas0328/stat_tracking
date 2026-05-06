"""Iteration 2 of the persistent-ID effort: post-process track merger.

Takes an existing tracker output (e.g. from
``scripts/run_baseline.py --tracker configs/botsort_persistent.yaml``),
clusters the resulting fragmented tracks into a small number of stable
identities by appearance similarity, and writes a merged prediction file +
fresh metrics JSON so it's directly comparable to the unmerged baseline.

Typical use
-----------
::

    # Step 1: produce per-frame tracks (fragmented)
    python scripts/run_baseline.py \\
        --tracker configs/botsort_persistent.yaml \\
        --weights yolo11m.pt --tag persistent_v2_yolo11m --no-video

    # Step 2: collapse fragments into 12 clusters (10 players + 2 slack)
    python scripts/merge_tracks.py \\
        --predictions outputs/baselines/v_BgwzTUxJaeU_c008_persistent_v2_yolo11m.txt \\
        --frames data/sportsmot_raw/val/v_BgwzTUxJaeU_c008/img1 \\
        --tag persistent_v2_yolo11m_merged_k12 \\
        --n-clusters 12

How merging is constrained
--------------------------
We use agglomerative clustering on cosine distance between mean per-track
appearance embeddings (EfficientNet-B0 features), with one critical
constraint: any two tracks that co-occur in the same frame are forbidden
from merging (because a player can't be in two places at once). This kills
the most common false-positive merge.

Picking n_clusters vs distance_threshold
-----------------------------------------
* ``--n-clusters K`` forces exactly K identities. Use this when you know
  the team size (10 players, maybe + 2 for refs/coaches => K=12).
* ``--distance-threshold T`` lets the algorithm decide; merge tracks whose
  embeddings are within cosine distance T. Try T ~ 0.30-0.45 as a starting
  point; lower => more clusters, higher => fewer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PREDS = ROOT / "outputs/baselines/v_BgwzTUxJaeU_c008_persistent_v2_yolo11m.txt"
DEFAULT_FRAMES = ROOT / "data/sportsmot_raw/val/v_BgwzTUxJaeU_c008/img1"
DEFAULT_GT = ROOT / "data/sportsmot_raw/val/v_BgwzTUxJaeU_c008/gt/gt.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDS,
                   help="Existing MOT-format predictions to merge.")
    p.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                   help="Source frames directory (000001.jpg, ...).")
    p.add_argument("--gt", type=Path, default=DEFAULT_GT,
                   help="Optional ground-truth file for re-evaluation. "
                        "Pass an empty path to skip eval.")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs/baselines",
                   help="Where to write the merged .txt + .json.")
    p.add_argument("--tag", default="merged_k12",
                   help="Tag appended to the output filename.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--n-clusters", type=int,
                     help="Force exactly this many identity clusters.")
    grp.add_argument("--distance-threshold", type=float,
                     help="Merge tracks within this cosine-distance threshold.")
    p.add_argument("--samples-per-track", type=int, default=12,
                   help="How many crops to embed per track for the mean embedding.")
    p.add_argument("--min-detections", type=int, default=8,
                   help="Drop tracks shorter than this — too short to embed reliably.")
    p.add_argument("--min-bbox-height", type=int, default=80,
                   help="Prefer crops at least this tall when sampling.")
    p.add_argument("--backbone", default="efficientnet_b0",
                   help="timm model name to use as the embedder.")
    p.add_argument("--device", default=None, help="cpu, cuda, or mps")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Don't enforce the 'tracks can't merge if they overlap in time' constraint. "
                        "Almost always wrong; useful only for ablation.")
    p.add_argument("--min-cluster-silhouette", type=float, default=0.0,
                   help="After clustering, any cluster with mean silhouette below this value is "
                        "treated as a 'garbage merge' (multiple distinct people incorrectly grouped) "
                        "and auto-split: each member fragment becomes its own ID. Default 0.0 means "
                        "any negative-silhouette cluster gets split. Lower (e.g. -0.2) is more lenient "
                        "and produces fewer IDs at the risk of the 'multiple people share one ID' bug.")
    p.add_argument("--min-final-detections", type=int, default=30,
                   help="After auto-split, drop any cluster with fewer than this many total "
                        "detections from the output (.txt + .mp4). These are almost always brief "
                        "noise tracks (refs walking through, fans in foreground, half-occluded "
                        "glimpses) that survived the tracker but aren't real persistent players. "
                        "Default 30 (~1.2s at 25fps, ~6%% of a 500-frame clip). Set to 0 to keep "
                        "every cluster in the output, useful for diagnostics.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip rendering the merged annotated MP4.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.predictions.exists():
        print(f"ERROR: predictions file not found: {args.predictions}", file=sys.stderr)
        return 1
    if not args.frames.exists():
        print(f"ERROR: frames directory not found: {args.frames}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seq = args.predictions.stem.split("_")[0:3]  # heuristic: e.g. 'v_BgwzTUxJaeU_c008'
    seq_name = "_".join(seq) if seq else args.predictions.stem
    out_pred  = args.out_dir / f"{seq_name}_{args.tag}.txt"
    out_video = args.out_dir / f"{seq_name}_{args.tag}.mp4"
    out_json  = args.out_dir / f"{seq_name}_{args.tag}.json"

    print(f"[input] predictions: {args.predictions}")
    print(f"[input] frames:      {args.frames}")
    print(f"[input] mode:        "
          f"{'n_clusters=' + str(args.n_clusters) if args.n_clusters is not None else 'distance_threshold=' + str(args.distance_threshold)}")

    from src.tracking.track_merger import (
        load_predictions,
        compute_track_embeddings,
        cluster_tracks,
        remap_predictions,
        write_predictions,
    )

    df = load_predictions(args.predictions)
    print(f"[input] {len(df)} detections, {df['track_id'].nunique()} unique tracks before merge")

    t0 = time.time()
    print("[embed] computing per-track appearance embeddings...")
    track_embs = compute_track_embeddings(
        df,
        frames_dir=args.frames,
        samples_per_track=args.samples_per_track,
        min_detections=args.min_detections,
        min_bbox_height=args.min_bbox_height,
        backbone=args.backbone,
        device=args.device,
    )
    print(f"[embed] embedded {len(track_embs)} tracks in {time.time() - t0:.1f}s")

    if not track_embs:
        print("[merge] no tracks long enough to merge; nothing to do.")
        return 2

    print("[merge] running constrained agglomerative clustering...")
    result = cluster_tracks(
        track_embs,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        forbid_overlap=not args.allow_overlap,
        min_cluster_silhouette=args.min_cluster_silhouette,
    )

    merged_df = remap_predictions(df, result.mapping, drop_unmapped=True, id_offset=1)

    # Drop noise singletons. A real player is on screen for hundreds of
    # frames; tracks with only a handful of detections are almost always
    # refs/fans/brief glimpses that survived the tracker. Filtering here
    # (rather than in the merger itself) keeps cluster_tracks() pure --
    # it always returns the full clustering and the caller decides what
    # to expose downstream.
    keep_ids = set()
    if args.min_final_detections > 0:
        counts = merged_df.groupby("track_id").size()
        keep_ids = set(int(t) for t, n in counts.items() if n >= args.min_final_detections)
        n_dropped = merged_df["track_id"].nunique() - len(keep_ids)
        n_dropped_dets = int(len(merged_df) - counts[counts >= args.min_final_detections].sum())
        merged_df = merged_df[merged_df["track_id"].isin(keep_ids)].copy()
        if n_dropped > 0:
            print(f"[merge] dropped {n_dropped} noise IDs ({n_dropped_dets} dets) "
                  f"with < {args.min_final_detections} detections")

    n_after = merged_df["track_id"].nunique()
    write_predictions(merged_df, out_pred)
    print(f"[merge] wrote merged predictions ({n_after} unique IDs) to {out_pred}")

    # Translate the merger's 0-indexed cluster IDs to the 1-indexed track
    # IDs that actually appear in the predictions file, for consistency.
    # If we filtered out noise singletons above, those clusters won't be in
    # `keep_ids` -- mark them as "dropped" so the JSON still records what
    # the clustering produced, but downstream tools (e.g. comparison
    # table, paper figures) only consider what made it into the .txt.
    cluster_summary = []
    for c in sorted(result.cluster_size.keys()):
        track_id = c + 1  # matches id_offset above
        kept = (not keep_ids) or (int(track_id) in keep_ids)
        cluster_summary.append({
            "track_id": int(track_id),
            "cluster_id": int(c),
            "n_fragments_merged": int(result.cluster_size[c]),
            "n_detections": int(result.cluster_total_detections[c]),
            "silhouette": float(result.cluster_silhouette[c]),
            "confidence": float(result.cluster_confidence[c]),
            "kept_in_output": bool(kept),
            "fragment_track_ids": sorted(
                int(t) for t, lab in result.mapping.items() if int(lab) == c
            ),
        })
    cluster_summary.sort(key=lambda r: (-r["kept_in_output"], -r["n_detections"]))

    print()
    print(f"[merge] kept identities (by detection count, dropped IDs hidden):")
    for cs in cluster_summary:
        if not cs["kept_in_output"]:
            continue
        label = ("HIGH" if cs["confidence"] >= 0.75
                 else "MED " if cs["confidence"] >= 0.55
                 else "LOW ")
        print(f"[merge]   ID {cs['track_id']:2d}: conf={cs['confidence']:.0%} [{label}]  "
              f"silhouette={cs['silhouette']:+.3f}  "
              f"merged {cs['n_fragments_merged']} fragments  "
              f"({cs['n_detections']} detections)")

    metrics = None
    if args.gt and args.gt.exists():
        print(f"[eval]  computing MOTA/IDF1 vs {args.gt}...")
        from src.eval.trackeval_wrapper import evaluate_sequence
        summary = evaluate_sequence(args.gt, out_pred)
        row = summary.iloc[0].to_dict()
        metrics = {k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()}
    else:
        print("[eval]  no ground truth; skipping evaluation.")

    if not args.no_video:
        print(f"[render] writing merged annotated MP4 to {out_video}...")
        from src.viz.overlay import render_video
        render_video(str(args.frames), merged_df, str(out_video), fps=25.0)

    n_frames = int(merged_df["frame"].max()) if not merged_df.empty else 0
    record = {
        # Schema-compatible fields with run_baseline.py so this run shows up
        # in scripts/compare_baselines.py's comparison table.
        "tag": args.tag,
        "weights": "(post-process merger)",
        "tracker": "post_process_agglomerative",
        "seq": seq_name,
        "frames": n_frames,
        "n_detections": int(len(merged_df)),
        "n_unique_tracks": int(n_after),
        "detect_seconds": None,
        "detect_fps": None,
        "metrics": metrics,
        "predictions_path": str(out_pred),
        "video_path": None if args.no_video else str(out_video),
        "ground_truth_path": str(args.gt) if args.gt and args.gt.exists() else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Merger-specific fields (informational; ignored by the compare script).
        "source_predictions": str(args.predictions),
        "merge_method": (
            f"agglomerative_k={args.n_clusters}"
            if args.n_clusters is not None
            else f"agglomerative_dist<={args.distance_threshold}"
        ),
        "samples_per_track": args.samples_per_track,
        "min_detections": args.min_detections,
        "backbone": args.backbone,
        "n_tracks_before_merge": int(df["track_id"].nunique()),
        "n_detections_before_merge": int(len(df)),
        # Per-identity confidence: the headline output the user actually
        # wants ("which IDs are solid, which are tentative"). Sorted by
        # confidence descending so reading the JSON top-down shows the
        # most reliable IDs first.
        "identities": cluster_summary,
        "n_high_confidence": sum(1 for c in cluster_summary if c["confidence"] >= 0.75),
        "n_med_confidence":  sum(1 for c in cluster_summary if 0.55 <= c["confidence"] < 0.75),
        "n_low_confidence":  sum(1 for c in cluster_summary if c["confidence"] < 0.55),
    }
    out_json.write_text(json.dumps(record, indent=2))

    print()
    print("=" * 64)
    print(f"MERGED RUN: {args.tag}")
    print("=" * 64)
    print(f"  Tracks before merge       : {df['track_id'].nunique()}")
    print(f"  Tracks after merge        : {n_after}")
    print(f"  Detections before/after   : {len(df)} / {len(merged_df)}")
    if metrics:
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
    print(f"  predictions  -> {out_pred}")
    if not args.no_video:
        print(f"  annotated mp4-> {out_video}")
    print(f"  metrics json -> {out_json}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
