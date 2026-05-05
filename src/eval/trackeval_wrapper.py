"""Compute tracking metrics (MOTA, MOTP, IDF1, etc.) on SportsMOT-format
ground truth and prediction files using the `motmetrics` library.

We use motmetrics rather than the official TrackEval toolkit by default
because it has a much lighter install footprint and the metrics it produces
are sufficient for iterating on the detector and tracker. When you want
proper HOTA numbers for the writeup, swap in TrackEval (a stub for that is
left at the bottom of this file).

Both ground-truth and prediction files are expected in the standard
MOT-Challenge text format (one detection per line):

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Example:
    from src.eval.trackeval_wrapper import evaluate_sequence
    summary = evaluate_sequence(
        gt_path  = "data/sportsmot/dataset/val/v_xxx_c001/gt/gt.txt",
        pred_path= "outputs/v_xxx_c001.txt",
    )
    print(summary)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# motmetrics 1.4.0 still calls np.asfarray, which NumPy 2.0 removed.
# Monkey-patch a backport so the metrics work on modern NumPy installs.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]


_MOT_COLS = [
    "frame", "id", "x", "y", "w", "h", "conf", "cls", "vis", "z",
]


def _read_mot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # Pad to 10 columns if the file uses the shorter 7-column variant.
    while df.shape[1] < len(_MOT_COLS):
        df[df.shape[1]] = -1
    df.columns = _MOT_COLS[: df.shape[1]]
    return df


def evaluate_sequence(
    gt_path: str | Path,
    pred_path: str | Path,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute MOT metrics for one sequence. Returns a one-row DataFrame."""
    import motmetrics as mm

    gt = _read_mot(Path(gt_path))
    pred = _read_mot(Path(pred_path))

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt["frame"]).union(set(pred["frame"])))
    for f in frames:
        gt_f = gt[gt["frame"] == f]
        pr_f = pred[pred["frame"] == f]
        gt_ids = gt_f["id"].astype(int).tolist()
        pr_ids = pr_f["id"].astype(int).tolist()
        # motmetrics expects (x, y, w, h) and computes IoU internally.
        dists = mm.distances.iou_matrix(
            gt_f[["x", "y", "w", "h"]].values,
            pr_f[["x", "y", "w", "h"]].values,
            max_iou=1.0 - iou_threshold,
        )
        acc.update(gt_ids, pr_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "mostly_lost",
        ],
        name=Path(pred_path).stem,
    )
    return summary


def evaluate_directory(
    gt_root: str | Path,
    pred_root: str | Path,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate every sequence under `pred_root` against matching gt under `gt_root`.

    Assumes:
      pred_root/<seq>.txt
      gt_root/<seq>/gt/gt.txt
    """
    gt_root, pred_root = Path(gt_root), Path(pred_root)
    summaries = []
    for pred_file in sorted(pred_root.glob("*.txt")):
        seq = pred_file.stem
        gt_file = gt_root / seq / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"[skip] no gt for {seq} at {gt_file}")
            continue
        summaries.append(evaluate_sequence(gt_file, pred_file, iou_threshold))
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries)


# ---------------------------------------------------------------------------
# Optional: TrackEval / HOTA. Left as a stub because installing TrackEval is
# heavier than motmetrics. Uncomment + flesh out when you want the official
# HOTA numbers for the writeup.
# ---------------------------------------------------------------------------

def evaluate_with_trackeval(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "TrackEval integration is intentionally left as a stub. "
        "Install with `pip install git+https://github.com/JonathonLuiten/TrackEval.git` "
        "and follow the dataset config conventions in their docs."
    )
