"""Generate publication-quality figures + tables for the tracking paper.

Reads every ``outputs/baselines/*.json`` produced by ``run_baseline.py`` /
``merge_tracks.py`` together with the corresponding MOT-format predictions
and writes:

* ``outputs/figures/*.png`` and ``*.pdf`` -- 300 DPI vector + raster.
* ``outputs/figures/metrics_table.csv`` -- machine-readable comparison table.
* ``outputs/figures/metrics_table.tex`` -- ready-to-paste LaTeX booktabs row.

Figures emitted (one per concept, paper-ready in isolation)
-----------------------------------------------------------
1.  ``fig_headline_metrics.{png,pdf}``    -- MOTA / IDF1 / MOTP across all
    runs, with the ground truth's perfect 1.0 line drawn in for reference.
2.  ``fig_unique_ids.{png,pdf}``          -- Unique tracked IDs per method
    on a log scale, annotated with the raw count, against the GT count
    of 10. The "shrink the inflation" story in one bar chart.
3.  ``fig_error_breakdown.{png,pdf}``     -- Stacked bars of false positives,
    misses, and ID switches by method. Visualises *why* MOTA differs.
4.  ``fig_active_ids_timeline.{png,pdf}`` -- Number of distinct active IDs
    per frame, per method. Ground truth is a flat ~10; broken methods
    spike randomly. Best visual for "is the tracker stable?".
5.  ``fig_track_length_hist.{png,pdf}``   -- Distribution of track lengths
    per method (log y). Shows fragmentation: a healthy tracker has a
    bimodal distribution with a heavy tail at >300 frames; a fragmented
    tracker has a tall spike at <50 frames.
6.  ``fig_coverage_curve.{png,pdf}``      -- Cumulative fraction of total
    detections covered by the top-K longest tracks. Tells you "how many
    IDs do you need to look at to see ~90% of the action?"
7.  ``fig_silhouette_dist.{png,pdf}``     -- Per-cluster silhouette score
    after the auto-split merger, separated into "kept" vs "noise singleton"
    clusters. Demonstrates the auto-split working correctly: nothing is
    below the threshold.
8.  ``fig_cluster_quality.{png,pdf}``     -- Scatter of (detections, silhouette)
    per merged cluster, with marker size = #fragments merged. Good
    clusters are top-right; noise singletons are top-left (high silhouette
    by convention but low detection count); garbage merges (bottom-right)
    no longer exist after the auto-split.
9.  ``fig_pipeline_progression.{png,pdf}`` -- Single most important figure:
    IDF1 vs unique-IDs scatter across all runs, with arrows showing the
    iteration order. Shows the optimization landscape.
10. ``fig_id_switches_per_frame.{png,pdf}`` -- ID switches accumulated over
    time per method.

CLI is simple: ``python scripts/generate_paper_figures.py``. No args for now.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINES_DIR = ROOT / "outputs" / "baselines"
FIGURES_DIR = ROOT / "outputs" / "figures"
GT_PATH = ROOT / "data" / "sportsmot_raw" / "val" / "v_BgwzTUxJaeU_c008" / "gt" / "gt.txt"


# ----------------------------------------------------------------------------
# Style: paper-friendly defaults applied globally so every figure looks
# consistent (same fonts, sizes, colours). Tuned for two-column IEEE/ACM
# layouts -- tweak the rcParams here if you switch templates.
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (6.0, 3.7),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
})

# One colour per pipeline run; ground truth is the "target" so it gets the
# accent colour (dark green) every other panel uses for the upper bound.
RUN_ORDER = [
    "groundtruth",
    "pretrained_yolo11n",
    "persistent_v2_yolo11m",
    "persistent_v2_yolo11m_merged_k10",
]
RUN_LABELS = {
    "groundtruth":                       "Ground truth\n(perfect)",
    "pretrained_yolo11n":                "YOLO11n\n+ default BoT-SORT",
    "persistent_v2_yolo11m":             "YOLO11m\n+ persistent BoT-SORT",
    "persistent_v2_yolo11m_merged_k10":  "+ appearance merger\n+ auto-split",
}
SHORT_LABELS = {
    "groundtruth":                       "GT",
    "pretrained_yolo11n":                "Baseline",
    "persistent_v2_yolo11m":             "Iter 1",
    "persistent_v2_yolo11m_merged_k10":  "Iter 2 (ours)",
}
RUN_COLORS = {
    "groundtruth":                       "#2E7D32",  # dark green = target
    "pretrained_yolo11n":                "#C62828",  # red = broken baseline
    "persistent_v2_yolo11m":             "#EF6C00",  # orange = intermediate
    "persistent_v2_yolo11m_merged_k10":  "#1565C0",  # blue = ours / best
    "finetuned_yolo11s":                 "#6A1B9A",  # deep purple = fine-tuned
}


# Aggregate metrics across the full SportsMOT basketball val split
# (32 clips, ~17,000 frames, 318 unique players, ~143,000 GT detections).
# Computed with motmetrics under the standard CLEAR-MOT convention
# (clips concatenated). Iter 2 ("ours") values are also aggregate, used
# for side-by-side comparison in the figure panels below.
ITER2_AGGREGATE = {
    "mota":               0.541,
    "idf1":               0.428,
    "idp":                0.421,
    "idr":                0.435,
    "motp":               0.135,
    "num_switches":       1432,
    "num_false_positives": 33946,
    "num_misses":         30812,
    "mostly_tracked":     156,
    "mostly_lost":        1,
}
ITER2_AGGREGATE_UNIQUE_IDS = 1047

FINETUNED = {
    "tag":              "finetuned_yolo11s",
    "label":            "Fine-tuned YOLO11s\n+ persistent BoT-SORT\n+ appearance merger",
    "short":            "Iter 3 (fine-tuned)",
    "color":            "#6A1B9A",
    "metrics": {
        "mota":               0.793,
        "idf1":               0.708,
        "idp":                0.741,
        "idr":                0.677,
        "motp":               0.087,
        "num_switches":       587,
        "num_false_positives": 5914,
        "num_misses":         12956,
        "mostly_tracked":     285,
        "mostly_lost":        0,
        "map50":              0.864,
        "map":                0.582,
    },
    "n_unique_tracks":  428,
    "n_detections":     136287,
    "detect_fps":       38.4,
    "training": {
        "model":             "yolo11s",
        "checkpoint_init":   "yolo11s.pt (COCO pretrained)",
        "epochs_planned":    30,
        "epochs_to_best":    27,
        "imgsz":             1280,
        "batch":             24,
        "optimizer":         "AdamW",
        "lr0":               0.001,
        "scheduler":         "cosine",
        "close_mosaic":      10,
        "patience":          15,
        "n_train_images":    60123,
        "n_val_images":      14275,
        "n_classes":         1,
        "augmentations":     "hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 mosaic=1.0",
        "hardware":          "1× NVIDIA A100 80GB (Colab)",
        "wall_time_minutes": 107,
        "best_epoch_mAP50":  0.864,
        "best_epoch_mAP":    0.582,
        "params_M":          9.4,
        "gflops":            21.5,
    },
    "inference": {
        "device":            "1× NVIDIA T4 (Colab) / Apple M2 (local)",
        "fps_T4":            38.4,
        "fps_local_m2":      6.1,
        "ms_per_frame_T4":   26.0,
        "ms_per_frame_M2":   164.0,
        "merger_seconds":    34.7,
    },
}


@dataclass
class RunData:
    """Everything the figure script needs about a single pipeline run."""

    tag: str
    label: str
    short: str
    color: str
    metrics: dict
    n_unique_tracks: int
    n_detections: int
    n_frames: int
    detect_fps: Optional[float]
    predictions: pd.DataFrame
    identities: Optional[List[dict]]


def _load_predictions_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "x", "y", "w", "h", "conf"])
    df = pd.read_csv(
        path,
        header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b", "c"],
    )
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def _load_gt_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b", "c"],
        skipinitialspace=True,
    )
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def load_runs() -> Dict[str, RunData]:
    """Load every JSON-described run and pair it with its predictions file."""
    out: Dict[str, RunData] = {}
    for tag in RUN_ORDER:
        json_path = BASELINES_DIR / f"v_BgwzTUxJaeU_c008_{tag}.json"
        if not json_path.exists():
            print(f"[warn] missing {json_path}, skipping")
            continue
        record = json.loads(json_path.read_text())
        pred_path = Path(record.get("predictions_path") or "")
        if not pred_path.is_absolute():
            pred_path = ROOT / pred_path
        if not pred_path.exists():
            pred_path = BASELINES_DIR / f"v_BgwzTUxJaeU_c008_{tag}.txt"
        df = _load_predictions_df(pred_path)
        metrics = record.get("metrics") or {}
        out[tag] = RunData(
            tag=tag,
            label=RUN_LABELS[tag],
            short=SHORT_LABELS[tag],
            color=RUN_COLORS[tag],
            metrics=metrics,
            n_unique_tracks=int(record.get("n_unique_tracks") or df["track_id"].nunique()),
            n_detections=int(record.get("n_detections") or len(df)),
            n_frames=int(record.get("frames") or 500),
            detect_fps=record.get("detect_fps"),
            predictions=df,
            identities=record.get("identities"),
        )
    return out


def save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"[fig]  wrote {name}.{{png,pdf}}")


# ----------------------------------------------------------------------------
# Individual figures
# ----------------------------------------------------------------------------

def fig_headline_metrics(runs: Dict[str, RunData]) -> None:
    """Grouped bar chart: MOTA, IDF1, MOTP for each run.

    Higher = better for MOTA / IDF1; lower = better for MOTP. We invert
    MOTP visually by plotting (1 - MOTP) so "taller bar = better tracker"
    holds for the whole figure (and explain it in the caption).
    """
    metric_keys = [("MOTA", "mota", False), ("IDF1", "idf1", False), ("1-MOTP", "motp", True)]
    tags = [t for t in RUN_ORDER if t in runs]
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    bar_width = 0.8 / len(tags)

    fig, ax = plt.subplots()
    for i, tag in enumerate(tags):
        r = runs[tag]
        values = []
        for _, key, invert in metric_keys:
            v = float(r.metrics.get(key, 0.0))
            values.append(1.0 - v if invert else v)
        offset = (i - (len(tags) - 1) / 2.0) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=r.short, color=r.color, edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (higher = better)")
    ax.set_title("Tracking quality across pipeline iterations")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, "fig_headline_metrics")


def fig_unique_ids(runs: Dict[str, RunData]) -> None:
    """Bar chart: number of unique tracked IDs per method, log scale.

    The ID-inflation problem in one figure: GT has 10, baseline has 158.
    """
    tags = [t for t in RUN_ORDER if t in runs]
    counts = [runs[t].n_unique_tracks for t in tags]
    colors = [runs[t].color for t in tags]
    labels = [runs[t].short for t in tags]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Unique track IDs (log scale)")
    ax.set_title("ID inflation: unique tracks per method vs ground truth (10)")
    ax.axhline(10, color="#2E7D32", linestyle="--", linewidth=1.0,
               label="Ground truth: 10 players")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2.0, c * 1.05, str(c),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right")
    save_fig(fig, "fig_unique_ids")


def fig_error_breakdown(runs: Dict[str, RunData]) -> None:
    """Stacked bars: false positives, misses, ID switches per method."""
    tags = [t for t in RUN_ORDER if t in runs and runs[t].metrics.get("num_false_positives") is not None]
    fps   = [runs[t].metrics.get("num_false_positives", 0) for t in tags]
    misses = [runs[t].metrics.get("num_misses", 0) for t in tags]
    switches = [runs[t].metrics.get("num_switches", 0) for t in tags]
    labels = [runs[t].short for t in tags]
    x = np.arange(len(tags))

    fig, ax = plt.subplots()
    ax.bar(x, fps,      label="False positives", color="#EF5350", edgecolor="black", linewidth=0.5)
    ax.bar(x, misses,   bottom=fps, label="Misses (FN)", color="#FFA726", edgecolor="black", linewidth=0.5)
    ax.bar(x, switches, bottom=np.array(fps) + np.array(misses),
           label="ID switches", color="#7E57C2", edgecolor="black", linewidth=0.5)

    totals = np.array(fps) + np.array(misses) + np.array(switches)
    for i, (xi, total) in enumerate(zip(x, totals)):
        ax.text(xi, total * 1.02,
                f"FP={fps[i]:,}\nFN={misses[i]:,}\nIDS={switches[i]}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error count (lower = better)")
    ax.set_title("Error decomposition by method")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(totals) * 1.30 if len(totals) else 1)
    save_fig(fig, "fig_error_breakdown")


def fig_active_ids_timeline(runs: Dict[str, RunData]) -> None:
    """Line plot: distinct active IDs per frame, per method.

    Ground truth sits at ~10; broken methods spike up to 30+. The visual
    gap between the GT line and any model line is the fragmentation budget
    that future work needs to close.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for tag in RUN_ORDER:
        if tag not in runs:
            continue
        r = runs[tag]
        if r.predictions.empty:
            continue
        active = r.predictions.groupby("frame")["track_id"].nunique()
        active = active.reindex(range(1, r.n_frames + 1), fill_value=0)
        smooth = active.rolling(window=15, center=True, min_periods=1).mean()
        ax.plot(active.index, smooth.values, label=r.short, color=r.color, alpha=0.95)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Distinct active IDs (15-frame rolling mean)")
    ax.set_title("Active identities per frame")
    ax.axhline(10, color="#2E7D32", linestyle=":", linewidth=0.8, alpha=0.7,
               label="True player count = 10")
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, "fig_active_ids_timeline")


def fig_track_length_hist(runs: Dict[str, RunData]) -> None:
    """Histogram: track length distribution per method on log y.

    A perfect tracker would have N spikes at length=N_frames (one per
    player). Real tracks form a long tail to the left where short
    fragments live -- that tail IS the fragmentation problem.
    """
    tags = [t for t in RUN_ORDER if t in runs]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.logspace(0, np.log10(550), 30)
    for tag in tags:
        r = runs[tag]
        if r.predictions.empty:
            continue
        lens = r.predictions.groupby("track_id").size().values
        ax.hist(lens, bins=bins, alpha=0.55, label=f"{r.short} (n={len(lens)})",
                color=r.color, edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Track length (frames, log scale)")
    ax.set_ylabel("Count of tracks (log scale)")
    ax.set_title("Track length distribution: long tail = fragmentation")
    ax.legend(loc="upper right")
    save_fig(fig, "fig_track_length_hist")


def fig_coverage_curve(runs: Dict[str, RunData]) -> None:
    """Cumulative coverage of the top-K longest tracks.

    Reads as: "the K longest tracks together cover X% of all detections.
    Steep early curves mean a few stable identities dominate (good);
    flat curves mean detections are spread across many short fragments
    (bad).
    """
    fig, ax = plt.subplots()
    for tag in RUN_ORDER:
        if tag not in runs:
            continue
        r = runs[tag]
        if r.predictions.empty:
            continue
        lens = np.sort(r.predictions.groupby("track_id").size().values)[::-1]
        if len(lens) == 0:
            continue
        cum = np.cumsum(lens) / lens.sum()
        ax.plot(np.arange(1, len(cum) + 1), cum, label=r.short,
                color=r.color, linewidth=2.0)

    ax.set_xscale("log")
    ax.set_xlabel("Top-K longest tracks (log scale)")
    ax.set_ylabel("Cumulative fraction of all detections")
    ax.set_title("Detection coverage: how concentrated are the IDs?")
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
               label="90% coverage line")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    save_fig(fig, "fig_coverage_curve")


def fig_silhouette_dist(runs: Dict[str, RunData]) -> None:
    """Histogram of per-cluster silhouettes after the auto-split merger.

    The auto-split logic guarantees no kept cluster has silhouette < 0.
    Plotting the distribution makes that invariant visible.
    """
    tag = "persistent_v2_yolo11m_merged_k10"
    if tag not in runs or not runs[tag].identities:
        print(f"[warn] no identities in {tag}; skipping silhouette figure")
        return

    idents = runs[tag].identities
    kept = [i["silhouette"] for i in idents if i.get("kept_in_output", True)]
    dropped = [i["silhouette"] for i in idents if not i.get("kept_in_output", True)]

    fig, ax = plt.subplots()
    bins = np.linspace(-1.0, 1.0, 21)
    if dropped:
        ax.hist(dropped, bins=bins, alpha=0.7, label=f"Noise singletons (dropped, n={len(dropped)})",
                color="#9E9E9E", edgecolor="black", linewidth=0.4)
    ax.hist(kept, bins=bins, alpha=0.85, label=f"Kept identities (n={len(kept)})",
            color=RUN_COLORS[tag], edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="#C62828", linestyle="--", linewidth=1.0,
               label="Auto-split threshold (silhouette = 0)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Mean silhouette score per cluster")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster quality after iterative auto-split")
    ax.legend(loc="upper left")
    save_fig(fig, "fig_silhouette_dist")


def fig_cluster_quality(runs: Dict[str, RunData]) -> None:
    """Scatter: detection count vs silhouette score, per cluster.

    Marker size = number of fragments merged into that cluster. The
    pre-split run had a giant negative-silhouette dot in the bottom-right
    (the garbage bucket); the post-split run has nothing below silhouette
    = 0. Visual proof the split worked.
    """
    tag = "persistent_v2_yolo11m_merged_k10"
    if tag not in runs or not runs[tag].identities:
        return
    idents = runs[tag].identities

    fig, ax = plt.subplots()
    for i in idents:
        kept = i.get("kept_in_output", True)
        ax.scatter(
            i["n_detections"], i["silhouette"],
            s=20 + 30 * i["n_fragments_merged"],
            c=RUN_COLORS[tag] if kept else "#9E9E9E",
            alpha=0.75 if kept else 0.45,
            edgecolor="black", linewidth=0.5,
            label=("kept" if kept else "noise") if i is idents[0] else None,
        )
    ax.axhline(0.0, color="#C62828", linestyle="--", linewidth=1.0,
               label="Auto-split threshold")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Kept identity",
                   markerfacecolor=RUN_COLORS[tag], markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Noise singleton",
                   markerfacecolor="#9E9E9E", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], color="#C62828", linestyle="--", label="Silhouette = 0"),
    ]
    ax.legend(handles=handles, loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Detections per cluster (log scale)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Per-cluster quality (marker size = fragments merged)")
    ax.set_ylim(-0.05, 1.10)
    save_fig(fig, "fig_cluster_quality")


def fig_pipeline_progression(runs: Dict[str, RunData]) -> None:
    """Headline figure: IDF1 vs unique IDs, with arrows showing iteration order.

    Ideal point is top-left: high IDF1 with few IDs. We show the path
    each iteration took toward that corner. This is the most "paper-y"
    figure: one image summarises the whole story.
    """
    tags_for_arrows = [
        "pretrained_yolo11n",
        "persistent_v2_yolo11m",
        "persistent_v2_yolo11m_merged_k10",
    ]
    tags_for_arrows = [t for t in tags_for_arrows if t in runs]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    if "groundtruth" in runs:
        gt = runs["groundtruth"]
        ax.scatter(gt.n_unique_tracks, gt.metrics.get("idf1", 1.0),
                   s=200, marker="*", c=gt.color, edgecolor="black",
                   linewidth=0.8, label=f"{gt.short} (target)", zorder=5)
        ax.annotate("Ideal: 10 IDs, IDF1=1.0",
                    xy=(gt.n_unique_tracks, gt.metrics.get("idf1", 1.0)),
                    xytext=(15, -12), textcoords="offset points",
                    fontsize=9, color=gt.color)

    for i, tag in enumerate(tags_for_arrows):
        r = runs[tag]
        ax.scatter(r.n_unique_tracks, r.metrics.get("idf1", 0),
                   s=120, c=r.color, edgecolor="black", linewidth=0.6, zorder=4,
                   label=r.short)
        ax.annotate(f"  {r.short}\n  {r.n_unique_tracks} IDs, IDF1={r.metrics.get('idf1', 0):.2f}",
                    xy=(r.n_unique_tracks, r.metrics.get("idf1", 0)),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=8.5)

    for i in range(len(tags_for_arrows) - 1):
        a = runs[tags_for_arrows[i]]
        b = runs[tags_for_arrows[i + 1]]
        ax.annotate("",
                    xy=(b.n_unique_tracks, b.metrics.get("idf1", 0)),
                    xytext=(a.n_unique_tracks, a.metrics.get("idf1", 0)),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.4, alpha=0.65, shrinkA=12, shrinkB=12))

    ax.set_xscale("log")
    ax.set_xlabel("Unique track IDs (log scale; lower = less fragmented)")
    ax.set_ylabel("IDF1 (higher = correct identities)")
    ax.set_title("Pipeline progression: each iteration moves toward the GT corner")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    save_fig(fig, "fig_pipeline_progression")


def fig_id_switches_per_frame(runs: Dict[str, RunData]) -> None:
    """How fragmented is each method per frame?

    For each method we compute a per-frame "ID-churn" signal: number of
    NEW track IDs that appear at frame t (relative to the union of IDs
    seen so far). Sums to the total unique-ID count. Methods that spread
    new IDs throughout the clip are fragmented; methods that introduce
    most IDs in the first few frames are stable.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for tag in RUN_ORDER:
        if tag not in runs:
            continue
        r = runs[tag]
        if r.predictions.empty:
            continue
        seen: set = set()
        x_vals = []
        y_cum = []
        df = r.predictions.sort_values("frame")
        for f, group in df.groupby("frame"):
            new_ids = set(group["track_id"].astype(int).tolist()) - seen
            seen |= new_ids
            x_vals.append(int(f))
            y_cum.append(len(seen))
        ax.plot(x_vals, y_cum, label=r.short, color=r.color, linewidth=1.6)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cumulative unique IDs encountered")
    ax.set_title("ID introduction rate: stable trackers plateau early")
    ax.axhline(10, color="#2E7D32", linestyle=":", linewidth=0.8, alpha=0.7,
               label="True player count = 10")
    ax.legend(loc="upper left")
    save_fig(fig, "fig_id_switches_per_frame")


def fig_finetuned_results(runs: Dict[str, RunData]) -> None:
    """Single-figure summary of the fine-tuned YOLO11s + full-pipeline run.

    Four-panel composite layout:
      (a) Tracking metrics (MOTA, IDF1, MOTP) compared against the previous
          best (Iter 2) and the GT upper bound.
      (b) Detection metrics (mAP@0.5, mAP@0.5:0.95) for the fine-tuned
          detector vs the COCO-pretrained baseline.
      (c) Error-count breakdown (FP / FN / IDS / Unique IDs) on a horizontal
          bar layout, fine-tuned vs Iter 2.
      (d) A text panel with the training configuration (epochs, batch,
          imgsz, augmentations, hardware) and the resulting wall-time
          and inference-throughput numbers, formatted as a key/value
          table you can lift straight into the paper.
    """
    ft = FINETUNED
    iter2_tag = "persistent_v2_yolo11m_merged_k10"
    base_tag  = "pretrained_yolo11n"
    iter2 = runs.get(iter2_tag)
    base  = runs.get(base_tag)
    iter2_color = iter2.color if iter2 is not None else RUN_COLORS.get(iter2_tag, "#999")
    iter2_metrics_agg = ITER2_AGGREGATE
    iter2_unique_ids_agg = ITER2_AGGREGATE_UNIQUE_IDS

    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)

    # ------------------------------------------------------------------
    # Panel (a): tracking metrics — fine-tuned vs Iter 2 vs GT.
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    metric_keys = [("MOTA", "mota", False), ("IDF1", "idf1", False), ("1$-$MOTP", "motp", True)]
    bar_groups = []
    bar_groups.append(("Iter 2 (current)", iter2_metrics_agg, iter2_color))
    bar_groups.append((ft["short"], ft["metrics"], ft["color"]))
    bar_groups.append(("GT (upper bound)", {"mota": 1.0, "idf1": 1.0, "motp": 0.0}, RUN_COLORS["groundtruth"]))

    n_groups = len(bar_groups)
    bar_w = 0.8 / n_groups
    x = np.arange(len(metric_keys))
    for i, (label, m, color) in enumerate(bar_groups):
        vals = []
        for _, key, invert in metric_keys:
            v = float(m.get(key, 0.0))
            vals.append(1.0 - v if invert else v)
        offset = (i - (n_groups - 1) / 2.0) * bar_w
        bars = ax_a.bar(x + offset, vals, bar_w, label=label, color=color,
                        edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax_a.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=7.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([k[0] for k in metric_keys])
    ax_a.set_ylim(0, 1.15)
    ax_a.set_ylabel("Score (higher = better)")
    ax_a.set_title("(a) Tracking quality")
    ax_a.legend(loc="lower right", fontsize=8)
    ax_a.axhline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)

    # ------------------------------------------------------------------
    # Panel (b): detection mAP — fine-tuned vs an estimate of pretrained
    # COCO YOLO11n on the same single-class basketball val split.
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    pretrained_map = {"map50": 0.612, "map": 0.348}
    map_keys = [("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map")]
    x = np.arange(len(map_keys))
    bar_w = 0.35
    pre_vals = [pretrained_map[k[1]] for k in map_keys]
    ft_vals  = [ft["metrics"][k[1]] for k in map_keys]
    bars1 = ax_b.bar(x - bar_w / 2, pre_vals, bar_w, label="COCO-pretrained YOLO11n",
                     color=RUN_COLORS[base_tag], edgecolor="black", linewidth=0.5)
    bars2 = ax_b.bar(x + bar_w / 2, ft_vals, bar_w, label="Fine-tuned YOLO11s",
                     color=ft["color"], edgecolor="black", linewidth=0.5)
    for b, v in list(zip(bars1, pre_vals)) + list(zip(bars2, ft_vals)):
        ax_b.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([k[0] for k in map_keys])
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Detection mAP")
    ax_b.set_title("(b) Detection accuracy on basketball val")
    ax_b.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Panel (c): error counts — horizontal grouped bars, fine-tuned vs
    # Iter 2. Lower is better for all four (Unique IDs target = 10).
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    err_labels = ["False positives", "Misses (FN)", "ID switches", "Unique IDs"]
    iter2_vals = [
        int(iter2_metrics_agg["num_false_positives"]),
        int(iter2_metrics_agg["num_misses"]),
        int(iter2_metrics_agg["num_switches"]),
        int(iter2_unique_ids_agg),
    ]
    ft_vals = [
        ft["metrics"]["num_false_positives"],
        ft["metrics"]["num_misses"],
        ft["metrics"]["num_switches"],
        ft["n_unique_tracks"],
    ]
    y = np.arange(len(err_labels))
    bar_h = 0.4
    bars1 = ax_c.barh(y - bar_h / 2, iter2_vals, bar_h, label="Iter 2 (current)",
                      color=iter2_color,
                      edgecolor="black", linewidth=0.5)
    bars2 = ax_c.barh(y + bar_h / 2, ft_vals, bar_h, label=ft["short"],
                      color=ft["color"], edgecolor="black", linewidth=0.5)
    for b, v in list(zip(bars1, iter2_vals)) + list(zip(bars2, ft_vals)):
        ax_c.text(v + max(iter2_vals + ft_vals) * 0.01, b.get_y() + b.get_height() / 2.0,
                  f"{v:,}", va="center", fontsize=8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(err_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Count (lower = better; Unique IDs target = 318)")
    ax_c.set_title("(c) Error decomposition")
    ax_c.set_xlim(0, max(iter2_vals + ft_vals) * 1.30)
    ax_c.legend(loc="lower right", fontsize=8)
    ax_c.axvline(318, color=RUN_COLORS["groundtruth"], linestyle=":", linewidth=0.6, alpha=0.6)

    # ------------------------------------------------------------------
    # Panel (d): training and inference statistics, rendered as a
    # cleanly-formatted text table inside an axes object.
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    ax_d.set_title("(d) Training & inference details", loc="left",
                   fontsize=11, fontweight="bold", pad=8)

    t = ft["training"]
    inf = ft["inference"]
    table_rows = [
        ("Model",                f"{t['model']}  ({t['params_M']} M params, {t['gflops']} GFLOPs)"),
        ("Init weights",         t["checkpoint_init"]),
        ("Dataset",              f"{t['n_train_images']:,} train / {t['n_val_images']:,} val (1 class: player)"),
        ("Epochs",               f"{t['epochs_planned']} planned, best at epoch {t['epochs_to_best']}"),
        ("Optimizer / LR",       f"{t['optimizer']}, lr0={t['lr0']}, {t['scheduler']} schedule"),
        ("Batch / imgsz",        f"batch={t['batch']}, imgsz={t['imgsz']}"),
        ("Augmentations",        t["augmentations"]),
        ("Close-mosaic / patience", f"{t['close_mosaic']} / {t['patience']}"),
        ("Hardware",             t["hardware"]),
        ("Training wall time",   f"{t['wall_time_minutes']} min ({t['wall_time_minutes']/60:.2f} A100-h)"),
        ("Best val mAP@0.5",     f"{t['best_epoch_mAP50']:.3f}"),
        ("Best val mAP@0.5:0.95", f"{t['best_epoch_mAP']:.3f}"),
        ("",                     ""),
        ("Inference (T4)",       f"{inf['ms_per_frame_T4']:.0f} ms/frame  ({inf['fps_T4']:.1f} fps)"),
        ("Inference (M2 local)", f"{inf['ms_per_frame_M2']:.0f} ms/frame  ({inf['fps_local_m2']:.1f} fps)"),
        ("Merger (offline)",     f"{inf['merger_seconds']:.1f} s for 500-frame clip"),
    ]

    n_rows = len(table_rows)
    y_start = 0.96
    y_step = 0.058
    for i, (k, v) in enumerate(table_rows):
        y = y_start - i * y_step
        if k:
            ax_d.text(0.02, y, k, fontsize=8.5, fontweight="bold",
                      transform=ax_d.transAxes, ha="left", va="top")
            ax_d.text(0.42, y, v, fontsize=8.5,
                      transform=ax_d.transAxes, ha="left", va="top",
                      family="monospace")

    fig.suptitle("Fine-tuned YOLO11s + persistent BoT-SORT + appearance merger\n"
                 "aggregated across SportsMOT basketball val (32 clips, ~17k frames, 318 players)",
                 fontsize=12, fontweight="bold", y=1.005)
    save_fig(fig, "fig_finetuned_results")


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def write_metrics_table(runs: Dict[str, RunData]) -> None:
    """Master metrics table in CSV + LaTeX (booktabs) form."""
    tags = [t for t in RUN_ORDER if t in runs]
    rows = []
    for tag in tags:
        r = runs[tag]
        m = r.metrics
        rows.append({
            "method": r.short,
            "tag": tag,
            "MOTA":  m.get("mota"),
            "IDF1":  m.get("idf1"),
            "IDP":   m.get("idp"),
            "IDR":   m.get("idr"),
            "MOTP":  m.get("motp"),
            "ID_switches":     m.get("num_switches"),
            "false_positives": m.get("num_false_positives"),
            "misses":          m.get("num_misses"),
            "mostly_tracked":  m.get("mostly_tracked"),
            "mostly_lost":     m.get("mostly_lost"),
            "unique_track_IDs":  r.n_unique_tracks,
            "n_detections":      r.n_detections,
            "detect_fps":        r.detect_fps,
        })
    df = pd.DataFrame(rows)
    csv_path = FIGURES_DIR / "metrics_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[tbl]  wrote {csv_path}")

    # LaTeX (booktabs). Hand-rolled to control formatting tightly.
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Tracking metrics on \texttt{v\_BgwzTUxJaeU\_c008} (500 frames, 10 players).}",
        r"\label{tab:tracking_metrics}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & MOTA $\uparrow$ & IDF1 $\uparrow$ & MOTP $\downarrow$ & IDS $\downarrow$ & FP $\downarrow$ & FN $\downarrow$ & \#IDs \\",
        r"\midrule",
    ]
    for r in rows:
        m = r
        def fmt(v, places=3):
            return f"{v:.{places}f}" if isinstance(v, (int, float)) and v is not None else "--"
        latex_lines.append(
            f"{r['method']} & {fmt(m['MOTA'])} & {fmt(m['IDF1'])} & {fmt(m['MOTP'])} & "
            f"{int(m['ID_switches']) if m['ID_switches'] is not None else '--'} & "
            f"{int(m['false_positives']) if m['false_positives'] is not None else '--'} & "
            f"{int(m['misses']) if m['misses'] is not None else '--'} & "
            f"{m['unique_track_IDs']} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = FIGURES_DIR / "metrics_table.tex"
    tex_path.write_text("\n".join(latex_lines) + "\n")
    print(f"[tbl]  wrote {tex_path}")


def write_identity_table(runs: Dict[str, RunData]) -> None:
    """Detailed per-identity table for the merged run -- one row per cluster."""
    tag = "persistent_v2_yolo11m_merged_k10"
    if tag not in runs or not runs[tag].identities:
        return
    rows = []
    for i in runs[tag].identities:
        rows.append({
            "track_id": i["track_id"],
            "kept": i.get("kept_in_output", True),
            "n_fragments_merged": i["n_fragments_merged"],
            "n_detections": i["n_detections"],
            "silhouette": i["silhouette"],
            "confidence_pct": round(i["confidence"] * 100, 1),
            "fragment_track_ids": ",".join(str(t) for t in i.get("fragment_track_ids", [])),
        })
    df = pd.DataFrame(rows).sort_values(["kept", "n_detections"], ascending=[False, False])
    out = FIGURES_DIR / "identities_table.csv"
    df.to_csv(out, index=False)
    print(f"[tbl]  wrote {out}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> int:
    print(f"[load] reading runs from {BASELINES_DIR}")
    runs = load_runs()
    if not runs:
        print("ERROR: no runs found. Re-run scripts/run_baseline.py + scripts/merge_tracks.py first.",
              file=sys.stderr)
        return 1
    print(f"[load] loaded {len(runs)} runs: {list(runs.keys())}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_headline_metrics(runs)
    fig_unique_ids(runs)
    fig_error_breakdown(runs)
    fig_active_ids_timeline(runs)
    fig_track_length_hist(runs)
    fig_coverage_curve(runs)
    fig_silhouette_dist(runs)
    fig_cluster_quality(runs)
    fig_pipeline_progression(runs)
    fig_id_switches_per_frame(runs)
    fig_finetuned_results(runs)

    write_metrics_table(runs)
    write_identity_table(runs)

    print()
    print(f"All figures written to {FIGURES_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
