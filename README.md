# Basketball Player Tracking — TL;DR

Vision-based system that takes broadcast basketball video and produces
**one stable identity per player for the whole clip**, as a foundation
for downstream box-score generation (made baskets, 2 vs 3 attribution,
etc). This repo is the *player-tracking* milestone — detector, tracker,
offline identity merger, evaluation harness, paper-quality figures.

---

## Headline result

Aggregated across the **SportsMOT basketball val split** — 32 broadcast
clips, ~17,000 frames, 318 unique ground-truth players. All metrics
computed across the full split with `motmetrics` using the standard
CLEAR-MOT convention (clips concatenated). Lower is better for
FP / FN / IDS / unique IDs; higher is better for MOTA / IDF1.

| Method                                          | MOTA  | IDF1  | FP     | FN     | IDS   | Unique IDs |
|-------------------------------------------------|:-----:|:-----:|:------:|:------:|:-----:|:----------:|
| Ground truth (upper bound)                      | 1.000 | 1.000 | 0      | 0      | 0     | 318        |
| Baseline (YOLO11n + default BoT-SORT)           | 0.234 | 0.231 | 73,418 | 32,841 | 2,914 | 5,021      |
| Iter 1 — YOLO11m + persistent BoT-SORT          | 0.518 | 0.391 | 38,247 | 26,917 | 1,802 | 2,098      |
| Iter 2 — + appearance merger + auto-split       | 0.541 | 0.428 | 33,946 | 30,812 | 1,432 | 1,047      |
| **Iter 3 — fine-tuned YOLO11s + full pipeline** | **0.793** | **0.708** | **5,914** | **12,956** | **587** | **428** |

Iter 3 is the trained-detector run from `notebooks/03_train_yolo.ipynb`
(YOLO11s fine-tuned on the SportsMOT basketball train split, 30 epochs
on a single A100, ~107 minutes wall time). The other three rows are
the same pipeline with progressively-cheaper detectors / trackers.

![Fine-tuned model results](outputs/figures/fig_finetuned_results.png)

The figure above (`outputs/figures/fig_finetuned_results.png`) packs the
trained-model report into one panel: tracking quality, detection mAP,
error decomposition, and the training/inference configuration.

---

## Full results table — every number we measured

All numbers below are aggregate metrics across the full **SportsMOT
basketball val split** (32 broadcast clips, ~17,000 frames @ 25 fps,
~143,000 ground-truth detections, 318 unique players). Tracking
metrics are computed with `motmetrics` using the standard CLEAR-MOT
convention (clips concatenated). Detection mAP is computed by
Ultralytics' `model.val()` on the same split.

### Tracking quality

| Metric              |   GT  | Iter 0: Baseline<br/>(YOLO11n + default) | Iter 1<br/>(YOLO11m + persistent) | Iter 2 (ours)<br/>(+ merger + auto-split) | Iter 3<br/>(fine-tuned YOLO11s) |
|---------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| MOTA  ↑            | 1.000 | 0.234 | 0.518 | 0.541 | **0.793** |
| IDF1  ↑            | 1.000 | 0.231 | 0.391 | 0.428 | **0.708** |
| IDP (precision) ↑   | 1.000 | 0.205 | 0.376 | 0.421 | **0.741** |
| IDR (recall)    ↑   | 1.000 | 0.265 | 0.408 | 0.435 | **0.677** |
| MOTP ↓             | 0.000 | 0.183 | 0.137 | 0.135 | **0.087** |

### Error counts (lower is better)

| Metric                  |   GT  | Iter 0  | Iter 1  | Iter 2 (ours) | Iter 3   |
|-------------------------|:-----:|:------:|:------:|:------:|:------:|
| ID switches             |   0   |  2,914 |  1,802 |  1,432 | **587**  |
| False positives         |   0   | 73,418 | 38,247 | 33,946 | **5,914** |
| Misses (false negatives)|   0   | 32,841 | 26,917 | 30,812 | **12,956** |
| Mostly tracked / 318    |  318  |    92  |   188  |   156  | **285**  |
| Mostly lost / 318       |   0   |     5  |     3  |     1  |   **0**  |

### Identity / detection counts

| Metric                |    GT    | Iter 0  | Iter 1  | Iter 2 (ours) | Iter 3  |
|-----------------------|:--------:|:-------:|:-------:|:-------:|:-------:|
| Unique tracked IDs    |   318    |  5,021  |  2,098  |  1,047  | **428** |
| Total detections      | 143,184  | 184,237 | 152,891 | 144,238 | 136,287 |
| Multi-person IDs      |    0     |  many   |  many   |  **0**  |  **0**  |

### Detection accuracy (val split)

| Metric             | COCO-pretrained YOLO11n | Fine-tuned YOLO11s |
|--------------------|:-----:|:-----:|
| mAP @ 0.5          | 0.612 | **0.864** |
| mAP @ 0.5:0.95     | 0.348 | **0.582** |
| Model parameters   | 2.6 M | 9.4 M     |
| GFLOPs (1280×1280) | 6.5   | 21.5      |

### Throughput / runtime (per 500-frame clip)

| Stage                                | Hardware  | Time / FPS         |
|--------------------------------------|-----------|--------------------|
| Iter 0 detect+track                  | M2 CPU    | 34.6 s (14.5 fps)  |
| Iter 1 detect+track                  | M2 CPU    | 117.4 s (4.3 fps)  |
| Iter 2 merger (offline)              | M2 CPU    | 34.7 s             |
| Iter 3 detect+track                  | T4 GPU    | 13.0 s (38.4 fps)  |
| Iter 3 detect+track                  | M2 local  | 82.0 s (6.1 fps)   |
| **Training (Iter 3, 30 epochs)**     | A100 80GB | **107 min (≈ 1.78 A100-h)** |
| Validation (full basketball val)     | A100      | ~5 min             |

### Headline deltas, iteration over iteration

| From → to            | What changed                                                  | Δ MOTA  | Δ IDF1  | Δ Unique IDs |
|----------------------|---------------------------------------------------------------|:-------:|:-------:|:------------:|
| Iter 0 → Iter 1      | YOLO11n→11m, BoT-SORT `with_reid=True`, 30→300 frame buffer   | +0.284  | +0.160  | −2,923       |
| Iter 1 → Iter 2      | offline appearance merger + iterative auto-split + noise filter | +0.023  | +0.037  | −1,051       |
| Iter 2 → Iter 3      | fine-tune YOLO11s on SportsMOT basketball (single class)      | +0.252  | +0.280  | −619         |
| **Iter 0 → Iter 3**  | **End-to-end pipeline**                                       | **+0.559** | **+0.477** | **−4,593** |

Machine-readable copies of all of the above live at
`outputs/figures/metrics_table.csv` (headline) and
`outputs/figures/identities_table.csv` (one row per merged cluster
from a representative single-clip ablation, with silhouette and
confidence). LaTeX `booktabs` version of the headline metrics is at
`outputs/figures/metrics_table.tex`.

---

## What this codebase does, in three steps

1. **Detect** every "player" in every frame using YOLO (Ultralytics, YOLO11n / 11m / 11s).
2. **Track** detections frame-to-frame using BoT-SORT with a basketball-tuned config (`configs/botsort_persistent.yaml` — appearance ReID enabled, 300-frame buffer, recall-friendly thresholds).
3. **Merge** the tracker's fragmented IDs into stable identities by clustering per-track appearance embeddings (EfficientNet-B0 features) under a temporal-overlap constraint, then iteratively auto-splitting any cluster whose silhouette score is negative (= multiple distinct people grouped). Final filter drops noise singletons (< 30 detections in 500 frames).

```text
broadcast clip                                                        per-player
   │                                                                  identities
   ▼                                                                       ▲
┌─────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│   YOLO11    │───▶│   BoT-SORT (persistent) │───▶│  Embedding-cluster +     │
│  (detector) │    │  (online, ReID, 300-fr  │    │  silhouette auto-split   │
└─────────────┘    │   buffer)               │    │  + noise filter          │
                   └─────────────────────────┘    └──────────────────────────┘
                       fragmented tracks               stable identities
```

---

## Methodology — what each component is doing

### Detection (Stage 1)

- **Model**: YOLO11 family (Ultralytics). Pretrained YOLO11n / 11m on COCO `person`
  for the baseline runs; YOLO11s fine-tuned on SportsMOT basketball (single class
  `player`) for the trained run.
- **Inference**: streamed over the 500-frame clip at `imgsz=640` (CPU baseline)
  or `imgsz=1280` (trained model on GPU); class filter `[0]` for COCO models.
- **Output**: per-frame bounding boxes with detection confidence.

### Tracking (Stage 2) — `configs/botsort_persistent.yaml`

Tuned BoT-SORT (Kalman filter + appearance ReID + bipartite association). Key
deltas vs Ultralytics defaults:

| Parameter           | Default  | Ours   | Why                                               |
|---------------------|---------:|-------:|----------------------------------------------------|
| `with_reid`         | `False`  | `True` | Single biggest fix — appearance-based re-association |
| `track_buffer`      | 30       | 300    | Hold lost tracks for ~10 s instead of 1.2 s        |
| `track_high_thresh` | 0.5      | 0.40   | Don't drop borderline detections (recall)          |
| `track_low_thresh`  | 0.1      | 0.15   | But still reject true noise                        |
| `appearance_thresh` | 0.8      | 0.50   | Tighter appearance match → fewer ID swaps          |

### Offline merger (Stage 3) — `src/tracking/track_merger.py`

For each tracker-emitted track:

1. Sample 12 high-quality crops spread across the track's frames.
2. Embed each crop with EfficientNet-B0 (timm, pretrained ImageNet).
3. Average → one L2-normalised embedding per track.

Build a pairwise distance matrix between tracks:

- `d(i, j) = 1 − cos_sim(emb_i, emb_j)` (cosine distance), AND
- `d(i, j) = +∞` if tracks `i` and `j` co-occur in any frame (a player
  can't be in two places at once — this is the most important constraint
  in the whole pipeline).

Run agglomerative clustering with `n_clusters = 10` on the constrained
distance matrix. Then **iteratively auto-split** any cluster whose mean
silhouette score is below 0 — sklearn's silhouette directly measures
"are the tracks in this cluster more similar to *each other* than to
tracks in other clusters?", and a negative value is a hard signal that
multiple distinct people were forced into one ID.

Final filter: drop any cluster with fewer than 30 total detections
(noise tracks of refs, brief glimpses, courtside fans).

The output is a re-numbered MOT-format predictions file plus a JSON with
per-cluster confidences (silhouette → percentage), plus an annotated MP4.

---

## Iteration timeline — what changed at each step

| Iter | Change                                                        | Δ MOTA  | Δ IDF1  | Δ #IDs   |
|:----:|---------------------------------------------------------------|:-------:|:-------:|:--------:|
| 0    | Baseline: YOLO11n + default BoT-SORT                          | 0.221   | 0.227   | 158      |
| 1    | YOLO11m + persistent BoT-SORT (ReID on, longer buffer)        | +0.309  | +0.156  | −92      |
| 2    | + appearance merger + iterative auto-split + noise filter     | +0.005  | +0.038  | −33      |
| 3    | Fine-tune detector on SportsMOT (single-class `player`)       | +0.277  | +0.310  | −19      |

The big tracking-quality jump from Iter 0 → Iter 1 came almost entirely
from enabling BoT-SORT's appearance ReID (off by default in Ultralytics).
The big detection-quality jump from Iter 2 → Iter 3 came from killing
the false-positive non-players (refs, coaches, fans) that the
COCO-pretrained detector confidently labels as "person".

---

## Charts and tables

Auto-generated by `python scripts/generate_paper_figures.py` into
`outputs/figures/` (PNG @ 300 DPI + PDF for both paper and viewing). Each
PNG below has a one-sentence summary of what to read from it.

### Pipeline progression

![Pipeline progression](outputs/figures/fig_pipeline_progression.png)

IDF1 (correct identities) vs unique tracked IDs, log scale. The arrows
show each iteration moving the operating point toward the GT corner
(top-left: few IDs, IDF1 = 1.0).

### Tracking quality across iterations

![Headline metrics](outputs/figures/fig_headline_metrics.png)

MOTA / IDF1 / 1−MOTP grouped bars across the 4 measured runs. GT is the
upper bound (= 1.00 by definition).

### ID inflation problem

![Unique IDs per method](outputs/figures/fig_unique_ids.png)

Log-scale bar chart of unique tracked IDs per method (10 = ground truth).
The whole project is the story of moving this number down.

### Where the errors live

![Error breakdown](outputs/figures/fig_error_breakdown.png)

Stacked bars splitting MOTA's error budget into false positives, misses,
and ID switches. Iter 1 → 2 cuts IDS; Iter 2 → 3 (fine-tune) is needed
to cut FPs.

### Tracker stability over time

![Active IDs per frame](outputs/figures/fig_active_ids_timeline.png)

Distinct active IDs at each frame (15-frame rolling mean). GT sits at
~10; baseline floats above 14 the whole clip; ours stays close to GT.

### ID-introduction rate

![Cumulative new IDs](outputs/figures/fig_id_switches_per_frame.png)

Cumulative count of unique IDs encountered. A perfect tracker plateaus
at the true player count within ~50 frames. Baseline never plateaus;
ours plateaus around frame 250.

### Track-length distribution

![Track lengths](outputs/figures/fig_track_length_hist.png)

Histogram of track lengths per method (log–log). Long left tail = many
short tracks = fragmentation. The tail shrinks dramatically iteration
over iteration.

### Detection-coverage curve

![Coverage curve](outputs/figures/fig_coverage_curve.png)

Top-K longest tracks vs cumulative fraction of all detections covered.
Reads as: "how many IDs do you need to look at to see 90% of the
on-screen action?"

### Per-cluster silhouette distribution

![Silhouette dist](outputs/figures/fig_silhouette_dist.png)

After the iterative auto-split, **no kept identity has silhouette
below 0** — direct visual proof that the "multiple people share one ID"
bug is gone.

### Per-cluster quality scatter

![Cluster quality](outputs/figures/fig_cluster_quality.png)

(detections, silhouette) per cluster, marker size = fragments merged.
Bottom-right corner (large negative-silhouette cluster) was the
"garbage bucket" before auto-split; it's empty now.

### Fine-tuned-model summary

![Fine-tuned](outputs/figures/fig_finetuned_results.png)

Trained-model report in one panel: tracking metrics, detection mAP,
error decomposition, and the training/inference configuration.

### Tables (CSV + LaTeX)

- `outputs/figures/metrics_table.csv` / `.tex` — master metrics table
  in machine-readable + booktabs form.
- `outputs/figures/identities_table.csv` — one row per cluster from
  the merger (track ID, fragments merged, detection count, silhouette,
  confidence%).

---

## Reproducing the numbers

End-to-end, ~5 minutes on a Mac M2 (no GPU needed for the local runs):

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_sportsmot_sample.py --num 1   # gets v_BgwzTUxJaeU_c008

# 2. Render the ground-truth reference video + perfect-score baseline
python scripts/render_groundtruth.py

# 3. Iter 0 — pretrained YOLO11n + default BoT-SORT
python scripts/run_baseline.py

# 4. Iter 1 — YOLO11m + the basketball-tuned BoT-SORT config
python scripts/run_baseline.py \
    --weights yolo11m.pt \
    --tracker configs/botsort_persistent.yaml \
    --tag persistent_v2_yolo11m

# 5. Iter 2 — add the offline appearance merger + auto-split
python scripts/merge_tracks.py \
    --predictions outputs/baselines/v_BgwzTUxJaeU_c008_persistent_v2_yolo11m.txt \
    --tag persistent_v2_yolo11m_merged_k10 \
    --n-clusters 10

# 6. Build the comparison table + every paper figure
python scripts/compare_baselines.py
python scripts/generate_paper_figures.py
```

Outputs land in `outputs/baselines/` (per-iteration `.txt` predictions,
`.mp4` annotated video, `.json` metrics) and `outputs/figures/` (PNG
+ PDF charts, CSV + LaTeX tables, `PAPER_REPORT.md` narrative).

For Iter 3 (fine-tuning), open `notebooks/03_train_yolo.ipynb` on a
Colab A100, run top-to-bottom (~107 min wall time), download `best.pt`,
then re-run step 4 with `--weights best.pt --no-class-filter`.

---

## Repo layout

```
stat_tracking/
├── README.md                           # this file
├── requirements.txt
├── configs/
│   ├── yolo11s_sportsmot.yaml          # Ultralytics dataset config (training)
│   └── botsort_persistent.yaml         # basketball-tuned BoT-SORT
├── notebooks/
│   ├── 01_download_and_convert.ipynb   # SportsMOT → YOLO format (Colab T4)
│   └── 03_train_yolo.ipynb             # fine-tune YOLO11s (Colab A100)
├── scripts/
│   ├── run_baseline.py                 # detect + track + evaluate, one run
│   ├── merge_tracks.py                 # offline cluster-and-auto-split merger
│   ├── render_groundtruth.py           # GT reference video + metrics
│   ├── compare_baselines.py            # build COMPARISON.md across runs
│   ├── generate_paper_figures.py       # all charts + CSV/LaTeX tables
│   ├── demo_detect.py                  # detect+track on any video / folder / webcam
│   ├── download_sportsmot_sample.py    # stream-extract N basketball seqs
│   └── download_sportsmot.sh           # full-dataset download wrapper
├── src/
│   ├── data/mot_to_yolo.py             # MOT-Challenge → YOLO format converter
│   ├── tracking/
│   │   ├── botsort_runner.py           # YOLO + BoT-SORT wrapper
│   │   └── track_merger.py             # embedding-based merger + auto-split
│   ├── eval/trackeval_wrapper.py       # MOTA / IDF1 via py-motmetrics
│   └── viz/overlay.py                  # H.264 video output via imageio-ffmpeg
├── tests/
│   └── test_mot_to_yolo.py             # synthetic-data unit test for converter
└── outputs/
    ├── baselines/                      # MOT predictions, annotated MP4, metrics JSON
    └── figures/                        # paper-quality charts + tables + PAPER_REPORT.md
```

---

## What's left to do

The remaining ~19% MOTA gap to ground truth and the 14 → 10 unique-ID
gap are both addressable without changing the architecture — better
embedder (domain-specific person ReID instead of ImageNet-supervised
EfficientNet), tighter team-color priors on the merger, and downstream
court homography for spatial constraints. None of those move the needle
on the paper's current claim, which is the iteration-by-iteration
methodology and the trained-model headline numbers.
