# Basketball Player Tracking

Track every player in a broadcast basketball clip with a **stable ID per
person for the whole clip**. Built on top of YOLO11 + BoT-SORT, with an
offline appearance-embedding merger to glue together fragmented IDs.

## What it does

Input: a basketball broadcast video.
Output:
- An annotated `.mp4` with bounding boxes and stable player IDs.
- A `.txt` of tracks in MOT-Challenge format.
- A `.json` with tracking metrics (MOTA, IDF1, ID switches, etc.) when
  ground truth is available.

## Pipeline

```
video frames
   │
   ▼
[ YOLO11 detector ]            ← person bounding boxes
   │
   ▼
[ BoT-SORT tracker ]           ← raw tracks (often fragmented)
   │
   ▼
[ Appearance merger ]          ← clusters fragments into persistent IDs
   │
   ▼
annotated video + metrics
```

1. **Detect** players with YOLO11 (n / s / m). A fine-tuned YOLO11s on
   SportsMOT basketball gives the best detections.
2. **Track** with BoT-SORT (`configs/botsort_persistent.yaml`) — long
   buffer + Re-ID for persistence through occlusions.
3. **Merge** the tracker's fragmented IDs by clustering per-track
   appearance embeddings (EfficientNet-B0 features) under a
   temporal-overlap constraint, with iterative auto-split for clusters
   that accidentally group multiple people.

## Repo layout

```
configs/         tracker + training configs
data/            SportsMOT samples (downloaded on demand)
notebooks/       Colab notebooks (data prep, fine-tuning)
scripts/         CLI entry points (run_baseline, merge_tracks, ...)
src/
  data/          dataset loaders / converters
  tracking/      appearance merger + post-processing
  eval/          motmetrics wrapper
  viz/           overlay drawing (boxes, IDs)
outputs/         annotated videos, metrics JSON, paper figures
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Grab a sample SportsMOT clip
python scripts/download_sportsmot_sample.py --num 1

# 2. Run detect + track + evaluate end-to-end
python scripts/run_baseline.py \
    --weights yolo11m.pt \
    --tracker configs/botsort_persistent.yaml

# 3. Merge fragmented IDs into stable identities
python scripts/merge_tracks.py \
    --predictions outputs/baselines/<clip>_persistent_v2_yolo11m.txt \
    --frames-dir  data/sportsmot/<clip>/img1
```

Outputs land in `outputs/baselines/`.

## Fine-tuning

`notebooks/03_train_yolo.ipynb` fine-tunes YOLO11s on the SportsMOT
basketball train split (single class: `player`). Drop it into Colab,
point it at the dataset, and the resulting `best.pt` plugs straight
into `run_baseline.py` via `--weights`.

## Evaluation

`scripts/run_baseline.py` reports MOTA / IDF1 / ID-switches against
SportsMOT ground truth using `motmetrics`. Use
`scripts/compare_baselines.py` to roll up all runs in
`outputs/baselines/` into a markdown comparison table.
