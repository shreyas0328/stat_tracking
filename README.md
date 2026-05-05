# Basketball Stat Tracking

Vision-based player detection, tracking, and (eventually) automatic
attribution of made baskets to individual players from broadcast video.

This repo extracts player-level scoring statistics from basketball
broadcasts by combining player tracking, scoring event detection, and
spatial reasoning.

---

## Project status

| Milestone | Goal | Status |
|-----------|------|--------|
| **M1. Player detection + short-horizon tracking** | Detect every player every frame; maintain stable track IDs over the few seconds around a play | **pretrained baseline working; fine-tuning pending** |
| M2. Scoring-event detection | Classify each made shot as a 2 or 3 from court geometry (Canny + Hough + homography) | not started |
| M3. Player attribution | Jersey-number OCR on key frames at the moment of a made shot to attribute the score to a specific player | not started |
| M4. End-to-end box-score generation | Run the full pipeline on a held-out game and compare the generated box score to the official one | not started |

### Current baseline (pretrained YOLO11n + BoT-SORT, no fine-tuning)

Sequence: `v_BgwzTUxJaeU_c008` (SportsMOT basketball val, 500 frames @ 1280×720).
Persisted in [`outputs/baselines/v_BgwzTUxJaeU_c008_pretrained_yolo11n.json`](outputs/baselines/).

| Metric | Value | Read |
|--------|------:|------|
| MOTA | **0.221** | Bad — anything < 0.5 is poor for sports MOT |
| IDF1 | **0.227** | Bad — identity assignment is unreliable |
| MOTP | 0.187 | Box localization on true matches is decent |
| num_false_positives | **2,358** | The killer: refs/coaches/fans detected as "person" |
| num_misses | 1,034 | Some players (small/distant) not detected |
| num_switches | 94 | ID flips across the 500 frames |
| Predictions / GT box count | 5,799 / 4,475 | 30% over-detection due to non-players |
| Unique track IDs | 158 | vs ~10–13 actual people → ~13× ID inflation |
| Inference speed | 8.1 fps on CPU | M-series Mac with `--device mps` would be ~3× faster |

These numbers are exactly the "ugly baseline" we want to demonstrate the
value of fine-tuning against. Almost the entire MOTA gap comes from
COCO-pretrained YOLO firing on every person in the broadcast frame, not
just on-court players. SportsMOT only labels the 10 active players — so a
fine-tuned model learns "player" is more specific than "person," which
should drop FP count by 60–80% and lift MOTA into the 0.6–0.8 range
without any other changes.

---

## Quick start (try it on your own machine)

This entire baseline reproduces locally on a Mac in ~5 minutes. **No GPU
needed** — the only thing that needs a GPU is fine-tuning, which happens
on Colab (see "Local vs Colab" below).

### Prerequisites

- Python 3.10+ (tested on 3.12)
- ~10 GB free disk (most is the venv + one SportsMOT sequence)

### Step 1 — clone and create a venv

```bash
git clone <this-repo-url> stat_tracking
cd stat_tracking
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — install dependencies

```bash
pip install -r requirements.txt
```

> **macOS SSL note:** if you hit `SSLCertVerificationError('OSStatus -26276')`,
> your system Python's cert bundle isn't trusted. Workaround:
> ```bash
> pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
> ```

### Step 3 — verify the code with a unit test (no downloads)

```bash
python tests/test_mot_to_yolo.py
```

Should print `PASS: synthetic SportsMOT -> YOLO conversion is correct.` in
under a second. This validates the most error-prone module (the
MOT-Challenge → YOLO format converter) without needing the real dataset.

### Step 4 — run the full baseline (downloads sample, runs detection, evaluates)

```bash
python scripts/run_baseline.py
```

What it does, end to end:
1. Stream-extracts one basketball val sequence from SportsMOT's Hugging
   Face mirror (~1 GB of bandwidth, ~50 MB on disk; aborts the connection
   as soon as it has one complete sequence).
2. Runs pretrained YOLO11n + BoT-SORT on all 500 frames (~1 min on CPU,
   first run also auto-downloads `yolo11n.pt`).
3. Renders an annotated MP4 with our visualization module.
4. Computes MOTA / IDF1 / etc. against the ground truth.
5. Saves everything under `outputs/baselines/`:
   - `..._pretrained_yolo11n.txt`  — MOT-format predictions
   - `..._pretrained_yolo11n.mp4`  — annotated video
   - `..._pretrained_yolo11n.json` — run metadata + tracking metrics

### Step 5 — actually look at the result

```bash
open outputs/baselines/v_BgwzTUxJaeU_c008_pretrained_yolo11n.mp4
```

You'll see all 500 frames with colored boxes + numbered IDs. Watch for
the failure modes the metrics quantify: refs in striped shirts boxed as
"player," IDs flipping when players cross paths, occasional missed
players in the paint.

### Try it on different content

```bash
# A different SportsMOT sequence (only the first val seq is downloaded by default)
python scripts/download_sportsmot_sample.py --num 3
python scripts/run_baseline.py --seq <one_of_the_seq_names_just_downloaded>

# Any random video file you have lying around (skips evaluation, no GT)
python scripts/demo_detect.py --source path/to/your/clip.mp4

# Webcam smoke test
python scripts/demo_detect.py --source 0
```

---

## Local vs. Colab — when to use which

| Task | Run where | Why |
|------|-----------|-----|
| Develop / edit code | **Local** | fast iteration, your editor, no upload friction |
| Run the unit test | **Local** | < 1 second, no network |
| Reproduce the pretrained baseline | **Local** | ~1 min on CPU |
| Run inference on a SportsMOT clip | **Local** | doesn't need a GPU |
| Convert SportsMOT → YOLO format for the full basketball subset | **Local or Colab** | CPU-only, but processes ~60 K files; either works |
| **Fine-tune YOLO11s on the basketball subset** | **Colab A100** | this is the only step that genuinely needs a GPU |
| Run inference with the fine-tuned model | **Local** | inference is fast on CPU; just download `best.pt` from Colab |

The Colab notebooks are *only* for the steps that need an A100. Specifically:

- `notebooks/01_download_and_convert.ipynb` — downloads the full SportsMOT
  basketball subset, converts it to YOLO format, mounts Google Drive so
  the converted dataset persists across Colab runtime restarts. **Use a
  free T4 runtime — this step needs no GPU; it's only on Colab so the
  output lands on the same Drive that the next notebook reads from.**
- `notebooks/02_smoke_test_train.ipynb` *(planned)* — fine-tune YOLO11n
  for 5 epochs on a tiny subset to verify the training pipeline before
  spending A100 hours. **Free T4.**
- `notebooks/03_train_yolo.ipynb` *(planned)* — the real fine-tune of
  YOLO11s on the full basketball train split. **A100; ~2 hours.**

After the Colab fine-tune finishes, you download the resulting
`best.pt` back to your local machine and re-run the same baseline command:

```bash
python scripts/run_baseline.py \
    --weights path/to/best.pt \
    --tag finetuned_yolo11s \
    --no-class-filter
```

That writes a second baseline JSON to `outputs/baselines/` so you can
A/B compare the pretrained vs fine-tuned numbers directly.

### Why training has to happen on a GPU

YOLO11s fine-tuning on the SportsMOT basketball subset means ~30 epochs
over ~60K images. On an A100 this is ~2 hours; on Apple Silicon CPU
it's roughly **2 days** of compute. The Google Colab Student plan
(~10 A100 hours/month) is comfortably enough to run the fine-tune plus
several ablations.

---

## Repo layout

```
stat_tracking/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── yolo11s_sportsmot.yaml          # Ultralytics dataset config (used during training)
├── notebooks/                          # all Colab-runnable
│   └── 01_download_and_convert.ipynb   # MOT → YOLO; T4
├── scripts/                            # local CLI entrypoints
│   ├── run_baseline.py                 # ★ one-command repro of the pretrained baseline
│   ├── demo_detect.py                  # detect+track on any video / image folder / webcam
│   ├── download_sportsmot_sample.py    # stream-extract N basketball seqs from HF
│   └── download_sportsmot.sh           # full-dataset download wrapper
├── src/
│   ├── data/mot_to_yolo.py             # MOT-Challenge → YOLO format converter
│   ├── tracking/botsort_runner.py      # YOLO + BoT-SORT wrapper, returns DataFrame + MOT dump
│   ├── eval/trackeval_wrapper.py       # MOTA / IDF1 via py-motmetrics
│   └── viz/overlay.py                  # draw boxes + IDs on a video
├── tests/
│   └── test_mot_to_yolo.py             # synthetic-data unit test for the converter
├── data/                               # gitignored; SportsMOT lands here
└── outputs/
    └── baselines/                      # MOT predictions, annotated MP4, metrics JSON per run
```

---

## Design decisions

- **SportsMOT basketball subset** as the training/eval dataset — explicitly
  designed for broadcast sports MOT, ~80 basketball clips with persistent
  player track IDs in MOT-Challenge format.
- **YOLO11s + BoT-SORT** as the model — fits in a couple of A100-hours,
  and Ultralytics ships the BoT-SORT integration so we get tracking
  almost for free.
- **Shot-window identity strategy** — long-horizon player re-identification
  across a full game (with camera cuts and replays) is an open research
  problem. We sidestep it by deciding that the tracker only needs to hold
  IDs across the short window around a scoring event, and a jersey-number
  OCR step (M3) is what actually resolves "who scored." This makes M1 a
  standard short-horizon MOT task that BoT-SORT handles well.
- **motmetrics over TrackEval** for evaluation — much lighter install,
  metrics are sufficient for iterating on the detector and tracker. Swap
  in TrackEval for the writeup if you want HOTA numbers.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'lap'`** during tracking — install it:
  `pip install lap` (already pinned in `requirements.txt`).
- **`AttributeError: np.asfarray was removed`** during evaluation —
  fixed in `src/eval/trackeval_wrapper.py` via a backport monkey-patch;
  pull latest if you see it.
- **macOS SSL cert errors during `pip install`** — see Step 2 above.
- **HF download produces a different sequence than expected** — that's
  fine; the streaming downloader extracts whichever basketball sequence
  it encounters first in the tar. Pass `--num 3` to `download_sportsmot_sample.py`
  to grab a few and pick one.

---

## License

TBD.
