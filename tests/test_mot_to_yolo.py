"""Self-contained test for src/data/mot_to_yolo.py.

Builds a tiny synthetic SportsMOT-like directory tree on disk, runs the
converter, and asserts the output is well-formed. No real dataset, no
network, no GPU needed. Runs in well under a second.

Usage:
    python tests/test_mot_to_yolo.py        # plain script, prints PASS/FAIL
    pytest tests/test_mot_to_yolo.py        # also works under pytest
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make `src` importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402  (after sys.path tweak)

from src.data.mot_to_yolo import convert  # noqa: E402


# Build a synthetic SportsMOT layout:
#   <root>/splits_txt/basketball.txt              -> includes our fake seq names
#   <root>/dataset/train/<seq>/seqinfo.ini
#   <root>/dataset/train/<seq>/img1/000001.jpg ...
#   <root>/dataset/train/<seq>/gt/gt.txt
#
# We make two basketball train sequences and one volleyball train sequence
# so we can verify the sport filter actually filters.
TRAIN_SEQS = [
    ("v_bball_c001", "basketball", 4),   # 4 frames
    ("v_bball_c002", "basketball", 3),   # 3 frames
    ("v_volley_c001", "volleyball", 2),  # 2 frames -- should be filtered out
]
VAL_SEQS = [
    ("v_bball_c003", "basketball", 2),
]


def _make_seq(root: Path, split: str, name: str, n_frames: int, img_w: int = 64, img_h: int = 48) -> None:
    seq_dir = root / "dataset" / split / name
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\n"
        f"name={name}\n"
        f"imDir=img1\n"
        f"frameRate=25\n"
        f"seqLength={n_frames}\n"
        f"imWidth={img_w}\n"
        f"imHeight={img_h}\n"
        f"imExt=.jpg\n"
    )

    # Tiny solid-color JPEGs: enough to exist as files; we never decode them.
    for i in range(1, n_frames + 1):
        Image.new("RGB", (img_w, img_h), color=(i * 30 % 255, 50, 50)).save(
            seq_dir / "img1" / f"{i:06d}.jpg", "JPEG", quality=70
        )

    # Two players per frame, except frame 2 which has zero (tests the
    # empty-label-file branch).
    lines = []
    for f in range(1, n_frames + 1):
        if f == 2:
            continue
        lines.append(f"{f},1,10,10,20,30,1,1,1")   # box A
        lines.append(f"{f},2,30,5,15,25,1,1,1")    # box B
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")


def _make_dataset(root: Path) -> None:
    for name, sport, n in TRAIN_SEQS:
        _make_seq(root, "train", name, n)
    for name, sport, n in VAL_SEQS:
        _make_seq(root, "val", name, n)

    splits = root / "splits_txt"
    splits.mkdir(parents=True, exist_ok=True)
    bball = [n for (n, s, _) in TRAIN_SEQS + VAL_SEQS if s == "basketball"]
    (splits / "basketball.txt").write_text("\n".join(bball) + "\n")
    volley = [n for (n, s, _) in TRAIN_SEQS + VAL_SEQS if s == "volleyball"]
    (splits / "volleyball.txt").write_text("\n".join(volley) + "\n")


def test_convert_basketball_subset() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = td / "sportsmot_raw"
        out = td / "sportsmot_yolo"
        _make_dataset(raw)

        stats = convert(raw, out, sport="basketball", copy=True)

        # ---- Structure ---------------------------------------------------
        for split in ("train", "val", "test"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

        # ---- Sport filter actually filters volleyball out ----------------
        train_imgs = sorted((out / "images" / "train").glob("*.jpg"))
        names = {p.name for p in train_imgs}
        assert all("volley" not in n for n in names), (
            "Volleyball seq leaked into output; sport filter is broken"
        )

        # ---- Counts ------------------------------------------------------
        # Two basketball train seqs, 4+3=7 frames total.
        assert stats["sequences"] == 3, stats   # 2 train bball + 1 val bball
        assert stats["frames"] == 4 + 3 + 2, stats
        # Each frame has 2 boxes EXCEPT frame 2 (0 boxes). So labeled frames:
        #   c001: frames 1,3,4 -> 3
        #   c002: frames 1,3   -> 2 (only 3 frames total, frame 2 empty)
        #   val_c003: frames 1 -> 1 (only 2 frames total, frame 2 empty)
        # = 6 labeled frames, 12 boxes total.
        assert stats["labeled_frames"] == 6, stats
        assert stats["boxes"] == 12, stats

        # ---- One label file content sanity check ------------------------
        # Frame 1 of v_bball_c001: img is 64x48, two boxes.
        # Box A: (10,10,20,30) -> cx=20, cy=25, w=20, h=30 -> normalized:
        #   cx/64=0.3125, cy/48=0.5208333, w/64=0.3125, h/48=0.625
        sample = out / "labels" / "train" / "v_bball_c001_000001.txt"
        assert sample.exists(), f"Missing expected label file: {sample}"
        rows = sample.read_text().strip().splitlines()
        assert len(rows) == 2, rows
        cls, cx, cy, w, h = rows[0].split()
        assert cls == "0"
        assert abs(float(cx) - 0.3125) < 1e-4
        assert abs(float(cy) - 25 / 48) < 1e-4
        assert abs(float(w) - 0.3125) < 1e-4
        assert abs(float(h) - 0.625) < 1e-4

        # ---- Empty label files exist for unlabeled frames ---------------
        # Frame 2 of v_bball_c001 has no annotations; YOLO needs an empty
        # .txt to treat it as a true negative rather than skipping it.
        empty = out / "labels" / "train" / "v_bball_c001_000002.txt"
        assert empty.exists() and empty.stat().st_size == 0, (
            "Expected an empty label file for an unlabeled frame; "
            "without this YOLO silently drops the frame."
        )

        # ---- Test split has images but no labels ------------------------
        # (Our synthetic data has no test split at all -- assert it stayed empty.)
        assert not list((out / "labels" / "test").glob("*.txt"))


def main() -> int:
    try:
        test_convert_basketball_subset()
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2
    print("PASS: synthetic SportsMOT -> YOLO conversion is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
