"""Download a small sample of SportsMOT basketball sequences for local demo.

The full SportsMOT release is ~10 GB and the Hugging Face mirror packages
each split as a single TAR file (val.tar is 6.6 GB). For a quick local demo
we only need a sequence or two, so this script *streams* the TAR and writes
out only basketball sequences, then closes the connection as soon as we
have enough. That keeps disk and bandwidth usage modest.

Usage:
    # Grab one basketball val sequence (~30 MB on disk; bandwidth varies)
    python scripts/download_sportsmot_sample.py

    # Grab three sequences
    python scripts/download_sportsmot_sample.py --num 3

    # Use the train split instead of val
    python scripts/download_sportsmot_sample.py --split train --num 1

After it finishes, run the demo on the downloaded sequence's image folder:
    python scripts/demo_detect.py --source data/sportsmot_raw/val/<seq>/img1
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_REPO = "MCG-NJU/SportsMOT"


def _seq_name_from_member(member_name: str) -> str | None:
    """Pull the v_xxx_cyyy sequence name out of a tar member path.

    Tar paths look like: 'val/v_G-vNjfx1GGc_c601/img1/000123.jpg'.
    Skip macOS AppleDouble entries -- they appear as components like
    '._val', '._v_xxx', or even leaf filenames like '._000123.jpg'.
    """
    parts = member_name.split("/")
    if any(p.startswith("._") for p in parts):
        return None
    for p in parts:
        if p.startswith("v_"):
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/sportsmot_raw"))
    p.add_argument("--num", type=int, default=1, help="How many basketball sequences to fetch")
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Which split to pull from. val is recommended (smaller, has gt).",
    )
    p.add_argument("--repo", default=HF_REPO)
    args = p.parse_args()

    try:
        import requests
        from huggingface_hub import hf_hub_url, snapshot_download
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install huggingface-hub requests")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # Step 1: pull splits_txt/ (tiny) to know which seqs are basketball.
    print(f"Fetching split metadata from {args.repo}...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["splits_txt/*"],
        local_dir=str(args.out),
    )

    bball_file = args.out / "splits_txt" / "basketball.txt"
    split_file = args.out / "splits_txt" / f"{args.split}.txt"
    if not (bball_file.exists() and split_file.exists()):
        print(f"ERROR: missing split files at {args.out}/splits_txt/")
        return 2

    bball = {l.strip() for l in bball_file.read_text().splitlines() if l.strip()}
    split_seqs = {l.strip() for l in split_file.read_text().splitlines() if l.strip()}
    targets = bball & split_seqs
    print(
        f"SportsMOT has {len(bball)} basketball sequences total; "
        f"{len(targets)} of them are in the {args.split} split."
    )
    if not targets:
        print(f"No basketball sequences in the {args.split} split. Try --split train.")
        return 3

    # Step 2: stream the TAR, write only matching basketball seqs, stop
    # once we have enough.
    tar_url = hf_hub_url(args.repo, f"dataset/{args.split}.tar", repo_type="dataset")
    print(f"Streaming {tar_url}")
    print(f"Will extract up to {args.num} basketball seq(s); aborting connection once done.")

    # The tar's internal paths already start with '<split>/', so we
    # extract into args.out (NOT args.out/<split>) to avoid doubling.
    out_split = args.out / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    # The TAR is packed as: (1) directory entries for ALL seqs first,
    # (2) then file contents grouped per-seq. We must therefore only
    # consider FILE entries when deciding whether a seq has started or
    # finished extracting -- treating directory entries as "the seq has
    # started" would falsely mark every seq complete after one dir.
    completed: set[str] = set()
    in_progress: str | None = None
    started_files_for: set[str] = set()
    bytes_read_estimate = 0
    files_written = 0
    t0 = time.time()

    with requests.get(tar_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:  # streaming, sequential
            for member in tar:
                seq = _seq_name_from_member(member.name)
                if seq is None or seq not in targets or seq in completed:
                    bytes_read_estimate += member.size
                    continue

                # Always extract directory entries (cheap, sets up the
                # parent dirs we need before extracting files).
                if member.isdir():
                    tar.extract(member, path=str(args.out))
                    continue

                # File entry for one of our target seqs.
                if in_progress != seq:
                    # We've moved to a new seq's files.
                    if in_progress is not None and in_progress in started_files_for:
                        completed.add(in_progress)
                        dt_so_far = time.time() - t0
                        print(
                            f"     finished {in_progress} "
                            f"({len(completed)}/{args.num}) at {dt_so_far:.1f}s"
                        )
                        if len(completed) >= args.num:
                            break
                    in_progress = seq
                    started_files_for.add(seq)
                    print(f"  -> extracting {seq}...")

                tar.extract(member, path=str(args.out))
                bytes_read_estimate += member.size
                files_written += 1
                if files_written % 500 == 0:
                    print(
                        f"     {files_written:>5} files | "
                        f"~{bytes_read_estimate / 1e6:6.0f} MB streamed | "
                        f"{time.time() - t0:.1f}s elapsed"
                    )

            if in_progress is not None and in_progress not in completed:
                completed.add(in_progress)

    dt = time.time() - t0
    mb = bytes_read_estimate / 1e6

    print()
    print("=" * 60)
    print(f"Done in {dt:.1f}s. Streamed ~{mb:.0f} MB; wrote {files_written} files.")
    print("Extracted basketball sequence(s):")
    for seq in sorted(completed):
        seq_dir = out_split / seq
        n_imgs = len(list((seq_dir / "img1").glob("*.jpg"))) if (seq_dir / "img1").exists() else 0
        gt_ok = (seq_dir / "gt" / "gt.txt").exists()
        print(f"  {seq}  ->  {seq_dir}  [{n_imgs} frames, gt={'yes' if gt_ok else 'no'}]")

    if completed:
        first = sorted(completed)[0]
        print()
        print("Run the detection demo on this sequence:")
        print(f"  python scripts/demo_detect.py --source {out_split / first / 'img1'} --device mps")
    print("=" * 60)
    return 0 if completed else 4


if __name__ == "__main__":
    raise SystemExit(main())
