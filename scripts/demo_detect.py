"""Run pretrained-or-fine-tuned YOLO + BoT-SORT on a basketball clip and
produce an annotated MP4 + summary stats. Designed as the simplest possible
"does this work" entrypoint.

Examples:
    # Pretrained YOLO11n (auto-downloads on first run), CPU, person-only
    python scripts/demo_detect.py --source path/to/clip.mp4

    # Same, but with the small variant for a bit more accuracy
    python scripts/demo_detect.py --source path/to/clip.mp4 --weights yolo11s.pt

    # Use your fine-tuned weights once milestone 1 training is done
    python scripts/demo_detect.py \\
        --source path/to/clip.mp4 \\
        --weights runs/detect/train/weights/best.pt \\
        --no-class-filter

    # Webcam (mostly useful for verifying the install works end-to-end)
    python scripts/demo_detect.py --source 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tracking.botsort_runner import track_video  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Video file path, image folder, or webcam id")
    p.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="YOLO checkpoint. Pretrained sizes auto-download: yolo11n/s/m/l/x.pt",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (lower = faster)")
    p.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    p.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. 0 (cuda), mps (Apple Silicon GPU), cpu",
    )
    p.add_argument(
        "--no-class-filter",
        action="store_true",
        help="Don't filter to person class. Use this with a fine-tuned single-class model.",
    )
    p.add_argument(
        "--no-track",
        action="store_true",
        help="Detection only; skip BoT-SORT tracking (faster, but no IDs)",
    )
    p.add_argument("--out", default="outputs", help="Where to write the annotated MP4")
    p.add_argument("--name", default="demo", help="Subfolder name under --out")
    args = p.parse_args()

    # When using a COCO-pretrained model we want only the "person" class
    # (id=0). When using a fine-tuned SportsMOT model we keep everything.
    classes = None if args.no_class_filter else [0]

    # Webcam id support: '0' -> 0
    src: str | int = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    print(f"Source       : {src}")
    print(f"Weights      : {args.weights}")
    print(f"Image size   : {args.imgsz}")
    print(f"Conf thresh  : {args.conf}")
    print(f"Device       : {args.device or 'auto'}")
    print(f"Class filter : {classes}")
    print(f"Tracker      : {'(none)' if args.no_track else 'botsort.yaml'}")
    print()

    t0 = time.time()

    if args.no_track:
        # Pure detection path: use Ultralytics directly, save annotated MP4.
        from ultralytics import YOLO
        model = YOLO(args.weights)
        results = model.predict(
            source=src,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            classes=classes,
            save=True,
            project=args.out,
            name=args.name,
            exist_ok=True,
            stream=True,
            verbose=False,
        )
        n_frames = 0
        n_dets = 0
        for r in results:
            n_frames += 1
            n_dets += 0 if r.boxes is None else len(r.boxes)
        n_tracks = None
    else:
        df = track_video(
            weights=args.weights,
            source=src,
            tracker="botsort.yaml",
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=classes,
            save_video=True,
            project=args.out,
            name=args.name,
        )
        n_frames = int(df["frame"].max()) if not df.empty else 0
        n_dets = len(df)
        n_tracks = df["track_id"].nunique() if not df.empty else 0

    dt = time.time() - t0

    print()
    print("=" * 50)
    print("Done.")
    print(f"  Frames processed : {n_frames:,}")
    print(f"  Total detections : {n_dets:,}")
    if n_tracks is not None:
        print(f"  Unique track IDs : {n_tracks:,}")
        if n_frames:
            print(f"  Avg dets / frame : {n_dets / n_frames:.1f}")
    print(f"  Wall time        : {dt:.1f} s ({n_frames / max(dt, 1e-6):.1f} fps)")
    print(f"  Annotated video  : {Path(args.out) / args.name}/")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
