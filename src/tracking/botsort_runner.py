"""Run a fine-tuned YOLO detector + BoT-SORT tracker over a video or image
folder, and return per-frame detections with stable track IDs.

Ultralytics ships BoT-SORT and ByteTrack as built-in tracker configs, so
all this module does is wrap `model.track(...)` with a sane interface that
returns a tidy DataFrame and (optionally) writes MOT-format output suitable
for evaluation with src/eval/trackeval_wrapper.py.

Example:
    from src.tracking.botsort_runner import track_video
    df = track_video(
        weights="runs/detect/train/weights/best.pt",
        source="data/sportsmot/dataset/val/v_xxx_c001/img1",
        tracker="botsort.yaml",
        out_mot="outputs/v_xxx_c001.txt",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TrackedDetection:
    frame: int          # 1-indexed to match MOT-Challenge convention
    track_id: int
    x: float            # bb_left
    y: float            # bb_top
    w: float
    h: float
    conf: float
    cls: int


def track_video(
    weights: str | Path,
    source: str | Path,
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 1280,
    device: str | int | None = None,
    classes: list[int] | None = None,
    out_mot: str | Path | None = None,
    save_video: bool = False,
    project: str | Path = "outputs",
    name: str = "track",
    persist: bool = True,
) -> pd.DataFrame:
    """Run detect + track over `source`. Returns a DataFrame of TrackedDetections.

    Args:
        weights:    path to a YOLO `.pt` checkpoint (fine-tuned or pretrained)
        source:     video file, image folder, or stream URL
        tracker:    'botsort.yaml' (default) or 'bytetrack.yaml'
        conf:       detection confidence threshold
        iou:        NMS IoU threshold
        imgsz:      inference image size (1280 for HD broadcast on GPU; drop
                    to 640 for CPU demos)
        device:     torch device id, e.g. 0 or 'cpu'
        classes:    list of class IDs to keep. Pass [0] when using a
                    COCO-pretrained model to keep only "person" detections.
                    Pass None for a fine-tuned single-class model.
        out_mot:    if given, also write a MOT-format text file at this path
        save_video: if True, Ultralytics writes its own annotated MP4 under
                    {project}/{name}/ -- handy for quick demos
        persist:    keep tracker state across the call (always True for video)
    """
    # Imported lazily so this module can be imported in environments that
    # don't yet have ultralytics installed (e.g. for editing or linting).
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.track(
        source=str(source),
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=classes,
        persist=persist,
        stream=True,        # avoid loading every frame into memory at once
        save=save_video,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    rows: list[TrackedDetection] = []
    for frame_idx, r in enumerate(results, start=1):
        if r.boxes is None or r.boxes.id is None:
            continue
        # xywh = (cx, cy, w, h) in pixel coords; we convert to MOT's (x, y, w, h)
        xywh = r.boxes.xywh.cpu().numpy()
        ids = r.boxes.id.int().cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.int().cpu().numpy()
        for (cx, cy, w, h), tid, c, k in zip(xywh, ids, confs, clses):
            rows.append(
                TrackedDetection(
                    frame=frame_idx,
                    track_id=int(tid),
                    x=float(cx - w / 2.0),
                    y=float(cy - h / 2.0),
                    w=float(w),
                    h=float(h),
                    conf=float(c),
                    cls=int(k),
                )
            )

    df = pd.DataFrame([d.__dict__ for d in rows])

    if out_mot is not None and not df.empty:
        out_mot = Path(out_mot)
        out_mot.parent.mkdir(parents=True, exist_ok=True)
        # MOT-Challenge format: frame, id, x, y, w, h, conf, -1, -1, -1
        with out_mot.open("w") as f:
            for d in rows:
                f.write(
                    f"{d.frame},{d.track_id},{d.x:.2f},{d.y:.2f},"
                    f"{d.w:.2f},{d.h:.2f},{d.conf:.4f},-1,-1,-1\n"
                )

    return df
