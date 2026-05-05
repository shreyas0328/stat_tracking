"""Render bounding boxes + track IDs onto a video for qualitative inspection.

Two entry points:
  - draw_on_frame(): the per-frame primitive. Useful inside notebooks.
  - render_video():  takes a directory of frames or a video file plus a
                     prediction DataFrame (or MOT-format text file) and
                     writes an annotated MP4 encoded as H.264 (so it plays
                     in QuickTime, Safari, browsers, etc. -- not just VLC).

Video encoding uses `imageio` with the bundled `imageio-ffmpeg` backend so
no system-wide ffmpeg install is required.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd


def _id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct color per ID (BGR for OpenCV)."""
    hue = (track_id * 0.6180339887) % 1.0  # golden-ratio hop for separation
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_on_frame(
    frame: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes on a frame in-place-ish (returns the same array).

    Args:
        frame: HxWx3 BGR uint8 array
        boxes: list of (track_id, x, y, w, h) in pixel coords
    """
    for track_id, x, y, w, h in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = _id_to_color(int(track_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"#{int(track_id)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return frame


def _load_predictions(predictions: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        return predictions
    path = Path(predictions)
    df = pd.read_csv(path, header=None)
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf"]
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    return df


def _open_h264_writer(out_path: Path, fps: float):
    """Create an imageio writer configured for H.264 (libx264) MP4.

    `macro_block_size=1` lets us write any (w, h) without padding;
    `quality=8` is roughly visually lossless at HD resolutions while still
    yielding ~10x smaller files than uncompressed.
    """
    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=1,
        quality=8,
        pixelformat="yuv420p",   # required for QuickTime / Safari compatibility
    )


def _frame_boxes(df: pd.DataFrame, frame_idx: int) -> list[tuple[int, float, float, float, float]]:
    sub = df[df["frame"] == frame_idx]
    return list(
        zip(
            sub["track_id"].astype(int),
            sub["x"], sub["y"], sub["w"], sub["h"],
        )
    )


def render_video(
    source: str | Path,
    predictions: pd.DataFrame | str | Path,
    out_path: str | Path,
    fps: float = 25.0,
) -> Path:
    """Write an annotated H.264 MP4 to out_path.

    `source` can be either a video file or a directory of frames named
    000001.jpg, 000002.jpg, ... (the SportsMOT layout).

    Frames are streamed through the encoder one at a time, so memory usage
    stays bounded regardless of clip length.
    """
    df = _load_predictions(predictions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = Path(source)
    writer = _open_h264_writer(out_path, fps)
    try:
        if source.is_dir():
            frame_paths = sorted(source.glob("*.jpg"))
            if not frame_paths:
                raise RuntimeError(f"No .jpg frames found in {source}")
            for i, fp in enumerate(frame_paths, start=1):
                frame = cv2.imread(str(fp))
                annotated = draw_on_frame(frame, _frame_boxes(df, i))
                # imageio expects RGB; OpenCV gives us BGR.
                writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {source}")
            i = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    i += 1
                    annotated = draw_on_frame(frame, _frame_boxes(df, i))
                    writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
    finally:
        writer.close()

    return out_path
