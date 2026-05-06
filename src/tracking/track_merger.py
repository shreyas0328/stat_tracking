"""Post-process track merger: collapse fragmented tracks into stable identities.

Problem
-------
Even with a tuned BoT-SORT (long ``track_buffer``, ReID enabled), an online
tracker cannot link a player who leaves the frame for >10s with the player
who reappears later — by that point the lost track is dead. The result is
"fragmentation": one real player ends up with 3-5 different track IDs.

This module fixes that with an offline pass over the tracker's output:

1.  Load the MOT-format predictions.
2.  For each track, sample a handful of frames and crop the player.
3.  Embed each crop with a pretrained CNN, average per track to get one
    embedding per track.
4.  Build a distance matrix between tracks where ``d(i, j) = 1 - cos_sim``,
    but force ``d(i, j) = +inf`` whenever tracks ``i`` and ``j`` co-exist
    in the same frame (a player can't be in two places at once).
5.  Run constrained agglomerative clustering with average linkage. Either:
    * a fixed ``n_clusters`` (use when you know the team size, e.g. 10), or
    * a ``distance_threshold`` (use when you want the algorithm to decide).
6.  Rewrite the predictions with the cluster ID as the new track ID.

Why this works
--------------
Two tracks of the same player in different parts of the clip will look
similar in any reasonable embedding space (same uniform colour, same body
shape, same pose distribution). Two tracks of different players will look
different. The temporal-overlap constraint guarantees we never merge two
people who appeared simultaneously — the most common false-positive merge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MOT_COLS = ["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b", "c"]


@dataclass
class TrackEmbedding:
    """One row of per-track aggregate info used by the merger."""

    track_id: int
    num_detections: int
    frames: np.ndarray            # sorted unique frame indices
    embedding: np.ndarray         # (D,) L2-normalised mean embedding


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a MOT-format predictions file as a DataFrame."""
    return pd.read_csv(path, header=None, names=MOT_COLS)


def write_predictions(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame back to MOT-format with the standard 10 columns."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in df.itertuples(index=False):
            conf = getattr(r, "conf", -1.0)
            f.write(
                f"{int(r.frame)},{int(r.track_id)},{r.x:.2f},{r.y:.2f},"
                f"{r.w:.2f},{r.h:.2f},{conf:.4f},-1,-1,-1\n"
            )


def _select_long_tracks(df: pd.DataFrame, min_detections: int) -> List[int]:
    counts = df.groupby("track_id").size()
    return counts[counts >= min_detections].index.astype(int).tolist()


def _sample_frames(track_df: pd.DataFrame, k: int) -> pd.DataFrame:
    track_df = track_df.sort_values("frame").reset_index(drop=True)
    if len(track_df) <= k:
        return track_df
    idx = np.linspace(0, len(track_df) - 1, num=k, dtype=int)
    return track_df.iloc[idx].reset_index(drop=True)


def _crop_player(frame_bgr: np.ndarray, x: float, y: float, w: float, h: float) -> Optional[np.ndarray]:
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(w_img, int(round(x + w)))
    y2 = min(h_img, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


class _Embedder:
    """Wraps a pretrained ImageNet CNN to produce L2-normalised embeddings.

    We default to ``efficientnet_b0`` because it's small and fast on CPU.
    Any timm model with ``num_classes=0`` works as a drop-in replacement
    (e.g. ``convnext_tiny`` for stronger embeddings at higher cost).
    """

    def __init__(self, backbone: str = "efficientnet_b0", device: Optional[str] = None) -> None:
        self._backbone = backbone
        self._device_arg = device
        self._model = None
        self._device = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch

        # All caches forced inside the workspace; macOS sandboxing + the
        # IDE's permission model means ~/.cache is not always writable, and
        # timm + huggingface_hub will silently die if their cache dirs
        # fail to create. Doing this here means ``track_merger`` is
        # self-contained and doesn't rely on env-var setup elsewhere.
        repo_root = Path(__file__).resolve().parents[2]
        embedder_cache = repo_root / "models" / "embedder"
        embedder_cache.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("TORCH_HOME", str(embedder_cache / "torch_home"))
        os.environ.setdefault("HF_HOME", str(embedder_cache / "hf_home"))
        os.environ.setdefault("HF_HUB_CACHE", str(embedder_cache / "hf_home" / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(embedder_cache / "hf_home" / "xet"))
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

        import timm

        if self._device_arg is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = self._device_arg

        model = timm.create_model(self._backbone, pretrained=True, num_classes=0)
        model.to(self._device).eval()
        self._model = model

    def embed(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Return ``(N, D)`` L2-normalised embeddings for ``N`` BGR crops."""
        import cv2
        import torch

        self._ensure()

        if not crops_bgr:
            return np.zeros((0, 1), dtype=np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
            arr = (resized.astype(np.float32) - mean) / std
            arr = np.transpose(arr, (2, 0, 1))
            tensors.append(torch.from_numpy(arr))

        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch)
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return feats / norms


def compute_track_embeddings(
    df: pd.DataFrame,
    frames_dir: str | Path,
    samples_per_track: int = 8,
    min_detections: int = 8,
    min_bbox_height: int = 80,
    backbone: str = "efficientnet_b0",
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[TrackEmbedding]:
    """Per-track mean embedding + frame ranges, used as input to clustering.

    Tracks shorter than ``min_detections`` are dropped — they're almost
    certainly fragments of a longer track that we can't represent reliably,
    and including them inflates the cluster count.
    """
    import cv2

    frames_dir = Path(frames_dir)
    long_tracks = _select_long_tracks(df, min_detections)
    if verbose:
        print(f"[merger] kept {len(long_tracks)} tracks with >= {min_detections} detections "
              f"(out of {df['track_id'].nunique()} total)")

    embedder = _Embedder(backbone=backbone, device=device)

    out: List[TrackEmbedding] = []
    for tid in long_tracks:
        track_df = df[df["track_id"] == tid]
        big_enough = track_df[track_df["h"] >= min_bbox_height]
        sampled = _sample_frames(big_enough if len(big_enough) > 0 else track_df, samples_per_track)

        crops: List[np.ndarray] = []
        for _, row in sampled.iterrows():
            frame_idx = int(row["frame"])
            frame_path = frames_dir / f"{frame_idx:06d}.jpg"
            if not frame_path.exists():
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            c = _crop_player(frame, row["x"], row["y"], row["w"], row["h"])
            if c is not None:
                crops.append(c)

        if not crops:
            if verbose:
                print(f"[merger] track {tid}: no usable crops, skipping")
            continue

        embs = embedder.embed(crops)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / max(1e-9, np.linalg.norm(mean_emb))
        frames = np.sort(track_df["frame"].astype(int).unique())
        out.append(TrackEmbedding(
            track_id=int(tid),
            num_detections=int(len(track_df)),
            frames=frames,
            embedding=mean_emb.astype(np.float32),
        ))
        if verbose:
            print(f"[merger] track {tid:4d}: {len(track_df):3d} dets, "
                  f"frames {frames.min():3d}..{frames.max():3d}, "
                  f"embedded from {len(crops)} crops")
    return out


def _build_distance_matrix(
    embeddings: List[TrackEmbedding],
    forbid_overlap: bool = True,
    overlap_penalty: float = 1e6,
) -> np.ndarray:
    """Cosine distance matrix with infinity on temporally-overlapping pairs."""
    n = len(embeddings)
    feat = np.stack([e.embedding for e in embeddings], axis=0)
    cos_sim = feat @ feat.T  # already L2-normalised
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    dist = 1.0 - cos_sim
    np.fill_diagonal(dist, 0.0)

    if forbid_overlap:
        frame_sets = [set(int(x) for x in e.frames.tolist()) for e in embeddings]
        for i in range(n):
            for j in range(i + 1, n):
                if frame_sets[i] & frame_sets[j]:
                    dist[i, j] = overlap_penalty
                    dist[j, i] = overlap_penalty
    return dist.astype(np.float64)


@dataclass
class ClusterResult:
    """Output of :func:`cluster_tracks`.

    Attributes
    ----------
    mapping
        ``{original_track_id: cluster_id}``. Cluster IDs are 0-indexed.
    track_silhouette
        ``{original_track_id: silhouette_score}``. The standard sklearn
        silhouette for that track in its assigned cluster, in [-1, 1].
        Higher = the track fits its cluster better than any alternative.
        Negative = the track is closer to a different cluster than its own
        (i.e. the merge is suspect).
    track_confidence
        ``{original_track_id: confidence}`` rescaled from silhouette into
        [0, 1] via ``(silhouette + 1) / 2``. Easier to read as a percentage.
    cluster_silhouette
        ``{cluster_id: mean_silhouette_over_member_tracks}``.
    cluster_confidence
        ``{cluster_id: confidence}`` rescaled to [0, 1].
    cluster_size
        ``{cluster_id: number_of_tracks_in_it}`` -- a cluster of 1 track
        has nothing to merge so its silhouette is conventionally 0; size
        is a useful tiebreaker when reading the report.
    cluster_total_detections
        ``{cluster_id: total_detections_across_all_tracks_in_cluster}`` --
        the most useful "is this a real player" signal: a cluster covering
        450/500 frames is almost certainly a real player even if its
        embedding silhouette is mediocre.
    """

    mapping: Dict[int, int]
    track_silhouette: Dict[int, float]
    track_confidence: Dict[int, float]
    cluster_silhouette: Dict[int, float]
    cluster_confidence: Dict[int, float]
    cluster_size: Dict[int, int]
    cluster_total_detections: Dict[int, int]


def _silhouette_to_confidence(s: float) -> float:
    """Rescale silhouette score from [-1, 1] to [0, 1]."""
    return float(max(0.0, min(1.0, (s + 1.0) / 2.0)))


def cluster_tracks(
    embeddings: List[TrackEmbedding],
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    forbid_overlap: bool = True,
    min_cluster_silhouette: float = 0.0,
    verbose: bool = True,
) -> ClusterResult:
    """Constrained agglomerative clustering with per-cluster confidences.

    Exactly one of ``n_clusters`` and ``distance_threshold`` must be set.
    Use ``n_clusters`` when you know the team size (e.g. 10 players +
    a slack buffer for refs / hard-to-merge fragments). Use
    ``distance_threshold`` (e.g. 0.4) when you want the algorithm to
    decide based on appearance similarity alone.

    Confidence comes from sklearn's silhouette score computed against the
    *appearance-only* distance matrix (i.e. the temporal-overlap penalty is
    removed before computing silhouettes -- otherwise tracks separated by
    "you're in two places at once, infinite distance" would dominate the
    silhouette and hide the real appearance signal). This means the
    confidence score answers the question: "given the visual evidence, how
    well does this track actually belong to the cluster it was assigned to?"

    Garbage-cluster auto-split
    --------------------------
    When ``n_clusters`` is forced to a small value (e.g. 10 because we
    know the team size), agglomerative is required to assign *every* track
    to one of those K clusters -- including refs, coaches, partial-frame
    glimpses, the ball, and other noise. The algorithm has no choice but
    to dump these into "garbage" clusters which then end up containing
    multiple distinct people sharing one rendered ID. The dead giveaway
    is a NEGATIVE mean silhouette: sklearn is literally saying "the tracks
    in this cluster are closer to other clusters than to each other".

    To prevent multiple people from being painted with the same ID we
    auto-split any cluster whose mean silhouette is below
    ``min_cluster_silhouette`` (default 0.0): each fragment in that
    cluster becomes its own singleton identity. The resulting total ID
    count may exceed ``n_clusters``, but each ID corresponds to a single
    coherent person -- which is the invariant player-tracking needs.
    """
    if (n_clusters is None) == (distance_threshold is None):
        raise ValueError("specify exactly one of n_clusters or distance_threshold")
    if not embeddings:
        return ClusterResult({}, {}, {}, {}, {}, {}, {})

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_samples

    dist_constrained = _build_distance_matrix(embeddings, forbid_overlap=forbid_overlap)
    dist_appearance = _build_distance_matrix(embeddings, forbid_overlap=False)

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    raw_labels = clusterer.fit_predict(dist_constrained)
    n_raw_clusters = len(set(int(l) for l in raw_labels))

    # Iteratively split bad clusters until no negative-silhouette merges
    # remain. Silhouette is a *relative* metric (each track's score depends
    # on the nearest other cluster), so splitting one bad cluster can flip
    # a previously-positive cluster to negative. A single pass is not
    # enough to guarantee the "each ID = one person" invariant; we have
    # to fixpoint. Capped at MAX_ITERS to avoid pathological loops on
    # degenerate inputs.
    MAX_ITERS = 8
    final_labels = list(map(int, raw_labels))
    n_split_clusters_total = 0
    n_split_tracks_total = 0

    for iteration in range(MAX_ITERS):
        labels_arr = np.array(final_labels, dtype=int)
        unique_now = sorted(set(int(l) for l in labels_arr))
        if len(unique_now) <= 1:
            break

        sil_now = silhouette_samples(dist_appearance, labels_arr, metric="precomputed")

        members_now: Dict[int, List[int]] = {}
        for i, l in enumerate(labels_arr):
            members_now.setdefault(int(l), []).append(i)

        cluster_sil_now: Dict[int, float] = {}
        for c, idxs in members_now.items():
            if len(idxs) == 1:
                cluster_sil_now[c] = 1.0
            else:
                cluster_sil_now[c] = float(np.mean([sil_now[i] for i in idxs]))

        bad_now = [c for c, s in cluster_sil_now.items()
                   if s < min_cluster_silhouette and len(members_now[c]) > 1]
        if not bad_now:
            break

        next_label = max(unique_now) + 1
        for bad in sorted(bad_now):
            idxs = members_now[bad]
            for i in idxs[1:]:
                final_labels[i] = next_label
                next_label += 1
                n_split_tracks_total += 1
            n_split_clusters_total += 1

    final_labels_arr = np.array(final_labels, dtype=int)
    final_unique = sorted(set(int(l) for l in final_labels_arr))
    if len(final_unique) <= 1:
        sil_per_track = np.zeros(len(embeddings), dtype=np.float64)
    else:
        sil_per_track = silhouette_samples(dist_appearance, final_labels_arr, metric="precomputed")

    raw_by_final: Dict[int, List[int]] = {}
    raw_dets_by_final: Dict[int, int] = {}
    for emb, lbl in zip(embeddings, final_labels_arr):
        c = int(lbl)
        raw_by_final.setdefault(c, []).append(int(emb.track_id))
        raw_dets_by_final[c] = raw_dets_by_final.get(c, 0) + emb.num_detections

    # Renumber clusters into contiguous 0..N-1 IDs sorted by total
    # detection count (descending), so the most-on-screen player gets
    # cluster 0 / track ID 1, the next gets 1, etc. Without this step,
    # auto-splitting leaves holes (clusters 0, 1, 2, 4, 7, 11, ...) which
    # surface as confusing gaps in the rendered video labels.
    sorted_old_ids = sorted(raw_by_final.keys(),
                            key=lambda c: (-raw_dets_by_final[c], c))
    relabel = {old: new for new, old in enumerate(sorted_old_ids)}

    mapping: Dict[int, int] = {}
    track_sil: Dict[int, float] = {}
    track_conf: Dict[int, float] = {}
    by_cluster: Dict[int, List[int]] = {}
    dets_by_cluster: Dict[int, int] = {}

    for emb, lbl, sil in zip(embeddings, final_labels_arr, sil_per_track):
        c = relabel[int(lbl)]
        tid = int(emb.track_id)
        mapping[tid] = c
        track_sil[tid] = float(sil)
        track_conf[tid] = _silhouette_to_confidence(float(sil))
        by_cluster.setdefault(c, []).append(tid)
        dets_by_cluster[c] = dets_by_cluster.get(c, 0) + emb.num_detections

    cluster_sil: Dict[int, float] = {}
    cluster_conf: Dict[int, float] = {}
    cluster_size: Dict[int, int] = {}
    for c, tids in by_cluster.items():
        if len(tids) == 1:
            cluster_sil[c] = 1.0
            cluster_conf[c] = 1.0
        else:
            mean_sil = float(np.mean([track_sil[t] for t in tids]))
            cluster_sil[c] = mean_sil
            cluster_conf[c] = _silhouette_to_confidence(mean_sil)
        cluster_size[c] = len(tids)

    if verbose:
        print(f"[merger] clustered {len(embeddings)} tracks into "
              f"{n_raw_clusters} raw groups -> {len(by_cluster)} after "
              f"auto-splitting {n_split_clusters_total} garbage cluster(s) "
              f"(silhouette < {min_cluster_silhouette:.2f}, "
              f"{n_split_tracks_total} fragments demoted to singletons)")
        for c in sorted(by_cluster.keys()):
            sil = cluster_sil[c]
            conf = cluster_conf[c]
            label = "high" if conf >= 0.75 else ("med" if conf >= 0.55 else "LOW")
            print(f"[merger]   cluster {c}: {cluster_size[c]} tracks, "
                  f"{dets_by_cluster[c]} detections, "
                  f"silhouette={sil:+.3f} conf={conf:.0%} [{label}]")

    return ClusterResult(
        mapping=mapping,
        track_silhouette=track_sil,
        track_confidence=track_conf,
        cluster_silhouette=cluster_sil,
        cluster_confidence=cluster_conf,
        cluster_size=cluster_size,
        cluster_total_detections=dets_by_cluster,
    )


def remap_predictions(
    df: pd.DataFrame,
    mapping: "Dict[int, int] | ClusterResult",
    drop_unmapped: bool = True,
    id_offset: int = 1,
) -> pd.DataFrame:
    """Rewrite ``track_id`` using ``mapping``; optionally drop tracks not in it.

    ``id_offset`` makes cluster labels human-readable (1-indexed instead of 0).
    Accepts either a raw ``{old_id: new_id}`` dict or a :class:`ClusterResult`
    (in which case its ``.mapping`` field is used).
    """
    if isinstance(mapping, ClusterResult):
        mapping = mapping.mapping
    df = df.copy()
    if drop_unmapped:
        df = df[df["track_id"].astype(int).isin(mapping.keys())].copy()
    df["track_id"] = df["track_id"].astype(int).map(
        lambda t: mapping.get(int(t), int(t)) + id_offset if int(t) in mapping
        else int(t)
    )
    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)
