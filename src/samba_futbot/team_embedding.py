from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .play_state import ROBOT_CLASSES
from .team import _crop_detection
from .types import Detection
from .video import require_cv2


def cluster_track_embeddings(
    track_embeddings: Mapping[int, np.ndarray],
    *,
    max_iterations: int = 50,
) -> dict[int, int]:
    """Split track-level appearance embeddings into two deterministic groups."""
    track_ids = sorted(track_embeddings)
    if len(track_ids) < 2:
        return {track_id: 0 for track_id in track_ids}
    matrix = np.stack([_unit_vector(track_embeddings[track_id]) for track_id in track_ids])
    center_a = matrix[0]
    center_b = matrix[int(np.argmax(np.linalg.norm(matrix - center_a, axis=1)))]
    labels = np.zeros(len(track_ids), dtype=int)

    for _ in range(max_iterations):
        distances_a = np.linalg.norm(matrix - center_a, axis=1)
        distances_b = np.linalg.norm(matrix - center_b, axis=1)
        updated = (distances_b < distances_a).astype(int)
        if np.array_equal(updated, labels) and np.any(updated == 1):
            break
        labels = updated
        if not np.any(labels == 0) or not np.any(labels == 1):
            farthest = int(np.argmax(distances_a))
            labels[farthest] = 1
        center_a = _unit_vector(np.mean(matrix[labels == 0], axis=0))
        center_b = _unit_vector(np.mean(matrix[labels == 1], axis=0))
    return {track_id: int(label) for track_id, label in zip(track_ids, labels, strict=True)}


def align_clusters_to_teams(
    cluster_by_track: Mapping[int, int],
    existing_team_by_track: Mapping[int, str],
    *,
    teams: tuple[str, str] = ("blue", "yellow"),
) -> tuple[dict[int, str], dict]:
    """Choose the one-to-one cluster/team mapping that best agrees with HSV votes."""
    mappings = ({0: teams[0], 1: teams[1]}, {0: teams[1], 1: teams[0]})
    scores = []
    for mapping in mappings:
        matches = 0
        evidence = 0
        for track_id, cluster in cluster_by_track.items():
            existing = existing_team_by_track.get(track_id)
            if existing not in teams:
                continue
            evidence += 1
            matches += int(mapping.get(cluster) == existing)
        scores.append(matches)
    best_index = int(np.argmax(np.asarray(scores)))
    return mappings[best_index], {
        "mapping_scores": scores,
        "color_evidence_tracks": sum(
            team in teams for team in existing_team_by_track.values()
        ),
        "mapping_ambiguous": scores[0] == scores[1],
    }


def assign_embedding_teams(
    detections: Iterable[Detection],
    cluster_by_track: Mapping[int, int],
    cluster_to_team: Mapping[int, str],
) -> list[Detection]:
    assigned = list(detections)
    for detection in assigned:
        if detection.class_name not in ROBOT_CLASSES or detection.track_id is None:
            continue
        cluster = cluster_by_track.get(detection.track_id)
        if cluster is not None and cluster in cluster_to_team:
            detection.team = cluster_to_team[cluster]
    return assigned


def extract_dinov2_track_embeddings(
    video_path: str | Path,
    detections: Iterable[Detection],
    *,
    model_id: str = "facebook/dinov2-small",
    samples_per_track: int = 8,
    min_frame_gap: int = 10,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Embedding team analysis requires torch, Pillow and transformers."
        ) from exc

    selected = _select_track_samples(
        detections,
        samples_per_track=samples_per_track,
        min_frame_gap=min_frame_gap,
    )
    if not selected:
        return {}, {}
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(resolved_device).eval()
    crops = _read_selected_crops(video_path, selected)
    vectors: dict[int, list[np.ndarray]] = defaultdict(list)

    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        images = [Image.fromarray(crop) for _, crop in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0].detach().float().cpu().numpy()
        for (track_id, _), embedding in zip(batch, embeddings, strict=True):
            vectors[track_id].append(_unit_vector(embedding))

    averaged = {
        track_id: _unit_vector(np.mean(items, axis=0))
        for track_id, items in vectors.items()
        if items
    }
    return averaged, {track_id: len(items) for track_id, items in vectors.items()}


def embedding_team_report(
    track_embeddings: Mapping[int, np.ndarray],
    samples_by_track: Mapping[int, int],
    cluster_by_track: Mapping[int, int],
    cluster_to_team: Mapping[int, str],
    mapping_metadata: Mapping,
    *,
    model_id: str,
) -> dict:
    cluster_counts = Counter(cluster_by_track.values())
    return {
        "schema": "samba_futbot.embedding_team_analysis.v1",
        "model_id": model_id,
        "tracks_embedded": len(track_embeddings),
        "embedding_dimensions": (
            int(next(iter(track_embeddings.values())).shape[0]) if track_embeddings else 0
        ),
        "samples_by_track": {str(key): int(value) for key, value in samples_by_track.items()},
        "cluster_by_track": {str(key): int(value) for key, value in cluster_by_track.items()},
        "cluster_counts": {str(key): int(value) for key, value in cluster_counts.items()},
        "cluster_to_team": {str(key): value for key, value in cluster_to_team.items()},
        "mapping": dict(mapping_metadata),
    }


def _select_track_samples(
    detections: Iterable[Detection],
    *,
    samples_per_track: int,
    min_frame_gap: int,
) -> dict[int, list[Detection]]:
    selected: dict[int, list[Detection]] = defaultdict(list)
    last_frame: dict[int, int] = {}
    robots = sorted(
        (
            detection
            for detection in detections
            if detection.class_name in ROBOT_CLASSES and detection.track_id is not None
        ),
        key=lambda item: (item.frame_index, item.track_id or -1),
    )
    for detection in robots:
        track_id = int(detection.track_id)
        if len(selected[track_id]) >= samples_per_track:
            continue
        if detection.frame_index - last_frame.get(track_id, -min_frame_gap) < min_frame_gap:
            continue
        selected[track_id].append(detection)
        last_frame[track_id] = detection.frame_index
    return dict(selected)


def _read_selected_crops(
    video_path: str | Path,
    selected: Mapping[int, list[Detection]],
) -> list[tuple[int, np.ndarray]]:
    cv2 = require_cv2()
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detections in selected.values():
        for detection in detections:
            by_frame[detection.frame_index].append(detection)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    crops: list[tuple[int, np.ndarray]] = []
    frame_index = 0
    wanted = set(by_frame)
    while wanted:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index in wanted:
            frame_rgb = frame_bgr[:, :, ::-1]
            for detection in by_frame[frame_index]:
                crop = _crop_detection(frame_rgb, detection)
                if crop.size and detection.track_id is not None:
                    crops.append((int(detection.track_id), crop.copy()))
            wanted.remove(frame_index)
        frame_index += 1
    cap.release()
    return crops


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector
