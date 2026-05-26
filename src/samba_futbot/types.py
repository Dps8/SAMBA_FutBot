from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(slots=True)
class Detection:
    frame_index: int
    class_name: str
    score: float
    box: Box
    prompt: str | None = None
    object_id: int | str | None = None
    track_id: int | None = None
    team: str | None = None
    mask_path: str | None = None
    area: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["box"] = list(self.box)
        cx, cy = self.centroid
        record["centroid_x"] = cx
        record["centroid_y"] = cy
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Detection":
        box_values = record.get("box")
        if not box_values or len(box_values) != 4:
            raise ValueError(f"Detection record has invalid box: {record}")
        fields = {
            "frame_index": int(record["frame_index"]),
            "class_name": str(record["class_name"]),
            "score": float(record.get("score", 1.0)),
            "box": tuple(float(v) for v in box_values),
            "prompt": record.get("prompt"),
            "object_id": record.get("object_id"),
            "track_id": record.get("track_id"),
            "team": record.get("team"),
            "mask_path": record.get("mask_path"),
            "area": record.get("area"),
            "extra": record.get("extra", {}),
        }
        return cls(**fields)


@dataclass(slots=True)
class Event:
    frame_index: int
    event_type: str
    description: str
    confidence: float
    actors: list[int | str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
