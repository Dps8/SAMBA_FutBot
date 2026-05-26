from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .types import Detection, Event


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, data: object) -> None:
    output = ensure_parent(path)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_detections(path: str | Path) -> list[Detection]:
    return [Detection.from_record(record) for record in read_jsonl(path)]


def write_detections(path: str | Path, detections: Iterable[Detection]) -> None:
    write_jsonl(path, (det.to_record() for det in detections))


def write_events(path: str | Path, events: Iterable[Event]) -> None:
    write_json(path, [event.to_record() for event in events])
