from __future__ import annotations

from pathlib import Path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError("Install opencv-python-headless to use video commands.") from exc
    return cv2


def video_info(video_path: str | Path) -> dict:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "path": str(video_path),
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps else None,
        "width": width,
        "height": height,
    }


def extract_video_clip(
    video_path: str | Path,
    out_path: str | Path,
    *,
    start_frame: int,
    end_frame: int,
) -> Path:
    cv2 = require_cv2()
    if end_frame <= start_frame:
        raise ValueError(f"Invalid clip range: {start_frame}..{end_frame}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    return output


def sample_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    every_seconds: float | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    cv2 = require_cv2()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_stride = stride or max(1, int(round((every_seconds or 1.0) * fps)))
    saved: list[Path] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_stride == 0:
            out_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            if max_frames is not None and len(saved) >= max_frames:
                break
        frame_index += 1

    cap.release()
    return saved
