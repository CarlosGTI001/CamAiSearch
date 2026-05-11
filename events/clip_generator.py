from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ClipGenerator:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir, "clips")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_clip(
        self,
        event_id: str,
        camera_id: str,
        frames: Iterable[np.ndarray],
        fps: float,
    ) -> str | None:
        frames_list = list(frames)
        if not frames_list:
            return None
        height, width = frames_list[0].shape[:2]
        out_path = self.output_dir / f"{camera_id}_{event_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, fps), (width, height))
        for frame in frames_list:
            writer.write(frame)
        writer.release()
        return str(out_path)
