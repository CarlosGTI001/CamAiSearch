from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from config.settings import AppSettings, CameraSettings
from database.models import EventRecord
from database.repository import DatabaseRepository
from pipeline.analysis_pipeline import VideoAnalysisPipeline


@dataclass(slots=True)
class LiveEventItem:
    event_id: str
    camera_id: str
    event_type: str
    severity: str
    description: str
    timestamp: float


class LiveMonitorService:
    def __init__(
        self,
        settings: AppSettings,
        pipeline: VideoAnalysisPipeline,
        repository: DatabaseRepository,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.repository = repository
        self.stop_event = threading.Event()
        self.threads: dict[str, threading.Thread] = {}
        self.latest_jpeg: dict[str, bytes] = {}
        self.latest_frame: dict[str, np.ndarray] = {}
        self.frame_buffers: dict[str, deque[np.ndarray]] = {}
        self.live_events: deque[LiveEventItem] = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.started = False

    async def start(self) -> None:
        if self.started:
            return
        self.stop_event.clear()
        for camera in self.settings.cameras:
            if not camera.enabled:
                continue
            self.frame_buffers[camera.camera_id] = deque(
                maxlen=max(1, self.settings.runtime.analyze_fps * self.settings.runtime.buffer_seconds)
            )
            thread = threading.Thread(
                target=self._camera_loop,
                args=(camera,),
                daemon=True,
                name=f"camera-worker-{camera.camera_id}",
            )
            thread.start()
            self.threads[camera.camera_id] = thread
        self.started = True
        await asyncio.sleep(0)

    async def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads.values():
            thread.join(timeout=2.0)
        self.threads.clear()
        self.started = False
        await asyncio.sleep(0)

    def camera_status(self) -> list[dict]:
        statuses = []
        for camera in self.settings.cameras:
            statuses.append(
                {
                    "camera_id": camera.camera_id,
                    "name": camera.name,
                    "source": camera.source,
                    "running": camera.camera_id in self.threads and self.threads[camera.camera_id].is_alive(),
                }
            )
        return statuses

    def get_latest_jpeg(self, camera_id: str) -> bytes | None:
        with self.lock:
            return self.latest_jpeg.get(camera_id)

    def get_live_events(self, limit: int = 100) -> list[LiveEventItem]:
        with self.lock:
            return list(self.live_events)[-limit:]

    def get_snapshot(self, camera_id: str) -> str:
        with self.lock:
            frame = self.latest_frame.get(camera_id)
            if frame is None:
                raise ValueError(f"No hay frame disponible para la cámara: {camera_id}")
            out_dir = Path(self.settings.output_dir, "frames")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"snapshot_{camera_id}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(out_path), frame)
            return str(out_path)

    def _camera_loop(self, camera: CameraSettings) -> None:
        source = int(camera.source) if camera.source.isdigit() else camera.source
        reconnect_delay = self.settings.runtime.reconnect_seconds

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                time.sleep(reconnect_delay)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 240:
                fps = 25.0
            analyze_fps = max(1, self.settings.runtime.analyze_fps)
            sample_interval = max(1, int(round(fps / analyze_fps)))
            frame_idx = 0
            zones = [zone.__dict__ for zone in camera.zones]
            lines = [line.__dict__ for line in camera.lines]

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                self._update_latest(camera.camera_id, frame)
                self.frame_buffers[camera.camera_id].append(frame.copy())

                if frame_idx % sample_interval == 0:
                    timestamp = time.time()
                    result = self.pipeline.process_frame(
                        camera_id=camera.camera_id,
                        frame=frame,
                        timestamp=timestamp,
                        zones=zones,
                        lines=lines,
                    )
                    self._update_latest(camera.camera_id, result.annotated_frame)
                    self.frame_buffers[camera.camera_id].append(result.annotated_frame.copy())
                    self._register_live_events(result.event_records)
                    self._generate_live_clips(camera.camera_id, result.event_records, analyze_fps, timestamp)
                frame_idx += 1
            cap.release()
            time.sleep(reconnect_delay)

    def _generate_live_clips(
        self,
        camera_id: str,
        events: list[EventRecord],
        fps: int,
        timestamp: float,
    ) -> None:
        for event in events:
            if event.event_type not in {
                "person_recognized",
                "intrusion",
                "person_running",
                "abandoned_object",
                "suspicious_activity",
            }:
                continue
            clip_path = self.pipeline.clip_generator.write_clip(
                event_id=event.id,
                camera_id=camera_id,
                frames=list(self.frame_buffers[camera_id]),
                fps=float(fps),
            )
            if not clip_path:
                continue
            self.repository.add_clip(
                event_id=event.id,
                camera_id=camera_id,
                start_ts=max(0.0, timestamp - len(self.frame_buffers[camera_id]) / fps),
                end_ts=timestamp,
                path=clip_path,
            )
            self.repository.update_event_clip_path(event.id, clip_path)

    def _register_live_events(self, events: list[EventRecord]) -> None:
        with self.lock:
            for event in events:
                self.live_events.append(
                    LiveEventItem(
                        event_id=event.id,
                        camera_id=event.camera_id,
                        event_type=event.event_type,
                        severity=event.severity,
                        description=event.description,
                        timestamp=event.timestamp,
                    )
                )

    def _update_latest(self, camera_id: str, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return
        with self.lock:
            self.latest_frame[camera_id] = frame.copy()
            self.latest_jpeg[camera_id] = encoded.tobytes()
