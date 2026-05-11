from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import uuid

import cv2
import numpy as np

from config.settings import AppSettings, CameraSettings
from database.models import EventRecord
from database.repository import DatabaseRepository
from detector.types import Detection
from detector.yolo_detector import YOLODetector
from embeddings.clip_embedder import CLIPEmbedder
from events.clip_generator import ClipGenerator
from events.event_engine import EventEngine
from face.recognizer import FaceRecognizer
from llm.scene_describer import SceneDescriber
from tracking.centroid_tracker import TrackerManager


@dataclass(slots=True)
class ProcessedFrame:
    camera_id: str
    timestamp: float
    detections: list[Detection]
    tracks: list
    recognized_faces: list[dict]
    event_records: list[EventRecord]
    annotated_frame: np.ndarray
    image_embedding: list[float]


class VideoAnalysisPipeline:
    def __init__(
        self,
        settings: AppSettings,
        repository: DatabaseRepository,
        detector: YOLODetector,
        face_recognizer: FaceRecognizer,
        embedder: CLIPEmbedder,
        describer: SceneDescriber,
        event_engine: EventEngine,
        clip_generator: ClipGenerator,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.detector = detector
        self.face_recognizer = face_recognizer
        self.embedder = embedder
        self.describer = describer
        self.event_engine = event_engine
        self.clip_generator = clip_generator
        self.trackers: dict[str, TrackerManager] = {}
        self.heatmaps: dict[str, np.ndarray] = {}
        self.frames_dir = Path(settings.output_dir, "frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
        zones: list[dict] | None = None,
        lines: list[dict] | None = None,
    ) -> ProcessedFrame:
        detections = self.detector.detect(frame)
        tracker = self.trackers.setdefault(
            camera_id,
            TrackerManager(max_distance=self.settings.thresholds.tracking_max_distance),
        )
        tracks = tracker.update(detections, timestamp=timestamp)
        self._update_heatmap(camera_id, frame.shape[:2], tracks)
        recognized_faces = self.face_recognizer.recognize(frame)
        events = self.event_engine.infer_events(
            camera_id=camera_id,
            timestamp=timestamp,
            detections=detections,
            tracks=tracks,
            recognized_faces=recognized_faces,
            zones=zones or [],
            lines=lines or [],
        )
        descriptions = [event["description"] for event in events]
        ai_description = self.describer.describe(frame, detections, recognized_faces, descriptions)
        image_embedding = self.embedder.embed_image(frame)

        event_records: list[EventRecord] = []
        frame_path = self._write_snapshot(camera_id, timestamp, frame) if events else None
        for event in events:
            payload = {
                **event,
                "ai_description": ai_description,
                "embedding": image_embedding,
                "frame_path": frame_path,
            }
            stored = self.repository.add_event(payload)
            event_records.append(stored)
            if event["severity"] in {"high", "critical"}:
                self.repository.add_alert(
                    event_id=stored.id,
                    camera_id=camera_id,
                    severity=event["severity"],
                    message=event["description"],
                )

        annotated = self.annotate_frame(frame.copy(), detections, tracks, recognized_faces)
        return ProcessedFrame(
            camera_id=camera_id,
            timestamp=timestamp,
            detections=detections,
            tracks=tracks,
            recognized_faces=recognized_faces,
            event_records=event_records,
            annotated_frame=annotated,
            image_embedding=image_embedding,
        )

    def analyze_video(
        self,
        source: str,
        camera_id: str,
        max_seconds: int | None = None,
        save_clips: bool = True,
        camera_settings: CameraSettings | None = None,
    ) -> dict:
        source_value = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            raise RuntimeError(f"No fue posible abrir el video/stream: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = float(self.settings.runtime.analyze_fps)
        analysis_fps = max(1, self.settings.runtime.analyze_fps)
        sample_interval = max(1, int(round(fps / analysis_fps)))
        max_frames = int(max_seconds * fps) if max_seconds else None
        frame_buffer: deque[np.ndarray] = deque(maxlen=max(analysis_fps * self.settings.runtime.buffer_seconds, 1))

        total_frames = 0
        events_detected = 0
        clips_generated = 0
        zones = [zone.__dict__ for zone in (camera_settings.zones if camera_settings else [])]
        lines = [line.__dict__ for line in (camera_settings.lines if camera_settings else [])]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame.copy())
            if max_frames is not None and total_frames >= max_frames:
                break

            if total_frames % sample_interval == 0:
                ts = total_frames / fps
                result = self.process_frame(camera_id, frame, ts, zones=zones, lines=lines)
                events_detected += len(result.event_records)
                if save_clips:
                    for event in result.event_records:
                        if event.event_type not in {
                            "person_running",
                            "intrusion",
                            "abandoned_object",
                            "suspicious_activity",
                            "person_recognized",
                        }:
                            continue
                        clip_path = self.clip_generator.write_clip(
                            event_id=event.id,
                            camera_id=camera_id,
                            frames=frame_buffer,
                            fps=analysis_fps,
                        )
                        if clip_path:
                            self.repository.add_clip(
                                event_id=event.id,
                                camera_id=camera_id,
                                start_ts=max(0.0, ts - len(frame_buffer) / analysis_fps),
                                end_ts=ts,
                                path=clip_path,
                            )
                            self.repository.update_event_clip_path(event.id, clip_path)
                            clips_generated += 1
            total_frames += 1

        cap.release()
        return {
            "status": "ok",
            "camera_id": camera_id,
            "total_frames": total_frames,
            "events_detected": events_detected,
            "clips_generated": clips_generated,
        }

    def annotate_frame(self, frame: np.ndarray, detections: list[Detection], tracks: list, faces: list[dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.as_xyxy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 255, 140), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 255, 140), 2)

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.putText(
                frame,
                f"ID {track.track_id} v={track.speed_px_per_sec:.1f}",
                (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 10),
                2,
            )

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 40, 40), 2)
            cv2.putText(
                frame,
                f"{face['name']} ({face['score']:.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 40, 40),
                2,
            )
        return frame

    def _write_snapshot(self, camera_id: str, timestamp: float, frame: np.ndarray) -> str:
        frame_name = f"{camera_id}_{int(timestamp * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = self.frames_dir / frame_name
        cv2.imwrite(str(out_path), frame)
        return str(out_path)

    def export_heatmap(self, camera_id: str) -> str:
        if camera_id not in self.heatmaps:
            raise ValueError(f"No hay heatmap disponible para la cámara: {camera_id}")
        heatmap = self.heatmaps[camera_id]
        normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        out_path = Path(self.settings.output_dir, "frames", f"heatmap_{camera_id}.jpg")
        cv2.imwrite(str(out_path), colored)
        return str(out_path)

    def _update_heatmap(self, camera_id: str, shape: tuple[int, int], tracks: list) -> None:
        h, w = shape
        heatmap = self.heatmaps.get(camera_id)
        if heatmap is None or heatmap.shape != (h, w):
            heatmap = np.zeros((h, w), dtype=np.float32)
            self.heatmaps[camera_id] = heatmap
        for track in tracks:
            if track.class_name != "person":
                continue
            x, y = [int(v) for v in track.centroid]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap, (x, y), 22, 1.0, -1)
