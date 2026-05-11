from __future__ import annotations

from dataclasses import dataclass, field
import math

from detector.types import Detection


@dataclass(slots=True)
class Track:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    speed_px_per_sec: float = 0.0
    missed: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    last_timestamp: float = 0.0


class TrackerManager:
    def __init__(self, max_distance: float = 120.0, max_missed: int = 25) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        matched_tracks: set[int] = set()
        unmatched_detections: list[Detection] = []

        for detection in detections:
            track_id = self._find_best_track(detection)
            if track_id is None:
                unmatched_detections.append(detection)
                continue
            self._update_track(self.tracks[track_id], detection, timestamp)
            matched_tracks.add(track_id)

        for detection in unmatched_detections:
            self._create_track(detection, timestamp)

        to_delete: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

        return list(self.tracks.values())

    def _find_best_track(self, detection: Detection) -> int | None:
        best_distance = math.inf
        best_track_id: int | None = None
        cx, cy = detection.center

        for track_id, track in self.tracks.items():
            if track.class_name != detection.class_name:
                continue
            tx, ty = track.centroid
            distance = math.dist((cx, cy), (tx, ty))
            if distance < best_distance and distance <= self.max_distance:
                best_distance = distance
                best_track_id = track_id
        return best_track_id

    def _create_track(self, detection: Detection, timestamp: float) -> None:
        cx, cy = detection.center
        self.tracks[self.next_track_id] = Track(
            track_id=self.next_track_id,
            class_name=detection.class_name,
            bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
            centroid=(cx, cy),
            history=[(cx, cy)],
            last_timestamp=timestamp,
        )
        self.next_track_id += 1

    def _update_track(self, track: Track, detection: Detection, timestamp: float) -> None:
        cx, cy = detection.center
        dt = max(1e-6, timestamp - track.last_timestamp)
        displacement = math.dist((cx, cy), track.centroid)
        speed = displacement / dt

        track.bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        track.centroid = (cx, cy)
        track.speed_px_per_sec = speed
        track.missed = 0
        track.last_timestamp = timestamp
        track.history.append((cx, cy))
        if len(track.history) > 40:
            track.history.pop(0)
