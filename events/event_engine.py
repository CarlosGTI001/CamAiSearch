from __future__ import annotations

import math

import cv2

from detector.types import Detection
from tracking.centroid_tracker import Track


class EventEngine:
    def __init__(self, running_speed_px_per_sec: float) -> None:
        self.running_speed_px_per_sec = running_speed_px_per_sec
        self.last_emitted: dict[tuple[str, str, int], float] = {}
        self.track_line_side: dict[tuple[str, str, int], float] = {}
        self.abandoned_started_at: dict[tuple[str, int], float] = {}
        self.abandoned_emitted: set[tuple[str, int]] = set()

    def infer_events(
        self,
        camera_id: str,
        timestamp: float,
        detections: list[Detection],
        tracks: list[Track],
        recognized_faces: list[dict],
        zones: list[dict],
        lines: list[dict],
    ) -> list[dict]:
        events: list[dict] = []
        person_tracks = [track for track in tracks if track.class_name == "person"]

        if person_tracks:
            events.extend(
                self._emit_with_cooldown(
                    camera_id,
                    timestamp,
                    event_type="person_detected",
                    severity="info",
                    description=f"{len(person_tracks)} persona(s) detectadas",
                    track_id=0,
                    persons=[],
                    objects=[],
                    metadata={"count": len(person_tracks)},
                    cooldown_seconds=2.0,
                )
            )

        for face in recognized_faces:
            if face["name"] == "unknown":
                continue
            events.extend(
                self._emit_with_cooldown(
                    camera_id,
                    timestamp,
                    event_type="person_recognized",
                    severity="info",
                    description=f"Persona reconocida: {face['name']}",
                    track_id=-1,
                    persons=[face["name"]],
                    objects=[],
                    metadata={"score": face["score"]},
                    cooldown_seconds=3.0,
                )
            )

        for track in person_tracks:
            if track.speed_px_per_sec >= self.running_speed_px_per_sec:
                events.extend(
                    self._emit_with_cooldown(
                        camera_id,
                        timestamp,
                        event_type="person_running",
                        severity="medium",
                        description=f"Persona corriendo (track {track.track_id})",
                        track_id=track.track_id,
                        persons=[],
                        objects=[],
                        metadata={"speed_px_per_sec": track.speed_px_per_sec},
                        cooldown_seconds=2.0,
                    )
                )

        for track in person_tracks:
            cx, cy = track.centroid
            for zone in zones:
                if zone.get("type") != "intrusion":
                    continue
                polygon = zone.get("polygon", [])
                if self._inside_polygon((cx, cy), polygon):
                    events.extend(
                        self._emit_with_cooldown(
                            camera_id,
                            timestamp,
                            event_type="intrusion",
                            severity="high",
                            description=f"Intrusión en zona {zone.get('name', 'zona')}",
                            track_id=track.track_id,
                            persons=[],
                            objects=[],
                            metadata={"zone": zone.get("name")},
                            cooldown_seconds=3.0,
                        )
                    )

        for track in person_tracks:
            for line in lines:
                crossed = self._check_line_crossing(camera_id, track.track_id, track.centroid, line)
                if crossed:
                    events.extend(
                        self._emit_with_cooldown(
                            camera_id,
                            timestamp,
                            event_type="line_crossing",
                            severity="medium",
                            description=f"Cruce de línea {line.get('name', 'line')}",
                            track_id=track.track_id,
                            persons=[],
                            objects=[],
                            metadata={"line": line.get("name")},
                            cooldown_seconds=2.0,
                        )
                    )

        weapon_detected = any(det.class_name in {"weapon", "knife", "gun"} for det in detections)
        if weapon_detected:
            events.extend(
                self._emit_with_cooldown(
                    camera_id,
                    timestamp,
                    event_type="suspicious_activity",
                    severity="critical",
                    description="Objeto potencialmente peligroso detectado",
                    track_id=0,
                    persons=[],
                    objects=["weapon"],
                    metadata={},
                    cooldown_seconds=2.0,
                )
            )

        events.extend(self._infer_abandoned_objects(camera_id, timestamp, tracks, person_tracks))
        return events

    def _infer_abandoned_objects(
        self,
        camera_id: str,
        timestamp: float,
        tracks: list[Track],
        person_tracks: list[Track],
    ) -> list[dict]:
        events: list[dict] = []
        person_centroids = [track.centroid for track in person_tracks]
        for track in tracks:
            if track.class_name in {"person", "car", "bus", "truck"}:
                continue
            key = (camera_id, track.track_id)
            nearest = self._nearest_distance(track.centroid, person_centroids)
            if nearest > 130.0:
                self.abandoned_started_at.setdefault(key, timestamp)
                elapsed = timestamp - self.abandoned_started_at[key]
                if elapsed >= 8.0 and key not in self.abandoned_emitted:
                    events.extend(
                        self._emit_with_cooldown(
                            camera_id,
                            timestamp,
                            event_type="abandoned_object",
                            severity="high",
                            description=f"Objeto abandonado detectado ({track.class_name})",
                            track_id=track.track_id,
                            persons=[],
                            objects=[track.class_name],
                            metadata={"elapsed_seconds": round(elapsed, 2)},
                            cooldown_seconds=6.0,
                        )
                    )
                    self.abandoned_emitted.add(key)
            else:
                if key in self.abandoned_started_at:
                    del self.abandoned_started_at[key]
                self.abandoned_emitted.discard(key)
        return events

    def _nearest_distance(self, source: tuple[float, float], targets: list[tuple[float, float]]) -> float:
        if not targets:
            return float("inf")
        return min(math.dist(source, target) for target in targets)

    def _inside_polygon(self, point: tuple[float, float], polygon: list[list[int]]) -> bool:
        if len(polygon) < 3:
            return False
        contour = [tuple(p) for p in polygon]
        return cv2.pointPolygonTest(np_array(contour), point, False) >= 0

    def _check_line_crossing(self, camera_id: str, track_id: int, centroid: tuple[float, float], line: dict) -> bool:
        p1 = tuple(line.get("p1", [0, 0]))
        p2 = tuple(line.get("p2", [0, 0]))
        side = self._point_line_side(centroid, p1, p2)
        key = (camera_id, line.get("name", "line"), track_id)
        prev_side = self.track_line_side.get(key)
        self.track_line_side[key] = side
        if prev_side is None:
            return False
        return side * prev_side < 0

    def _point_line_side(
        self,
        point: tuple[float, float],
        line_p1: tuple[int, int],
        line_p2: tuple[int, int],
    ) -> float:
        return (line_p2[0] - line_p1[0]) * (point[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (
            point[0] - line_p1[0]
        )

    def _emit_with_cooldown(
        self,
        camera_id: str,
        timestamp: float,
        event_type: str,
        severity: str,
        description: str,
        track_id: int,
        persons: list[str],
        objects: list[str],
        metadata: dict,
        cooldown_seconds: float,
    ) -> list[dict]:
        key = (camera_id, event_type, track_id)
        last_ts = self.last_emitted.get(key)
        if last_ts is not None and (timestamp - last_ts) < cooldown_seconds:
            return []
        self.last_emitted[key] = timestamp
        return [
            {
                "timestamp": timestamp,
                "camera_id": camera_id,
                "event_type": event_type,
                "severity": severity,
                "description": description,
                "persons": persons,
                "objects": objects,
                "metadata_json": metadata,
            }
        ]


def np_array(points: list[tuple[int, int]]):
    import numpy as np

    return np.array(points, dtype=np.int32)
