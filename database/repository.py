from __future__ import annotations

import math
from typing import Any

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import AlertRecord, Base, ClipRecord, EventRecord, FaceRecord


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0 or math.isnan(denom):
        return 0.0
    return float(np.dot(va, vb) / denom)


class DatabaseRepository:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self.session_maker = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def add_event(self, payload: dict[str, Any]) -> EventRecord:
        event = EventRecord(**payload)
        with self.session_maker() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def update_event_clip_path(self, event_id: str, clip_path: str) -> bool:
        with self.session_maker() as session:
            event = session.get(EventRecord, event_id)
            if event is None:
                return False
            event.clip_path = clip_path
            session.commit()
            return True

    def list_events(self, limit: int = 200, camera_id: str | None = None) -> list[EventRecord]:
        with self.session_maker() as session:
            stmt = select(EventRecord).order_by(EventRecord.timestamp.desc()).limit(limit)
            if camera_id:
                stmt = stmt.where(EventRecord.camera_id == camera_id)
            return list(session.scalars(stmt).all())

    def semantic_search_events(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        camera_id: str | None = None,
        event_types: list[str] | None = None,
    ) -> list[tuple[float, EventRecord]]:
        with self.session_maker() as session:
            stmt = select(EventRecord).order_by(EventRecord.timestamp.desc()).limit(3000)
            if camera_id:
                stmt = stmt.where(EventRecord.camera_id == camera_id)
            if event_types:
                stmt = stmt.where(EventRecord.event_type.in_(event_types))
            rows = list(session.scalars(stmt).all())

        scored: list[tuple[float, EventRecord]] = []
        for row in rows:
            if not row.embedding:
                continue
            score = _cosine_similarity(query_embedding, row.embedding)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def add_face(self, name: str, embedding: list[float], image_path: str | None = None) -> FaceRecord:
        face = FaceRecord(name=name, embedding=embedding, image_path=image_path)
        with self.session_maker() as session:
            session.add(face)
            session.commit()
            session.refresh(face)
            return face

    def list_faces(self) -> list[FaceRecord]:
        with self.session_maker() as session:
            stmt = select(FaceRecord).order_by(FaceRecord.created_at.desc())
            return list(session.scalars(stmt).all())

    def get_face_records(self) -> list[FaceRecord]:
        return self.list_faces()

    def add_clip(self, event_id: str, camera_id: str, start_ts: float, end_ts: float, path: str) -> ClipRecord:
        clip = ClipRecord(event_id=event_id, camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, path=path)
        with self.session_maker() as session:
            session.add(clip)
            session.commit()
            session.refresh(clip)
            return clip

    def list_clips(self, limit: int = 200) -> list[ClipRecord]:
        with self.session_maker() as session:
            stmt = select(ClipRecord).order_by(ClipRecord.created_at.desc()).limit(limit)
            return list(session.scalars(stmt).all())

    def add_alert(self, event_id: str, camera_id: str, severity: str, message: str) -> AlertRecord:
        alert = AlertRecord(event_id=event_id, camera_id=camera_id, severity=severity, message=message)
        with self.session_maker() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return alert

    def list_alerts(self, limit: int = 200, acknowledged: bool | None = None) -> list[AlertRecord]:
        with self.session_maker() as session:
            stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(limit)
            if acknowledged is not None:
                stmt = stmt.where(AlertRecord.acknowledged == acknowledged)
            return list(session.scalars(stmt).all())

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self.session_maker() as session:
            alert = session.get(AlertRecord, alert_id)
            if alert is None:
                return False
            alert.acknowledged = True
            session.commit()
            return True
