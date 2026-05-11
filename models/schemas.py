from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    source: str = Field(..., description="Ruta de video, URL RTSP o ID de cámara")
    camera_id: str | None = Field(default=None)
    max_seconds: int | None = Field(default=None, ge=1, le=7200)
    save_clips: bool = True


class AnalyzeResponse(BaseModel):
    status: str
    camera_id: str
    total_frames: int
    events_detected: int
    clips_generated: int


class EventResponse(BaseModel):
    event_id: str
    timestamp: float
    camera_id: str
    event_type: str
    severity: str
    description: str
    persons: list[str]
    objects: list[str]
    ai_description: str
    frame_path: str | None = None
    clip_path: str | None = None
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[EventResponse]


class FaceRegisterRequest(BaseModel):
    name: str
    image_path: str


class FaceResponse(BaseModel):
    face_id: str
    name: str
    image_path: str | None = None


class ClipResponse(BaseModel):
    clip_id: str
    event_id: str
    camera_id: str
    path: str
    start_ts: float
    end_ts: float


class CameraStatus(BaseModel):
    camera_id: str
    name: str
    running: bool
    source: str


class LiveEventResponse(BaseModel):
    event_id: str
    camera_id: str
    event_type: str
    severity: str
    description: str
    timestamp: float
