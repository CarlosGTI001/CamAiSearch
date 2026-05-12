from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.live_monitor import LiveMonitorService
from config.settings import AppSettings, CameraSettings, load_settings
from database.models import EventRecord
from database.repository import DatabaseRepository
from detector.yolo_detector import YOLODetector
from embeddings.clip_embedder import CLIPEmbedder
from events.clip_generator import ClipGenerator
from events.event_engine import EventEngine
from face.recognizer import FaceRecognizer
from llm.scene_describer import SceneDescriber
from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClipResponse,
    EventResponse,
    FaceRegisterRequest,
    FaceResponse,
    LiveEventResponse,
    SearchResponse,
)
from pipeline.analysis_pipeline import VideoAnalysisPipeline
from search.semantic_search import SemanticSearchService


class AppContainer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.repository = DatabaseRepository(settings.database_url)
        self.detector = YOLODetector(
            model_path=settings.models.yolo_model,
            confidence=settings.thresholds.detector_confidence,
            device=settings.models.device,
            fp16=settings.models.fp16,
            target_classes={"person", "car", "truck", "bus", "backpack", "box", "weapon"},
        )
        self.embedder = CLIPEmbedder(
            model_name=settings.models.clip_model,
            device=settings.models.device,
            fp16=settings.models.fp16,
        )
        self.face_recognizer = FaceRecognizer(
            repository=self.repository,
            model_name=settings.models.face_model,
            similarity_threshold=settings.thresholds.face_similarity,
            use_cuda=settings.models.device == "cuda",
        )
        self.describer = SceneDescriber(
            model_name=settings.models.blip_model,
            device=settings.models.device,
        )
        self.event_engine = EventEngine(
            running_speed_px_per_sec=settings.thresholds.running_speed_px_per_sec,
        )
        self.clip_generator = ClipGenerator(settings.output_dir)
        self.pipeline = VideoAnalysisPipeline(
            settings=settings,
            repository=self.repository,
            detector=self.detector,
            face_recognizer=self.face_recognizer,
            embedder=self.embedder,
            describer=self.describer,
            event_engine=self.event_engine,
            clip_generator=self.clip_generator,
        )
        self.search = SemanticSearchService(
            repository=self.repository,
            embedder=self.embedder,
            semantic_threshold=settings.thresholds.semantic_similarity,
        )
        self.live_monitor = LiveMonitorService(
            settings=settings,
            pipeline=self.pipeline,
            repository=self.repository,
        )
        self.camera_map: dict[str, CameraSettings] = {camera.camera_id: camera for camera in settings.cameras}
        self.analysis_jobs: dict[str, dict] = {}
        self.analysis_jobs_lock = threading.Lock()


def _event_to_response(event: EventRecord) -> EventResponse:
    return EventResponse(
        event_id=event.id,
        timestamp=event.timestamp,
        camera_id=event.camera_id,
        event_type=event.event_type,
        severity=event.severity,
        description=event.description,
        persons=event.persons or [],
        objects=event.objects or [],
        ai_description=event.ai_description,
        frame_path=event.frame_path,
        clip_path=event.clip_path,
        metadata=event.metadata_json or {},
    )


def create_app(config_path: str | Path = Path("config", "config.json")) -> FastAPI:
    settings = load_settings(config_path)
    container = AppContainer(settings)
    app = FastAPI(title="CamAiSearch API", version="1.0.0")
    app.state.container = container

    static_dir = Path("api", "static")
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def startup() -> None:
        await container.live_monitor.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await container.live_monitor.stop()

    @app.get("/")
    async def dashboard() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard no encontrado")
        return FileResponse(index_path)

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
        camera_id = payload.camera_id or f"source_{Path(payload.source).stem or 'stream'}"
        camera_settings = container.camera_map.get(camera_id)
        result = await asyncio.to_thread(
            container.pipeline.analyze_video,
            payload.source,
            camera_id,
            payload.max_seconds,
            payload.save_clips,
            camera_settings,
        )
        return AnalyzeResponse(**result)

    @app.post("/upload-video")
    async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
        videos_dir = Path(container.settings.videos_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{int(time.time() * 1000)}_{filename}"
        output_path = videos_dir / output_name

        size_bytes = 0
        with output_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                out_file.write(chunk)
        await file.close()
        return JSONResponse(
            {
                "filename": output_name,
                "path": str(output_path),
                "size_bytes": size_bytes,
            }
        )

    def _run_analysis_job(
        job_id: str,
        source: str,
        camera_id: str,
        max_seconds: int | None,
        save_clips: bool,
        camera_settings: CameraSettings | None,
    ) -> None:
        def update_progress(
            processed_frames: int,
            estimated_total_frames: int | None,
            events_detected: int,
            clips_generated: int,
        ) -> None:
            progress_percent = None
            if estimated_total_frames and estimated_total_frames > 0:
                progress_percent = round(min(100.0, (processed_frames / estimated_total_frames) * 100.0), 2)
            with container.analysis_jobs_lock:
                job = container.analysis_jobs.get(job_id)
                if job is None:
                    return
                job.update(
                    {
                        "processed_frames": processed_frames,
                        "estimated_total_frames": estimated_total_frames,
                        "events_detected": events_detected,
                        "clips_generated": clips_generated,
                        "progress_percent": progress_percent,
                        "updated_at": time.time(),
                    }
                )

        with container.analysis_jobs_lock:
            job = container.analysis_jobs[job_id]
            job["status"] = "running"
            job["started_at"] = time.time()
            job["updated_at"] = time.time()

        try:
            result = container.pipeline.analyze_video(
                source=source,
                camera_id=camera_id,
                max_seconds=max_seconds,
                save_clips=save_clips,
                camera_settings=camera_settings,
                progress_callback=update_progress,
            )
            with container.analysis_jobs_lock:
                job = container.analysis_jobs[job_id]
                job["status"] = "completed"
                job["result"] = result
                job["error"] = None
                job["finished_at"] = time.time()
                job["updated_at"] = time.time()
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
            with container.analysis_jobs_lock:
                job = container.analysis_jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                job["updated_at"] = time.time()

    @app.post("/analyze-async")
    async def analyze_async(payload: AnalyzeRequest) -> JSONResponse:
        camera_id = payload.camera_id or f"source_{Path(payload.source).stem or 'stream'}"
        camera_settings = container.camera_map.get(camera_id)
        job_id = uuid.uuid4().hex
        created_at = time.time()
        with container.analysis_jobs_lock:
            container.analysis_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "source": payload.source,
                "camera_id": camera_id,
                "max_seconds": payload.max_seconds,
                "save_clips": payload.save_clips,
                "processed_frames": 0,
                "estimated_total_frames": None,
                "events_detected": 0,
                "clips_generated": 0,
                "progress_percent": 0.0,
                "result": None,
                "error": None,
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "updated_at": created_at,
            }
        thread = threading.Thread(
            target=_run_analysis_job,
            args=(
                job_id,
                payload.source,
                camera_id,
                payload.max_seconds,
                payload.save_clips,
                camera_settings,
            ),
            daemon=True,
            name=f"analyze-job-{job_id[:8]}",
        )
        thread.start()
        return JSONResponse({"job_id": job_id, "status": "queued", "camera_id": camera_id})

    @app.get("/analyze-jobs/{job_id}")
    async def analyze_job_status(job_id: str) -> JSONResponse:
        with container.analysis_jobs_lock:
            job = container.analysis_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No existe el job: {job_id}")
            return JSONResponse(job)

    @app.get("/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., min_length=2),
        top_k: int = Query(20, ge=1, le=100),
        camera_id: str | None = None,
    ) -> SearchResponse:
        results = container.search.search(q, top_k=top_k, camera_id=camera_id)
        items = [_event_to_response(event) for _, event in results]
        return SearchResponse(query=q, total=len(items), results=items)

    @app.get("/events", response_model=list[EventResponse])
    async def events(limit: int = Query(200, ge=1, le=1000), camera_id: str | None = None) -> list[EventResponse]:
        data = container.repository.list_events(limit=limit, camera_id=camera_id)
        return [_event_to_response(event) for event in data]

    @app.get("/faces", response_model=list[FaceResponse])
    async def faces() -> list[FaceResponse]:
        data = container.repository.list_faces()
        return [FaceResponse(face_id=face.id, name=face.name, image_path=face.image_path) for face in data]

    @app.post("/faces", response_model=FaceResponse)
    async def register_face(payload: FaceRegisterRequest) -> FaceResponse:
        try:
            face_id = await asyncio.to_thread(container.face_recognizer.register_face, payload.name, payload.image_path)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FaceResponse(face_id=face_id, name=payload.name, image_path=payload.image_path)

    @app.get("/clips", response_model=list[ClipResponse])
    async def clips(limit: int = Query(200, ge=1, le=1000)) -> list[ClipResponse]:
        rows = container.repository.list_clips(limit=limit)
        return [
            ClipResponse(
                clip_id=row.id,
                event_id=row.event_id,
                camera_id=row.camera_id,
                path=row.path,
                start_ts=row.start_ts,
                end_ts=row.end_ts,
            )
            for row in rows
        ]

    @app.get("/alerts")
    async def alerts(
        limit: int = Query(200, ge=1, le=1000),
        acknowledged: bool | None = None,
    ) -> JSONResponse:
        rows = container.repository.list_alerts(limit=limit, acknowledged=acknowledged)
        return JSONResponse(
            [
                {
                    "alert_id": row.id,
                    "event_id": row.event_id,
                    "camera_id": row.camera_id,
                    "severity": row.severity,
                    "message": row.message,
                    "acknowledged": row.acknowledged,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
        )

    @app.get("/live-events", response_model=list[LiveEventResponse])
    async def live_events(limit: int = Query(100, ge=1, le=500)) -> list[LiveEventResponse]:
        items = container.live_monitor.get_live_events(limit=limit)
        return [
            LiveEventResponse(
                event_id=item.event_id,
                camera_id=item.camera_id,
                event_type=item.event_type,
                severity=item.severity,
                description=item.description,
                timestamp=item.timestamp,
            )
            for item in items
        ]

    @app.get("/cameras")
    async def cameras() -> JSONResponse:
        return JSONResponse(container.live_monitor.camera_status())

    @app.get("/snapshot")
    async def snapshot(camera_id: str = Query(...)) -> JSONResponse:
        try:
            path = container.live_monitor.get_snapshot(camera_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"camera_id": camera_id, "snapshot_path": path})

    @app.get("/recordings")
    async def recordings() -> JSONResponse:
        recording_dir = Path(container.settings.output_dir, "recordings")
        recording_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(str(path) for path in recording_dir.glob("*.mp4"))
        return JSONResponse({"count": len(files), "items": files})

    @app.get("/heatmap/{camera_id}")
    async def heatmap(camera_id: str) -> JSONResponse:
        try:
            path = container.pipeline.export_heatmap(camera_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"camera_id": camera_id, "heatmap_path": path})

    @app.get("/stream/{camera_id}")
    async def stream(camera_id: str) -> StreamingResponse:
        async def frame_generator():
            while True:
                frame = container.live_monitor.get_latest_jpeg(camera_id)
                if frame is None:
                    await asyncio.sleep(0.08)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                await asyncio.sleep(0.03)

        return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/live-events")
    async def ws_live_events(ws: WebSocket) -> None:
        await ws.accept()
        sent_ids: set[str] = set()
        try:
            while True:
                events_batch = container.live_monitor.get_live_events(limit=250)
                for item in events_batch:
                    if item.event_id in sent_ids:
                        continue
                    await ws.send_json(
                        {
                            "event_id": item.event_id,
                            "camera_id": item.camera_id,
                            "event_type": item.event_type,
                            "severity": item.severity,
                            "description": item.description,
                            "timestamp": item.timestamp,
                        }
                    )
                    sent_ids.add(item.event_id)
                if len(sent_ids) > 5000:
                    sent_ids.clear()
                await asyncio.sleep(0.4)
        except WebSocketDisconnect:
            return

    return app
