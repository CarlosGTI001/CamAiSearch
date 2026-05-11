from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ZoneSettings:
    name: str
    type: str
    polygon: list[list[int]]


@dataclass(slots=True)
class LineSettings:
    name: str
    p1: list[int]
    p2: list[int]


@dataclass(slots=True)
class CameraSettings:
    camera_id: str
    name: str
    source: str
    enabled: bool
    zones: list[ZoneSettings]
    lines: list[LineSettings]


@dataclass(slots=True)
class ModelSettings:
    yolo_model: str
    clip_model: str
    blip_model: str
    whisper_model: str
    face_model: str
    device: str
    fp16: bool


@dataclass(slots=True)
class ThresholdSettings:
    detector_confidence: float
    face_similarity: float
    semantic_similarity: float
    tracking_max_distance: float
    running_speed_px_per_sec: float


@dataclass(slots=True)
class RuntimeSettings:
    analyze_fps: int
    batch_size: int
    reconnect_seconds: float
    buffer_seconds: int
    max_parallel_cameras: int


@dataclass(slots=True)
class AlertSettings:
    telegram_enabled: bool
    discord_enabled: bool
    email_enabled: bool


@dataclass(slots=True)
class AppSettings:
    database_url: str
    output_dir: str
    videos_dir: str
    models: ModelSettings
    thresholds: ThresholdSettings
    runtime: RuntimeSettings
    alerts: AlertSettings
    cameras: list[CameraSettings]

    def ensure_directories(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.videos_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "frames").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "clips").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "recordings").mkdir(parents=True, exist_ok=True)


def _build_camera(raw: dict[str, Any]) -> CameraSettings:
    zones = [ZoneSettings(**zone) for zone in raw.get("zones", [])]
    lines = [LineSettings(**line) for line in raw.get("lines", [])]
    source_value = str(raw["source"])
    return CameraSettings(
        camera_id=raw["camera_id"],
        name=raw["name"],
        source=source_value,
        enabled=bool(raw.get("enabled", True)),
        zones=zones,
        lines=lines,
    )


def load_settings(config_path: str | Path = "config\\config.json") -> AppSettings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    settings = AppSettings(
        database_url=raw["database_url"],
        output_dir=raw["output_dir"],
        videos_dir=raw["videos_dir"],
        models=ModelSettings(**raw["models"]),
        thresholds=ThresholdSettings(**raw["thresholds"]),
        runtime=RuntimeSettings(**raw["runtime"]),
        alerts=AlertSettings(**raw["alerts"]),
        cameras=[_build_camera(camera) for camera in raw.get("cameras", [])],
    )
    settings.ensure_directories()
    return settings
