from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database.repository import DatabaseRepository

LOGGER = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceRecognizer:
    def __init__(
        self,
        repository: DatabaseRepository,
        model_name: str,
        similarity_threshold: float,
        use_cuda: bool,
    ) -> None:
        self.repository = repository
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.use_cuda = use_cuda
        self.app: Any | None = None

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            LOGGER.warning("InsightFace no disponible. Reconocimiento facial desactivado.")
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))

    @property
    def enabled(self) -> bool:
        return self.app is not None

    def register_face(self, name: str, image_path: str) -> str:
        if self.app is None:
            raise RuntimeError("InsightFace no está instalado. No se puede registrar rostros.")

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"No existe la imagen para registrar rostro: {image_file}")

        image = cv2.imread(str(image_file))
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No se detectó ningún rostro en la imagen.")

        embedding = faces[0].normed_embedding.astype(np.float32).tolist()
        record = self.repository.add_face(name=name, embedding=embedding, image_path=str(image_file))
        return record.id

    def recognize(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.app is None:
            return []
        faces = self.app.get(frame)
        if not faces:
            return []

        known = self.repository.get_face_records()
        known_embeddings = [(item.name, np.array(item.embedding, dtype=np.float32)) for item in known]

        matches: list[dict[str, Any]] = []
        for face in faces:
            emb = face.normed_embedding.astype(np.float32)
            name, score = self._best_match(emb, known_embeddings)
            bbox = [int(v) for v in face.bbox.tolist()]
            if name is None or score < self.similarity_threshold:
                matches.append({"name": "unknown", "score": score, "bbox": bbox})
                continue
            matches.append({"name": name, "score": score, "bbox": bbox})
        return matches

    def _best_match(self, emb: np.ndarray, known_embeddings: list[tuple[str, np.ndarray]]) -> tuple[str | None, float]:
        best_name: str | None = None
        best_score = -1.0
        for name, known_emb in known_embeddings:
            score = _cosine_similarity(emb, known_emb)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score
