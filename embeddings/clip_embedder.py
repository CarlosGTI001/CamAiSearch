from __future__ import annotations

import hashlib
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32).tolist()
    return (vec / norm).astype(np.float32).tolist()


class CLIPEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", fp16: bool = False) -> None:
        self.model_name = model_name
        self.device = device
        self.fp16 = fp16
        self.using_clip = False
        self.dim = 512

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            LOGGER.warning("Transformers/torch no disponible. Se usa embedder determinístico fallback.")
            self.torch = None
            self.processor = None
            self.model = None
            return

        self.torch = torch
        dtype = torch.float16 if fp16 and device == "cuda" else torch.float32
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.using_clip = True

    def embed_text(self, text: str) -> list[float]:
        if self.using_clip:
            return self._embed_text_clip(text)
        return self._embed_text_hash(text)

    def embed_image(self, image: np.ndarray) -> list[float]:
        if self.using_clip:
            return self._embed_image_clip(image)
        return self._embed_image_hash(image)

    def _embed_text_clip(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return _normalize(features[0].detach().float().cpu().numpy())

    def _embed_image_clip(self, image: np.ndarray) -> list[float]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=[rgb], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return _normalize(features[0].detach().float().cpu().numpy())

    def _embed_text_hash(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer(digest * (self.dim // len(digest) + 1), dtype=np.uint8)[: self.dim]
        centered = arr.astype(np.float32) - 127.5
        return _normalize(centered)

    def _embed_image_hash(self, image: np.ndarray) -> list[float]:
        thumb = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        raw = gray.flatten()
        if len(raw) < self.dim:
            repeats = self.dim // len(raw) + 1
            raw = np.tile(raw, repeats)
        vec = raw[: self.dim].astype(np.float32) - 127.5
        return _normalize(vec)
