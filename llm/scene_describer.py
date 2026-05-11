from __future__ import annotations

import logging

import cv2
import numpy as np

from detector.types import Detection

LOGGER = logging.getLogger(__name__)


class SceneDescriber:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.use_blip = False

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            LOGGER.warning("BLIP no disponible. Se usará descripción basada en reglas.")
            self.torch = None
            self.processor = None
            self.model = None
            return

        self.torch = torch
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.use_blip = True

    def describe(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        recognized_faces: list[dict],
        event_descriptions: list[str],
    ) -> str:
        if self.use_blip:
            return self._describe_with_blip(frame)
        return self._describe_with_rules(detections, recognized_faces, event_descriptions)

    def _describe_with_blip(self, frame: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return text if text else "Escena sin descripción."

    def _describe_with_rules(
        self,
        detections: list[Detection],
        recognized_faces: list[dict],
        event_descriptions: list[str],
    ) -> str:
        objects = [d.class_name for d in detections]
        persons = [face["name"] for face in recognized_faces if face["name"] != "unknown"]
        chunks: list[str] = []
        if objects:
            chunks.append(f"Objetos/personas detectados: {', '.join(sorted(set(objects)))}")
        if persons:
            chunks.append(f"Personas reconocidas: {', '.join(sorted(set(persons)))}")
        if event_descriptions:
            chunks.append(f"Eventos: {', '.join(event_descriptions)}")
        if not chunks:
            return "No se detectaron eventos relevantes en este frame."
        return " | ".join(chunks)
