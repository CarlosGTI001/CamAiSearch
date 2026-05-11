from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np

from detector.types import Detection

LOGGER = logging.getLogger(__name__)

COCO_MAP: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    39: "bottle",
    41: "cup",
    56: "chair",
    63: "laptop",
    67: "cell phone",
    73: "book",
    75: "vase",
}


class YOLODetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        device: str = "cpu",
        fp16: bool = False,
        target_classes: Iterable[str] | None = None,
    ) -> None:
        self.confidence = confidence
        self.target_classes = set(target_classes or [])
        self.use_ultralytics = False
        self.device = device
        self.fp16 = fp16
        self.class_map = COCO_MAP

        try:
            from ultralytics import YOLO
        except ImportError:
            LOGGER.warning("Ultralytics no disponible. Se usa fallback HOG para personas.")
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return

        self.model = YOLO(model_path)
        self.use_ultralytics = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.use_ultralytics:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            half=self.fp16,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.class_map.get(class_id, result.names.get(class_id, str(class_id)))
                if self.target_classes and class_name not in self.target_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
        return detections

    def _detect_hog(self, frame: np.ndarray) -> list[Detection]:
        rects, weights = self.model.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            if conf < self.confidence:
                continue
            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    confidence=conf,
                    class_id=0,
                    class_name="person",
                )
            )
        return detections
