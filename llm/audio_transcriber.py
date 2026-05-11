from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self.model = None
        try:
            import whisper
        except ImportError:
            LOGGER.warning("Whisper no disponible. Transcripción de audio desactivada.")
            self.whisper = None
            return

        self.whisper = whisper
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Whisper no está instalado o no se pudo cargar el modelo.")
        result = self.model.transcribe(audio_path)
        return str(result.get("text", "")).strip()
