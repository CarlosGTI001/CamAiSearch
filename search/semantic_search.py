from __future__ import annotations

from database.models import EventRecord
from database.repository import DatabaseRepository
from embeddings.clip_embedder import CLIPEmbedder


KEYWORD_EVENT_HINTS: dict[str, str] = {
    "correr": "person_running",
    "corriendo": "person_running",
    "intrusión": "intrusion",
    "intrusion": "intrusion",
    "sospechoso": "suspicious_activity",
    "sospechosa": "suspicious_activity",
    "mochila": "abandoned_object",
    "dejó": "abandoned_object",
    "dejar": "abandoned_object",
    "cruzó": "line_crossing",
    "cruce": "line_crossing",
}


class SemanticSearchService:
    def __init__(
        self,
        repository: DatabaseRepository,
        embedder: CLIPEmbedder,
        semantic_threshold: float,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold

    def search(self, query: str, top_k: int = 20, camera_id: str | None = None) -> list[tuple[float, EventRecord]]:
        event_types = self._hinted_event_types(query)
        query_embedding = self.embedder.embed_text(query)
        scored = self.repository.semantic_search_events(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            camera_id=camera_id,
            event_types=event_types if event_types else None,
        )
        filtered = [(score, event) for score, event in scored if score >= self.semantic_threshold]
        if filtered:
            return filtered[:top_k]
        return scored[:top_k]

    def _hinted_event_types(self, query: str) -> list[str]:
        lowered = query.lower()
        event_types = {event for keyword, event in KEYWORD_EVENT_HINTS.items() if keyword in lowered}
        return sorted(event_types)
