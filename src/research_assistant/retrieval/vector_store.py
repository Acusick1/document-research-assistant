from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import FacetValueHit
from qdrant_client.http.models.models import Direction
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from research_assistant.config import Settings
from research_assistant.corpus.models import Chunk


class ChunkPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    document_id: str
    section_name: str
    chunk_index: int
    ticker: str
    company_name: str
    fiscal_year: int
    filing_type: str
    source: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkPayload:
        metadata = chunk.metadata.model_dump(mode="json")
        metadata.update(
            chunk_id=chunk.id,
            text=chunk.text,
            document_id=chunk.document_id,
            section_name=chunk.section_name,
            chunk_index=chunk.chunk_index,
        )
        return cls(**metadata)


class SearchResult(ChunkPayload):
    id: str
    score: float


def _str_to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def create_qdrant_client(settings: Settings) -> QdrantClient:
    match settings.qdrant_mode:
        case "memory":
            return QdrantClient(":memory:")
        case "local":
            return QdrantClient(path=settings.qdrant_path)
        case "server":
            return QdrantClient(url=settings.qdrant_url)
        case "cloud":
            api_key = settings.qdrant_api_key
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=api_key.get_secret_value() if api_key else None,
            )


class QdrantStore:
    def __init__(self, client: QdrantClient, collection_name: str, vector_dim: int) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    def ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="fiscal_year",
                field_schema=PayloadSchemaType.INTEGER,
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=_str_to_uuid(chunk.id),
                vector=vector,
                payload=ChunkPayload.from_chunk(chunk).model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        qdrant_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if isinstance(v, list):
                    conditions.append(FieldCondition(key=k, match=MatchAny(any=v)))
                else:
                    conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            SearchResult(id=str(point.id), score=point.score, **point.payload)
            for point in results.points
            if point.payload
        ]

    def get_latest_fiscal_year(self, ticker: str) -> int | None:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
            ),
            order_by=OrderBy(key="fiscal_year", direction=Direction.DESC),
            limit=1,
            with_payload=["fiscal_year"],
        )
        if points and points[0].payload:
            return int(points[0].payload["fiscal_year"])
        return None

    def get_field_values(self, field: str, limit: int = 100) -> list[FacetValueHit]:
        response = self.client.facet(
            collection_name=self.collection_name,
            key=field,
            limit=limit,
        )
        return response.hits

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
