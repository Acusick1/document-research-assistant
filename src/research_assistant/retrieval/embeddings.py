from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding, TextEmbedding


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return np.array(embeddings, dtype=np.float32)


@dataclasses.dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@runtime_checkable
class SparseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]: ...


class FastEmbedSparseEmbedder:
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model: SparseTextEmbedding | None = None

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[SparseVector]:
        model = self._get_model()
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in model.embed(texts)
        ]
