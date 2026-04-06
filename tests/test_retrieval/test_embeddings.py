from __future__ import annotations

import pytest

from research_assistant.retrieval.embeddings import FastEmbedEmbedder


class TestFastEmbedEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self) -> FastEmbedEmbedder:
        return FastEmbedEmbedder()

    def test_dim(self, embedder: FastEmbedEmbedder) -> None:
        assert embedder.dim == 384

    def test_embed_single(self, embedder: FastEmbedEmbedder) -> None:
        result = embedder.embed(["Hello world"])
        assert result.shape == (1, 384)

    def test_embed_batch(self, embedder: FastEmbedEmbedder) -> None:
        texts = ["First document", "Second document", "Third document"]
        result = embedder.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_produces_different_vectors(self, embedder: FastEmbedEmbedder) -> None:
        result = embedder.embed(["cats are great", "quantum physics equations"])
        assert not (result[0] == result[1]).all()


class TestSpladeValidation:
    def test_splade_produces_sparse_vectors(self) -> None:
        from fastembed import SparseTextEmbedding

        model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        results = list(model.embed(["Apple reported strong revenue growth in fiscal year 2024"]))
        assert len(results) == 1
        sparse = results[0]
        assert hasattr(sparse, "indices")
        assert hasattr(sparse, "values")
        assert len(sparse.indices) > 0
        assert len(sparse.values) == len(sparse.indices)
