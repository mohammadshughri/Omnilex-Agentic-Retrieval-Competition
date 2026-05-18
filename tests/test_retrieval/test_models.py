"""Tests for model protocol compliance."""

import numpy as np

from omnilex.retrieval.models import EmbeddingModel, RerankerModel


class FakeEmbedder(EmbeddingModel):
    """Minimal embedder for protocol compliance testing."""

    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.random.randn(len(texts), 4).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.random.randn(len(queries), 4).astype(np.float32)


class FakeReranker(RerankerModel):
    """Minimal reranker for protocol compliance testing."""

    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray:
        return np.random.rand(len(documents)).astype(np.float32)


class TestEmbeddingModel:
    def test_encode_documents_returns_correct_shape(self):
        model = FakeEmbedder()
        vecs = model.encode_documents(["hello", "world"])
        assert vecs.shape == (2, 4)

    def test_encode_queries_returns_correct_shape(self):
        model = FakeEmbedder()
        vecs = model.encode_queries(["test query"])
        assert vecs.shape == (1, 4)

    def test_embedding_dim(self):
        model = FakeEmbedder()
        assert model.embedding_dim == 4


class TestRerankerModel:
    def test_score_pairs_returns_correct_length(self):
        model = FakeReranker()
        scores = model.score_pairs("query", ["doc1", "doc2", "doc3"])
        assert scores.shape == (3,)
