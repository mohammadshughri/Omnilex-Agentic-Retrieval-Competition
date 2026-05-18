"""Tests for the dense retriever pipeline."""

import tempfile
from pathlib import Path

import numpy as np

from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.dense_retriever import DenseRetriever
from omnilex.retrieval.models import EmbeddingModel


class StubEmbedder(EmbeddingModel):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.eye(min(len(texts), 4), 4, dtype=np.float32)
        if len(texts) > 4:
            extra = np.zeros((len(texts) - 4, 4), dtype=np.float32)
            vecs = np.vstack([vecs, extra])
        return vecs[:len(texts)]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        vecs = np.zeros((len(queries), 4), dtype=np.float32)
        vecs[:, 0] = 1.0  # Always match first doc
        return vecs


class TestDenseRetriever:
    def _build_index(self, tmpdir: Path, docs: list[dict], title: str | None = None):
        embedder = StubEmbedder()
        builder = DenseIndexBuilder(embedder)
        builder.build_from_records(
            records=docs,
            citation_field="citation",
            text_field="text",
            title_field=title,
            output_dir=tmpdir,
        )
        return DenseIndex.load(tmpdir)

    def test_retrieve_returns_candidates(self):
        docs = [
            {"citation": "Art. 1 ZGB", "text": "Das Gesetz"},
            {"citation": "Art. 2 ZGB", "text": "Guter Glaube"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index = self._build_index(Path(tmpdir), docs)
            embedder = StubEmbedder()
            retriever = DenseRetriever(
                embedder=embedder,
                laws_index=index,
                courts_index=None,
            )
            results = retriever.retrieve("test query", top_k=5)
            assert len(results) > 0
            assert all(hasattr(c, "citation_raw") for c in results)

    def test_retrieve_includes_anchors(self):
        docs = [
            {"citation": "Art. 221 StPO", "text": "Haftgruende"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index = self._build_index(Path(tmpdir), docs)
            embedder = StubEmbedder()
            retriever = DenseRetriever(
                embedder=embedder,
                laws_index=index,
                courts_index=None,
            )
            results = retriever.retrieve(
                "Art. 221 Abs. 1 StPO applies here",
                top_k=10,
            )
            citations = [c.citation_raw for c in results]
            assert any("Art. 221" in c for c in citations)
