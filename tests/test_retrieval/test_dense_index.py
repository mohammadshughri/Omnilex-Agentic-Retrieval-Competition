"""Tests for dense index building and loading."""

import json
import tempfile
from pathlib import Path

import numpy as np

from omnilex.retrieval.dense_index import DenseIndexBuilder, DenseIndex
from omnilex.retrieval.models import EmbeddingModel


class StubEmbedder(EmbeddingModel):
    """Deterministic embedder for testing."""

    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.zeros((len(texts), 4), dtype=np.float32)
        for i in range(len(texts)):
            vecs[i, i % 4] = 1.0
        return vecs

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        vecs = np.zeros((len(queries), 4), dtype=np.float32)
        for i in range(len(queries)):
            vecs[i, i % 4] = 1.0
        return vecs


class TestDenseIndexBuilder:
    def test_build_and_search(self):
        docs = [
            {"citation": "Art. 1 ZGB", "text": "Das Gesetz", "title": "Recht"},
            {"citation": "Art. 2 ZGB", "text": "Guter Glaube", "title": "Treu"},
            {"citation": "Art. 3 ZGB", "text": "Handlung", "title": "Pflicht"},
        ]
        embedder = StubEmbedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            builder = DenseIndexBuilder(embedder)
            builder.build_from_records(
                records=docs,
                citation_field="citation",
                text_field="text",
                title_field="title",
                output_dir=out,
            )

            assert (out / "dense.npy").exists()
            assert (out / "faiss.index").exists()
            assert (out / "metadata.jsonl").exists()

            with open(out / "metadata.jsonl") as f:
                metadata = [json.loads(line) for line in f]
            assert len(metadata) == 3
            assert metadata[0]["citation_raw"] == "Art. 1 ZGB"
            assert metadata[0]["idx"] == 0

    def test_text_composition_includes_citation_and_title(self):
        docs = [
            {"citation": "Art. 1 ZGB", "text": "Das Gesetz", "title": "Recht"},
        ]
        embedder = StubEmbedder()
        builder = DenseIndexBuilder(embedder)
        composed = builder._compose_text(docs[0], "text", "citation", "title")
        assert "Art. 1 ZGB" in composed
        assert "Recht" in composed
        assert "Das Gesetz" in composed

    def test_text_composition_without_title(self):
        doc = {"citation": "BGE 116 Ia 56", "text": "Meinungsfreiheit"}
        embedder = StubEmbedder()
        builder = DenseIndexBuilder(embedder)
        composed = builder._compose_text(doc, "text", "citation", None)
        assert "BGE 116 Ia 56" in composed
        assert "Meinungsfreiheit" in composed


class TestDenseIndex:
    def test_load_and_search(self):
        docs = [
            {"citation": "Art. 1 ZGB", "text": "Das Gesetz", "title": "Recht"},
            {"citation": "Art. 2 ZGB", "text": "Guter Glaube", "title": "Treu"},
        ]
        embedder = StubEmbedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            builder = DenseIndexBuilder(embedder)
            builder.build_from_records(
                records=docs,
                citation_field="citation",
                text_field="text",
                title_field="title",
                output_dir=out,
            )

            index = DenseIndex.load(out)
            assert len(index.metadata) == 2

            query_vec = np.zeros((1, 4), dtype=np.float32)
            query_vec[0, 0] = 1.0
            results = index.search(query_vec, top_k=2)
            assert len(results) <= 2
            assert results[0]["citation_raw"] == "Art. 1 ZGB"
