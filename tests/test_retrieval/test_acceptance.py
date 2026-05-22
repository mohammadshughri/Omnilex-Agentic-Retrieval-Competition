"""Acceptance tests — verify pipeline correctness on fixtures before large indexing."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.fusion import Candidate, rrf_fuse
from omnilex.retrieval.models import EmbeddingModel
from omnilex.retrieval.submission import generate_submission


class StubEmbedder(EmbeddingModel):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 4).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(queries), 4).astype(np.float32)


FIXTURE_LAWS = [
    {"citation": "Art. 1 ZGB", "text": "Das Gesetz findet auf alle Rechtsfragen Anwendung", "title": "Anwendung"},
    {"citation": "Art. 1 OR", "text": "Zum Abschlusse eines Vertrages ist die Willensaeusserung erforderlich", "title": "Vertrag"},
    {"citation": "Art. 221 StPO", "text": "Untersuchungshaft wegen Kollusionsgefahr", "title": "Haft"},
]


FIXTURE_COURTS = [
    {"citation": "BGE 137 IV 122", "text": "Haftgruende und Verhaeltnismaessigkeit"},
    {"citation": "BGE 116 Ia 56", "text": "Meinungsfreiheit und Pressefreiheit"},
]


class TestCitationNormalizationRoundtrip:
    def test_retrieved_raw_citations_normalize(self):
        normalizer = CitationNormalizer()
        for law in FIXTURE_LAWS:
            result = normalizer.normalize(law["citation"])
            assert result is not None, f"Failed to normalize: {law['citation']}"

        for court in FIXTURE_COURTS:
            result = normalizer.normalize(court["citation"])
            assert result is not None, f"Failed to normalize: {court['citation']}"


class TestRRFDeterminism:
    def test_same_input_same_output(self):
        channel = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d"),
            Candidate(citation_raw="Art. 2 OR", score=0.7, source="d"),
        ]
        r1 = rrf_fuse([channel])
        r2 = rrf_fuse([channel])
        assert [c.rrf_score for c in r1] == [c.rrf_score for c in r2]


class TestSubmissionValidation:
    def test_generated_csv_has_correct_format(self):
        results = {
            "q1": [
                Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
            ],
            "q2": [
                Candidate(citation_raw="BGE 116 Ia 56", score=0.8, source="d", rrf_score=0.04),
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        generate_submission(results, path, top_k=5)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert set(rows[0].keys()) == {"query_id", "predicted_citations"}
        assert len(rows) == 2
        path.unlink()


class TestTopKBounds:
    def test_k_values_produce_valid_output(self):
        results = {
            "q1": [
                Candidate(
                    citation_raw=f"Art. {i} ZGB",
                    score=1.0 - i * 0.01,
                    source="d",
                    rrf_score=0.05 - i * 0.001,
                )
                for i in range(20)
            ],
        }
        for k in [2, 3, 5, 10]:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
                path = Path(f.name)

            generate_submission(results, path, top_k=k)

            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)

            citations = [c.strip() for c in row["predicted_citations"].split(";") if c.strip()]
            assert len(citations) <= k
            path.unlink()


class TestFaissToMetadataAlignment:
    def test_metadata_idx_matches_faiss_row(self):
        embedder = StubEmbedder()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            builder = DenseIndexBuilder(embedder)
            builder.build_from_records(
                records=FIXTURE_LAWS,
                citation_field="citation",
                text_field="text",
                title_field="title",
                output_dir=out,
            )
            index = DenseIndex.load(out)
            for i, meta in enumerate(index.metadata):
                assert meta["idx"] == i
                assert meta["citation_raw"] == FIXTURE_LAWS[i]["citation"]
