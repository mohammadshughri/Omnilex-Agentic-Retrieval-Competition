"""Acceptance tests — verify pipeline correctness on fixtures before large indexing."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.fusion import Candidate, deduplicate_candidates, rrf_fuse
from omnilex.retrieval.models import EmbeddingModel
from omnilex.retrieval.submission import generate_submission


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FIXTURE_LAWS = [
    {"citation": "Art. 1 ZGB", "text": "Das Gesetz findet auf alle Rechtsfragen Anwendung", "title": "Anwendung"},
    {"citation": "Art. 1 OR", "text": "Zum Abschlusse eines Vertrages ist die Willensaeusserung erforderlich", "title": "Vertrag"},
    {"citation": "Art. 221 StPO", "text": "Untersuchungshaft wegen Kollusionsgefahr", "title": "Haft"},
]

FIXTURE_COURTS = [
    {"citation": "BGE 137 IV 122", "text": "Haftgruende und Verhaeltnismaessigkeit"},
    {"citation": "BGE 116 Ia 56", "text": "Meinungsfreiheit und Pressefreiheit"},
]


class StubEmbedder(EmbeddingModel):
    """Deterministic embedder using a fixed random seed."""

    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 4).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(queries), 4).astype(np.float32)


def _build_index(records: list[dict], tmpdir: Path, title_field: str | None = "title") -> DenseIndex:
    builder = DenseIndexBuilder(StubEmbedder())
    builder.build_from_records(
        records=records,
        citation_field="citation",
        text_field="text",
        title_field=title_field,
        output_dir=tmpdir,
    )
    return DenseIndex.load(tmpdir)


# ---------------------------------------------------------------------------
# TestFaissMetadataAlignment
# ---------------------------------------------------------------------------

class TestFaissMetadataAlignment:
    """FAISS row IDs and metadata must stay in sync for all records."""

    def test_metadata_idx_matches_faiss_row_for_all_records(self):
        """Every metadata[i]['idx'] equals i — not just index 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _build_index(FIXTURE_LAWS, Path(tmpdir))
            for i, meta in enumerate(index.metadata):
                assert meta["idx"] == i, f"metadata[{i}]['idx'] = {meta['idx']}, expected {i}"

    def test_metadata_citation_raw_matches_source_record(self):
        """metadata[i]['citation_raw'] matches the original record's citation field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _build_index(FIXTURE_LAWS, Path(tmpdir))
            for i, meta in enumerate(index.metadata):
                expected = FIXTURE_LAWS[i]["citation"]
                assert meta["citation_raw"] == expected, (
                    f"Row {i}: got '{meta['citation_raw']}', expected '{expected}'"
                )


# ---------------------------------------------------------------------------
# TestCitationNormalizationRoundtrip
# ---------------------------------------------------------------------------

class TestCitationNormalizationRoundtrip:
    """Raw citation strings from the corpus must survive CitationNormalizer."""

    def setup_method(self):
        self.normalizer = CitationNormalizer()

    def test_law_fixture_citations_normalize(self):
        """All fixture law citations parse to a non-None Citation object."""
        for rec in FIXTURE_LAWS:
            result = self.normalizer.normalize(rec["citation"])
            assert result is not None, f"Failed to normalize: '{rec['citation']}'"

    def test_court_fixture_citations_normalize(self):
        """All fixture court citations parse to a non-None Citation object."""
        for rec in FIXTURE_COURTS:
            result = self.normalizer.normalize(rec["citation"])
            assert result is not None, f"Failed to normalize: '{rec['citation']}'"

    def test_citation_raw_returned_by_search_normalizes(self):
        """citation_raw values returned by DenseIndex.search() are parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _build_index(FIXTURE_LAWS, Path(tmpdir))
            query_vec = np.random.RandomState(0).randn(1, 4).astype(np.float32)
            results = index.search(query_vec, top_k=len(FIXTURE_LAWS))
            for r in results:
                result = self.normalizer.normalize(r["citation_raw"])
                assert result is not None, f"Search result not normalizable: '{r['citation_raw']}'"


# ---------------------------------------------------------------------------
# TestDuplicateCitationCollapse
# ---------------------------------------------------------------------------

class TestDuplicateCitationCollapse:
    """The same citation from multiple passages must collapse to one candidate."""

    def test_same_citation_two_passages_collapses_to_one(self):
        """Two candidates with identical citation_raw deduplicate to one."""
        channel = [
            Candidate(citation_raw="Art. 1 OR", score=0.9, source="dense", rrf_score=0.05),
            Candidate(citation_raw="Art. 1 OR", score=0.7, source="dense", rrf_score=0.03),
            Candidate(citation_raw="Art. 2 OR", score=0.8, source="dense", rrf_score=0.04),
        ]
        deduped = deduplicate_candidates(channel)
        raw_citations = [c.citation_raw for c in deduped]
        assert raw_citations.count("Art. 1 OR") == 1

    def test_dedup_keeps_better_rrf_score(self):
        """The surviving duplicate carries the higher rrf_score."""
        channel = [
            Candidate(citation_raw="Art. 1 OR", score=0.9, source="dense", rrf_score=0.05),
            Candidate(citation_raw="Art. 1 OR", score=0.7, source="dense", rrf_score=0.03),
        ]
        deduped = deduplicate_candidates(channel)
        assert len(deduped) == 1
        assert deduped[0].rrf_score == 0.05


# ---------------------------------------------------------------------------
# TestRRFDeterminism
# ---------------------------------------------------------------------------

class TestRRFDeterminism:
    """Three-channel RRF fusion must produce identical output on repeated calls."""

    def _make_channels(self) -> list[list[Candidate]]:
        anchor = [Candidate(citation_raw="Art. 221 StPO", score=1.0, source="anchor")]
        dense_laws = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.91, source="laws_dense"),
            Candidate(citation_raw="Art. 1 OR", score=0.88, source="laws_dense"),
            Candidate(citation_raw="Art. 221 StPO", score=0.85, source="laws_dense"),
        ]
        dense_courts = [
            Candidate(citation_raw="BGE 137 IV 122", score=0.87, source="courts_dense"),
            Candidate(citation_raw="BGE 116 Ia 56", score=0.82, source="courts_dense"),
        ]
        return [anchor, dense_laws, dense_courts]

    def test_three_channel_fusion_is_deterministic(self):
        """Same three-channel input produces identical ranked output twice."""
        result_a = rrf_fuse(self._make_channels(), k=60)
        result_b = rrf_fuse(self._make_channels(), k=60)
        assert len(result_a) == len(result_b)
        for a, b in zip(result_a, result_b):
            assert a.citation_raw == b.citation_raw
            assert a.rrf_score == b.rrf_score


# ---------------------------------------------------------------------------
# TestSubmissionFormat
# ---------------------------------------------------------------------------

class TestSubmissionFormat:
    """The generated CSV must be structurally valid for the competition scorer."""

    def _make_results(self, n_queries: int = 3) -> dict[str, list[Candidate]]:
        return {
            f"q{i}": [
                Candidate(citation_raw=f"Art. {i} ZGB", score=0.9, source="d", rrf_score=0.05),
                Candidate(citation_raw=f"BGE 11{i} II 1", score=0.8, source="d", rrf_score=0.04),
            ]
            for i in range(1, n_queries + 1)
        }

    def test_generated_csv_has_required_columns(self):
        """Output CSV has exactly query_id and predicted_citations columns."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._make_results(), path, top_k=5)
        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {"query_id", "predicted_citations"}
        path.unlink()

    def test_generated_csv_has_no_duplicate_query_ids(self):
        """Each query_id appears exactly once in the output CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._make_results(n_queries=4), path, top_k=5)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        query_ids = [r["query_id"] for r in rows]
        assert len(query_ids) == len(set(query_ids))
        path.unlink()

    def test_citations_are_semicolon_separated(self):
        """Multi-citation fields use ';' as separator with no empty parts."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._make_results(n_queries=1), path, top_k=5)
        with open(path) as f:
            row = next(csv.DictReader(f))
        parts = [p.strip() for p in row["predicted_citations"].split(";")]
        assert all(len(p) > 0 for p in parts)
        path.unlink()


# ---------------------------------------------------------------------------
# TestTopKBounds
# ---------------------------------------------------------------------------

class TestTopKBounds:
    """Fixed-k selection must respect bounds through the full generate_submission path."""

    def _results_with_many_candidates(self) -> dict[str, list[Candidate]]:
        return {
            "q1": [
                Candidate(
                    citation_raw=f"Art. {i} ZGB",
                    score=1.0 - i * 0.01,
                    source="d",
                    rrf_score=0.05 - i * 0.001,
                )
                for i in range(20)
            ]
        }

    def _citation_count(self, path: Path) -> int:
        with open(path) as f:
            row = next(csv.DictReader(f))
        raw = row["predicted_citations"]
        if not raw.strip():
            return 0
        return len([p for p in raw.split(";") if p.strip()])

    def test_top_2_produces_at_most_2_citations(self):
        """top_k=2 yields ≤ 2 citations per row."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._results_with_many_candidates(), path, top_k=2)
        assert self._citation_count(path) <= 2
        path.unlink()

    def test_top_3_produces_at_most_3_citations(self):
        """top_k=3 yields ≤ 3 citations per row."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._results_with_many_candidates(), path, top_k=3)
        assert self._citation_count(path) <= 3
        path.unlink()

    def test_top_5_produces_at_most_5_citations(self):
        """top_k=5 yields ≤ 5 citations per row."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._results_with_many_candidates(), path, top_k=5)
        assert self._citation_count(path) <= 5
        path.unlink()

    def test_top_10_produces_at_most_10_citations(self):
        """top_k=10 yields ≤ 10 citations per row."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(self._results_with_many_candidates(), path, top_k=10)
        assert self._citation_count(path) <= 10
        path.unlink()

    def test_fewer_candidates_than_k_returns_all(self):
        """When fewer candidates exist than k, all are returned without padding."""
        results = {
            "q1": [
                Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
                Candidate(citation_raw="Art. 2 OR", score=0.8, source="d", rrf_score=0.04),
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        generate_submission(results, path, top_k=10)
        assert self._citation_count(path) == 2
        path.unlink()


# ---------------------------------------------------------------------------
# TestSparseRoundTrip
# ---------------------------------------------------------------------------

class TestSparseRoundTrip:
    """scipy.sparse save/load round-trip for sparse embedding weights."""

    def test_sparse_npz_save_and_load_roundtrip(self):
        """A CSR matrix saved as sparse.npz loads back with identical data."""
        data = np.array([0.8, 0.5, 0.3, 0.9], dtype=np.float32)
        indices = np.array([0, 3, 7, 2])
        indptr = np.array([0, 1, 2, 3, 4])
        original = scipy.sparse.csr_matrix((data, indices, indptr), shape=(4, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse.npz"
            scipy.sparse.save_npz(path, original)
            loaded = scipy.sparse.load_npz(path)

        assert loaded.shape == original.shape
        assert loaded.dtype == original.dtype
        diff = (loaded - original)
        assert diff.nnz == 0

    def test_sparse_query_scoring(self):
        """Dot product of a sparse query vector against sparse doc matrix yields correct scores."""
        # 3 docs, 10-dim vocabulary
        doc_rows = [
            {1: 0.9, 5: 0.4},   # doc 0: tokens 1, 5
            {2: 0.7, 5: 0.6},   # doc 1: tokens 2, 5
            {3: 0.8},            # doc 2: token 3 only
        ]
        doc_matrix = scipy.sparse.lil_matrix((3, 10), dtype=np.float32)
        for row_idx, weights in enumerate(doc_rows):
            for col, val in weights.items():
                doc_matrix[row_idx, col] = val
        doc_matrix = doc_matrix.tocsr()

        # Query overlaps with tokens 1 and 2
        query_vec = scipy.sparse.lil_matrix((1, 10), dtype=np.float32)
        query_vec[0, 1] = 1.0
        query_vec[0, 2] = 1.0
        query_vec = query_vec.tocsr()

        scores = (query_vec @ doc_matrix.T).toarray()[0]

        # doc 0 shares token 1 → positive score
        assert scores[0] > 0
        # doc 1 shares token 2 → positive score
        assert scores[1] > 0
        # doc 2 shares no tokens → zero score
        assert scores[2] == 0.0
        # doc 0 score = 1.0 * 0.9, doc 1 score = 1.0 * 0.7
        assert abs(scores[0] - 0.9) < 1e-5
        assert abs(scores[1] - 0.7) < 1e-5
