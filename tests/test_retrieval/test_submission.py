"""Tests for submission generation."""

import csv
import tempfile
from pathlib import Path

from omnilex.retrieval.fusion import Candidate
from omnilex.retrieval.submission import generate_submission, select_top_k


class TestSelectTopK:
    def test_returns_k_items(self):
        candidates = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
            Candidate(citation_raw="Art. 2 ZGB", score=0.8, source="d", rrf_score=0.04),
            Candidate(citation_raw="Art. 3 ZGB", score=0.7, source="d", rrf_score=0.03),
        ]
        selected = select_top_k(candidates, k=2)
        assert len(selected) == 2

    def test_returns_all_if_fewer_than_k(self):
        candidates = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
        ]
        selected = select_top_k(candidates, k=5)
        assert len(selected) == 1

    def test_preserves_order(self):
        candidates = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
            Candidate(citation_raw="Art. 2 ZGB", score=0.8, source="d", rrf_score=0.04),
        ]
        selected = select_top_k(candidates, k=2)
        assert selected[0].citation_raw == "Art. 1 ZGB"


class TestGenerateSubmission:
    def test_writes_valid_csv(self):
        results = {
            "q1": [
                Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
                Candidate(citation_raw="BGE 116 Ia 56", score=0.8, source="d", rrf_score=0.04),
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        generate_submission(results, path, top_k=5)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["query_id"] == "q1"
        assert "Art. 1 ZGB" in rows[0]["predicted_citations"]
        path.unlink()

    def test_semicolon_separated(self):
        results = {
            "q1": [
                Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="d", rrf_score=0.05),
                Candidate(citation_raw="Art. 2 OR", score=0.8, source="d", rrf_score=0.04),
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)

        generate_submission(results, path, top_k=5)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert ";" in row["predicted_citations"]
        path.unlink()
