"""Tests for RRF fusion and candidate deduplication."""

from omnilex.retrieval.fusion import Candidate, rrf_fuse, deduplicate_candidates


class TestRRFFuse:
    def test_single_channel(self):
        channel = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="dense"),
            Candidate(citation_raw="Art. 2 ZGB", score=0.8, source="dense"),
        ]
        fused = rrf_fuse([channel], k=60)
        assert len(fused) == 2
        assert fused[0].citation_raw == "Art. 1 ZGB"
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_two_channels_boost_shared_candidate(self):
        dense = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="dense"),
            Candidate(citation_raw="Art. 2 ZGB", score=0.8, source="dense"),
        ]
        sparse = [
            Candidate(citation_raw="Art. 2 ZGB", score=0.7, source="sparse"),
            Candidate(citation_raw="Art. 3 ZGB", score=0.6, source="sparse"),
        ]
        fused = rrf_fuse([dense, sparse], k=60)
        scores = {c.citation_raw: c.rrf_score for c in fused}
        assert scores["Art. 2 ZGB"] > scores["Art. 3 ZGB"]

    def test_rrf_scores_are_deterministic(self):
        channel = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="dense"),
        ]
        fused1 = rrf_fuse([channel], k=60)
        fused2 = rrf_fuse([channel], k=60)
        assert fused1[0].rrf_score == fused2[0].rrf_score

    def test_empty_channels(self):
        fused = rrf_fuse([], k=60)
        assert fused == []

    def test_rrf_score_formula(self):
        channel = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="dense"),
        ]
        fused = rrf_fuse([channel], k=60)
        expected = 1.0 / (60 + 1)
        assert abs(fused[0].rrf_score - expected) < 1e-9


class TestDeduplicateCandidates:
    def test_dedup_keeps_highest_score(self):
        candidates = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.5, source="dense", rrf_score=0.02),
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="sparse", rrf_score=0.05),
        ]
        deduped = deduplicate_candidates(candidates)
        assert len(deduped) == 1
        assert deduped[0].rrf_score == 0.05

    def test_dedup_preserves_order(self):
        candidates = [
            Candidate(citation_raw="Art. 1 ZGB", score=0.9, source="a", rrf_score=0.05),
            Candidate(citation_raw="Art. 2 ZGB", score=0.8, source="a", rrf_score=0.04),
            Candidate(citation_raw="Art. 1 ZGB", score=0.7, source="b", rrf_score=0.03),
        ]
        deduped = deduplicate_candidates(candidates)
        assert len(deduped) == 2
        assert deduped[0].citation_raw == "Art. 1 ZGB"
