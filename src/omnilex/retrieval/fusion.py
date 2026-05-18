"""Reciprocal Rank Fusion and candidate management."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A retrieval candidate with citation and score."""

    citation_raw: str
    score: float
    source: str
    rrf_score: float = 0.0
    metadata: dict = field(default_factory=dict)


def rrf_fuse(
    channels: list[list[Candidate]],
    k: int = 60,
) -> list[Candidate]:
    """Fuse multiple ranked candidate lists via Reciprocal Rank Fusion.

    Each channel is a list of Candidates sorted by score descending.
    RRF score = sum across channels of 1 / (k + rank), rank is 1-indexed.
    """
    if not channels:
        return []

    rrf_scores: dict[str, float] = {}
    best_candidate: dict[str, Candidate] = {}

    for channel in channels:
        for rank_0, cand in enumerate(channel):
            rank_1 = rank_0 + 1
            contrib = 1.0 / (k + rank_1)
            rrf_scores[cand.citation_raw] = rrf_scores.get(cand.citation_raw, 0.0) + contrib

            if cand.citation_raw not in best_candidate or cand.score > best_candidate[cand.citation_raw].score:
                best_candidate[cand.citation_raw] = cand

    results = []
    for cite, total_score in rrf_scores.items():
        c = Candidate(
            citation_raw=cite,
            score=best_candidate[cite].score,
            source=best_candidate[cite].source,
            rrf_score=total_score,
            metadata=best_candidate[cite].metadata,
        )
        results.append(c)

    results.sort(key=lambda c: c.rrf_score, reverse=True)
    return results


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Deduplicate candidates by citation_raw, keeping highest rrf_score."""
    seen: dict[str, Candidate] = {}
    for c in candidates:
        if c.citation_raw not in seen or c.rrf_score > seen[c.citation_raw].rrf_score:
            seen[c.citation_raw] = c

    order = []
    added = set()
    for c in candidates:
        if c.citation_raw not in added:
            order.append(seen[c.citation_raw])
            added.add(c.citation_raw)
    return order
