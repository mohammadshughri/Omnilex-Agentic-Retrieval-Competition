"""Submission CSV generation from retrieval results."""

from __future__ import annotations

import csv
from pathlib import Path

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.fusion import Candidate


def select_top_k(candidates: list[Candidate], k: int) -> list[Candidate]:
    """Select top-k candidates by RRF score (assumes already sorted)."""
    return candidates[:k]


def generate_submission(
    results: dict[str, list[Candidate]],
    output_path: Path | str,
    top_k: int = 5,
) -> None:
    """Generate a submission CSV from retrieval results.

    Args:
        results: Dict mapping query_id to ranked Candidate lists.
        output_path: Path to write the CSV.
        top_k: Number of citations to keep per query.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer = CitationNormalizer()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()

        for query_id in sorted(results.keys()):
            top = select_top_k(results[query_id], k=top_k)
            raw_citations = [c.citation_raw for c in top]
            canonical = normalizer.canonicalize_list(raw_citations)
            writer.writerow({
                "query_id": query_id,
                "predicted_citations": ";".join(canonical),
            })
