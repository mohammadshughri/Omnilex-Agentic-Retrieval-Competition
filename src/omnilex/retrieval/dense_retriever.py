"""Dense retriever — orchestrates query encoding, FAISS search, and RRF fusion."""

from __future__ import annotations

from omnilex.retrieval.anchor_extractor import extract_citation_anchors
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.fusion import Candidate, deduplicate_candidates, rrf_fuse
from omnilex.retrieval.models import EmbeddingModel


class DenseRetriever:
    """End-to-end retriever: query -> encode -> search -> fuse -> candidates."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        laws_index: DenseIndex | None = None,
        courts_index: DenseIndex | None = None,
    ):
        self._embedder = embedder
        self._laws_index = laws_index
        self._courts_index = courts_index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        faiss_top_k: int = 100,
        rrf_k: int = 60,
    ) -> list[Candidate]:
        channels: list[list[Candidate]] = []

        anchor_citations = extract_citation_anchors(query)
        if anchor_citations:
            anchor_channel = [
                Candidate(citation_raw=c, score=1.0, source="anchor")
                for c in anchor_citations
            ]
            channels.append(anchor_channel)

        query_vec = self._embedder.encode_queries([query])

        if self._laws_index is not None:
            law_results = self._laws_index.search(query_vec, top_k=faiss_top_k)
            law_candidates = [
                Candidate(
                    citation_raw=r["citation_raw"],
                    score=r["score"],
                    source="laws_dense",
                    metadata=r,
                )
                for r in law_results
            ]
            channels.append(law_candidates)

        if self._courts_index is not None:
            court_results = self._courts_index.search(query_vec, top_k=faiss_top_k)
            court_candidates = [
                Candidate(
                    citation_raw=r["citation_raw"],
                    score=r["score"],
                    source="courts_dense",
                    metadata=r,
                )
                for r in court_results
            ]
            channels.append(court_candidates)

        fused = rrf_fuse(channels, k=rrf_k)
        deduped = deduplicate_candidates(fused)
        return deduped[:top_k]
