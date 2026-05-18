"""Retrieval tools and indexing for Swiss legal documents."""

from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
from .tools import CourtSearchTool, LawSearchTool
from .fusion import Candidate, rrf_fuse, deduplicate_candidates
from .anchor_extractor import extract_citation_anchors
from .submission import generate_submission, select_top_k

__all__ = [
    "BM25Index",
    "build_index",
    "load_jsonl_corpus",
    "search",
    "LawSearchTool",
    "CourtSearchTool",
    "Candidate",
    "rrf_fuse",
    "deduplicate_candidates",
    "extract_citation_anchors",
    "generate_submission",
    "select_top_k",
]
