"""Retrieval tools and indexing for Swiss legal documents."""

# Pure-Python modules — safe to import unconditionally.
from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
from .tools import CourtSearchTool, LawSearchTool
from .fusion import Candidate, rrf_fuse, deduplicate_candidates
from .anchor_extractor import extract_citation_anchors
from .submission import generate_submission, select_top_k

# faiss / FlagEmbedding modules are imported lazily so that
# `import omnilex.retrieval` succeeds before those libraries are installed.
# Import them directly: `from omnilex.retrieval.dense_index import DenseIndex`
def __getattr__(name: str):
    if name in ("DenseIndex", "DenseIndexBuilder"):
        from .dense_index import DenseIndex, DenseIndexBuilder
        return {"DenseIndex": DenseIndex, "DenseIndexBuilder": DenseIndexBuilder}[name]
    if name == "DenseRetriever":
        from .dense_retriever import DenseRetriever
        return DenseRetriever
    if name == "BgeM3Embedder":
        from .models import BgeM3Embedder
        return BgeM3Embedder
    raise AttributeError(f"module 'omnilex.retrieval' has no attribute {name!r}")

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
    "DenseIndex",
    "DenseIndexBuilder",
    "DenseRetriever",
    "BgeM3Embedder",
]
