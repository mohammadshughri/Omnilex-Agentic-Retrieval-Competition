# BGE-M3 Hybrid Retrieval Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end cross-lingual retrieval pipeline using BGE-M3 hybrid embeddings, RRF fusion, and citation-anchor extraction — producing Macro F1 scores on the 10 val queries by May 21.

**Architecture:** English legal queries are processed through three parallel channels: (1) citation-anchor extraction via regex, (2) dense FAISS retrieval, (3) sparse retrieval. Per-channel ranked lists are fused via Reciprocal Rank Fusion, deduplicated, thresholded to top-k, normalized through `CitationNormalizer`, and scored against gold labels.

**Tech Stack:** Python 3.10+, FlagEmbedding (BGE-M3), faiss-cpu, numpy, pandas. Extends existing `omnilex` package under `src/omnilex/retrieval/`.

**Spec:** `docs/superpowers/specs/2026-05-17-implementation-strategy-design.md`

**Working directory:** All paths are relative to `Omnilex-Agentic-Retrieval-Competition/`.

---

## Chunk 1: Foundation — Model Protocols, RRF Fusion, Citation-Anchor Extraction

These are pure-logic modules with no model dependencies. They can be built and tested with synthetic data.

---

### Task 1: Model Protocols (`src/omnilex/retrieval/models.py`)

Define the `EmbeddingModel` and `RerankerModel` protocols so all downstream code is model-agnostic.

**Files:**
- Create: `src/omnilex/retrieval/models.py`
- Test: `tests/test_retrieval/test_models.py`
- Create: `tests/test_retrieval/__init__.py` (if not exists)

- [ ] **Step 1: Create test directory**

```bash
mkdir -p tests/test_retrieval
touch tests/test_retrieval/__init__.py
```

- [ ] **Step 2: Write the test**

File: `tests/test_retrieval/test_models.py`

```python
"""Tests for model protocol compliance."""

import numpy as np

from omnilex.retrieval.models import EmbeddingModel, RerankerModel


class FakeEmbedder(EmbeddingModel):
    """Minimal embedder for protocol compliance testing."""

    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.random.randn(len(texts), 4).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.random.randn(len(queries), 4).astype(np.float32)


class FakeReranker(RerankerModel):
    """Minimal reranker for protocol compliance testing."""

    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray:
        return np.random.rand(len(documents)).astype(np.float32)


class TestEmbeddingModel:
    def test_encode_documents_returns_correct_shape(self):
        model = FakeEmbedder()
        vecs = model.encode_documents(["hello", "world"])
        assert vecs.shape == (2, 4)

    def test_encode_queries_returns_correct_shape(self):
        model = FakeEmbedder()
        vecs = model.encode_queries(["test query"])
        assert vecs.shape == (1, 4)

    def test_embedding_dim(self):
        model = FakeEmbedder()
        assert model.embedding_dim == 4


class TestRerankerModel:
    def test_score_pairs_returns_correct_length(self):
        model = FakeReranker()
        scores = model.score_pairs("query", ["doc1", "doc2", "doc3"])
        assert scores.shape == (3,)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'omnilex.retrieval.models'`

- [ ] **Step 4: Write the implementation**

File: `src/omnilex/retrieval/models.py`

```python
"""Model protocols for retrieval — embedding and reranking abstractions."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Protocol for dense embedding models (BGE-M3, Qwen3, etc.)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @abstractmethod
    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


class RerankerModel(ABC):
    """Protocol for reranker models (cross-encoder, causal LM, etc.)."""

    @abstractmethod
    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray: ...
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_models.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/omnilex/retrieval/models.py tests/test_retrieval/
git commit -m "feat(retrieval): add EmbeddingModel and RerankerModel protocols"
```

---

### Task 2: RRF Fusion (`src/omnilex/retrieval/fusion.py`)

Pure-logic module. Takes per-channel ranked candidate lists, fuses via RRF, deduplicates by normalized citation.

**Files:**
- Create: `src/omnilex/retrieval/fusion.py`
- Test: `tests/test_retrieval/test_fusion.py`

- [ ] **Step 1: Write the test**

File: `tests/test_retrieval/test_fusion.py`

```python
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
        # Art. 2 ZGB appears in both channels so should be boosted
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
        # rank 0 (0-indexed) -> score = 1 / (60 + 1) when using 1-indexed ranks
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_fusion.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

File: `src/omnilex/retrieval/fusion.py`

```python
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
    RRF score for a candidate = sum across channels of 1 / (k + rank),
    where rank is 1-indexed.

    Args:
        channels: List of per-channel candidate lists (each sorted best-first).
        k: RRF constant (default 60, per original RRF paper).

    Returns:
        Merged list sorted by RRF score descending.
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_fusion.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/omnilex/retrieval/fusion.py tests/test_retrieval/test_fusion.py
git commit -m "feat(retrieval): add RRF fusion and candidate deduplication"
```

---

### Task 3: Citation-Anchor Extraction (`src/omnilex/retrieval/anchor_extractor.py`)

Extract explicit `Art.` and `BGE` citations from query text, normalize them, and return as high-priority candidates.

**Files:**
- Create: `src/omnilex/retrieval/anchor_extractor.py`
- Test: `tests/test_retrieval/test_anchor_extractor.py`

- [ ] **Step 1: Write the test**

File: `tests/test_retrieval/test_anchor_extractor.py`

```python
"""Tests for citation-anchor extraction from query text."""

from omnilex.retrieval.anchor_extractor import extract_citation_anchors


class TestExtractCitationAnchors:
    def test_extracts_art_with_book(self):
        query = "Does Art. 221 Abs. 1 StPO apply to pre-trial detention?"
        anchors = extract_citation_anchors(query)
        assert any("Art. 221" in a and "StPO" in a for a in anchors)

    def test_extracts_bge(self):
        query = "According to BGE 137 IV 122, what are the requirements?"
        anchors = extract_citation_anchors(query)
        assert any("BGE 137 IV 122" in a for a in anchors)

    def test_extracts_multiple(self):
        query = (
            "Under Art. 83 SVG and Art. 59 Abs. 1 SVG, "
            "is the driver liable?"
        )
        anchors = extract_citation_anchors(query)
        assert len(anchors) >= 2

    def test_no_citations_returns_empty(self):
        query = "What are the requirements for a valid contract?"
        anchors = extract_citation_anchors(query)
        assert anchors == []

    def test_strips_lit_and_ziff(self):
        query = "Art. 221 Abs. 1 lit. b StPO mentions collusion risk"
        anchors = extract_citation_anchors(query)
        # lit. b should be stripped — gold labels stop at paragraph level
        for a in anchors:
            assert "lit." not in a

    def test_deduplicates(self):
        query = "Art. 1 ZGB states that... Art. 1 ZGB also applies to..."
        anchors = extract_citation_anchors(query)
        assert len(anchors) == 1

    def test_returns_canonical_form(self):
        query = "Artikel 41 Abs. 1 OR governs tort liability"
        anchors = extract_citation_anchors(query)
        assert "Art. 41 Abs. 1 OR" in anchors
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_anchor_extractor.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

File: `src/omnilex/retrieval/anchor_extractor.py`

```python
"""Extract citation anchors from query text."""

from __future__ import annotations

import re

from omnilex.citations.normalizer import CitationNormalizer


_NORMALIZER = None


def _get_normalizer() -> CitationNormalizer:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = CitationNormalizer()
    return _NORMALIZER


# Match "Art. X [Abs. Y] [lit. z] BOOK" where BOOK is 2-5 uppercase letters
_ART_PATTERN = re.compile(
    r"(?:Art\.?|Artikel)\s*\d+[a-z]?"
    r"(?:\s+(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*\d+[a-z]?)?"
    r"(?:\s+(?:lit\.?|Ziff\.?|Nr\.?)\s*[a-z0-9]+)?"
    r"\s+([A-Z][A-Za-z]{1,10})",
    re.UNICODE,
)

# Match "BGE VOL SECTION PAGE"
_BGE_PATTERN = re.compile(
    r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+",
    re.IGNORECASE,
)


def extract_citation_anchors(query: str) -> list[str]:
    """Extract and normalize citation anchors from a query string.

    Finds explicit Art./BGE patterns, normalizes them via CitationNormalizer,
    and returns deduplicated canonical IDs.

    Args:
        query: English (or German) legal query text.

    Returns:
        List of unique canonical citation IDs found in the query.
    """
    normalizer = _get_normalizer()
    raw_matches: list[str] = []

    for m in _ART_PATTERN.finditer(query):
        raw_matches.append(m.group(0).strip())

    for m in _BGE_PATTERN.finditer(query):
        raw_matches.append(m.group(0).strip())

    return normalizer.canonicalize_list(raw_matches)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_anchor_extractor.py -v
```

Expected: all 7 tests PASS. If any regex edge case fails, adjust the pattern and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/omnilex/retrieval/anchor_extractor.py tests/test_retrieval/test_anchor_extractor.py
git commit -m "feat(retrieval): add citation-anchor extraction from query text"
```

---

### Task 4: Submission Generator (`src/omnilex/retrieval/submission.py`)

Takes a list of `Candidate` objects per query, applies top-k selection, normalizes, and writes CSV.

**Files:**
- Create: `src/omnilex/retrieval/submission.py`
- Test: `tests/test_retrieval/test_submission.py`

- [ ] **Step 1: Write the test**

File: `tests/test_retrieval/test_submission.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_submission.py -v
```

- [ ] **Step 3: Write the implementation**

File: `src/omnilex/retrieval/submission.py`

```python
"""Submission CSV generation from retrieval results."""

from __future__ import annotations

import csv
from pathlib import Path

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.fusion import Candidate


def select_top_k(candidates: list[Candidate], k: int) -> list[Candidate]:
    """Select top-k candidates by RRF score (already sorted)."""
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_submission.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omnilex/retrieval/submission.py tests/test_retrieval/test_submission.py
git commit -m "feat(retrieval): add submission CSV generator with top-k selection"
```

---

### Task 5: Update `__init__.py` exports

**Files:**
- Modify: `src/omnilex/retrieval/__init__.py`

- [ ] **Step 1: Update the init file**

Replace the contents of `src/omnilex/retrieval/__init__.py` with:

```python
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
```

- [ ] **Step 2: Run all tests to verify nothing broke**

```bash
pytest tests/ -v
```

Expected: all tests PASS (existing + new)

- [ ] **Step 3: Commit**

```bash
git add src/omnilex/retrieval/__init__.py
git commit -m "feat(retrieval): export new fusion, anchor, and submission modules"
```

---

## Chunk 2: Dense Index Builder and BGE-M3 Adapter

These tasks add the embedding model adapter and the FAISS index builder.

---

### Task 6: BGE-M3 Adapter (`src/omnilex/retrieval/models.py` — extend)

Implement `BgeM3Embedder` that wraps `FlagEmbedding` behind the `EmbeddingModel` protocol.

**Files:**
- Modify: `src/omnilex/retrieval/models.py`
- Test: `tests/test_retrieval/test_models.py` (extend)

- [ ] **Step 1: Write the integration test**

Append to `tests/test_retrieval/test_models.py`:

```python
import pytest


class TestBgeM3Embedder:
    """Integration tests — skipped if FlagEmbedding is not installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_flag(self):
        pytest.importorskip("FlagEmbedding")

    @pytest.fixture
    def embedder(self):
        from omnilex.retrieval.models import BgeM3Embedder
        return BgeM3Embedder()

    def test_embedding_dim_is_1024(self, embedder):
        assert embedder.embedding_dim == 1024

    def test_encode_documents_shape(self, embedder):
        vecs = embedder.encode_documents(["test document"])
        assert vecs.shape == (1, 1024)

    def test_encode_queries_shape(self, embedder):
        vecs = embedder.encode_queries(["test query"])
        assert vecs.shape == (1, 1024)

    def test_encode_documents_returns_sparse(self, embedder):
        vecs = embedder.encode_documents(["test"], return_sparse=True)
        assert vecs.shape == (1, 1024)
        assert hasattr(embedder, "last_sparse_weights")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_models.py::TestBgeM3Embedder -v
```

Expected: FAIL or SKIP (if FlagEmbedding not installed — that's OK for CI)

- [ ] **Step 3: Write the implementation**

Add to the end of `src/omnilex/retrieval/models.py`:

```python
class BgeM3Embedder(EmbeddingModel):
    """BGE-M3 embedding model via FlagEmbedding.

    Produces dense vectors. Sparse weights are stored in
    self.last_sparse_weights after each encode_documents call
    with return_sparse=True.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.last_sparse_weights: list[dict] | None = None

    @property
    def embedding_dim(self) -> int:
        return 1024

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_sparse: bool = False,
    ) -> np.ndarray:
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        if return_sparse:
            self.last_sparse_weights = output["lexical_weights"]
        return np.array(output["dense_vecs"], dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        output = self._model.encode(
            queries,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(output["dense_vecs"], dtype=np.float32)
```

- [ ] **Step 4: Commit**

```bash
git add src/omnilex/retrieval/models.py tests/test_retrieval/test_models.py
git commit -m "feat(retrieval): add BgeM3Embedder adapter wrapping FlagEmbedding"
```

---

### Task 7: Dense Index Builder (`src/omnilex/retrieval/dense_index.py`)

Builds FAISS index + metadata JSONL from a corpus CSV. Uses `citation + title + text` composition per spec.

**Files:**
- Create: `src/omnilex/retrieval/dense_index.py`
- Test: `tests/test_retrieval/test_dense_index.py`

- [ ] **Step 1: Write the test**

File: `tests/test_retrieval/test_dense_index.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_dense_index.py -v
```

- [ ] **Step 3: Write the implementation**

File: `src/omnilex/retrieval/dense_index.py`

```python
"""Dense FAISS index builder and loader."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from omnilex.retrieval.models import EmbeddingModel


class DenseIndexBuilder:
    """Builds a FAISS index + metadata from document records."""

    def __init__(self, embedder: EmbeddingModel):
        self._embedder = embedder

    def _compose_text(
        self,
        record: dict,
        text_field: str,
        citation_field: str | None,
        title_field: str | None,
    ) -> str:
        parts = []
        if citation_field and citation_field in record:
            parts.append(record[citation_field])
        if title_field and title_field in record:
            parts.append(record[title_field])
        parts.append(record.get(text_field, ""))
        return " ".join(parts)

    def build_from_records(
        self,
        records: list[dict],
        citation_field: str,
        text_field: str,
        output_dir: Path | str,
        title_field: str | None = None,
        batch_size: int = 256,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        texts = [
            self._compose_text(r, text_field, citation_field, title_field)
            for r in records
        ]

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._embedder.encode_documents(batch, batch_size=batch_size)
            all_vecs.append(vecs)

        embeddings = np.vstack(all_vecs).astype(np.float32)
        np.save(output_dir / "dense.npy", embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(output_dir / "faiss.index"))

        with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for idx, rec in enumerate(records):
                meta = {
                    "idx": idx,
                    "citation_raw": rec.get(citation_field, ""),
                    "text_preview": rec.get(text_field, "")[:200],
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")


class DenseIndex:
    """Loaded dense FAISS index with metadata."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict],
        embeddings: np.ndarray | None = None,
    ):
        self.index = index
        self.metadata = metadata
        self.embeddings = embeddings

    @classmethod
    def load(cls, index_dir: Path | str) -> DenseIndex:
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "faiss.index"))

        metadata = []
        with open(index_dir / "metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))

        embeddings = None
        npy_path = index_dir / "dense.npy"
        if npy_path.exists():
            embeddings = np.load(npy_path)

        return cls(index=index, metadata=metadata, embeddings=embeddings)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
    ) -> list[dict]:
        qv = query_vectors.astype(np.float32).copy()
        faiss.normalize_L2(qv)
        scores, indices = self.index.search(qv, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_dense_index.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/omnilex/retrieval/dense_index.py tests/test_retrieval/test_dense_index.py
git commit -m "feat(retrieval): add DenseIndexBuilder and DenseIndex with FAISS"
```

---

## Chunk 3: Dense Retriever, Embedding Script, and Evaluation Harness

These tasks wire everything together into a runnable pipeline.

---

### Task 8: Dense Retriever (`src/omnilex/retrieval/dense_retriever.py`)

Orchestrates query -> encode -> search -> RRF -> top-k.

**Files:**
- Create: `src/omnilex/retrieval/dense_retriever.py`
- Test: `tests/test_retrieval/test_dense_retriever.py`

- [ ] **Step 1: Write the test**

File: `tests/test_retrieval/test_dense_retriever.py`

```python
"""Tests for the dense retriever pipeline."""

import tempfile
from pathlib import Path

import numpy as np

from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.dense_retriever import DenseRetriever
from omnilex.retrieval.models import EmbeddingModel


class StubEmbedder(EmbeddingModel):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.eye(min(len(texts), 4), 4, dtype=np.float32)
        if len(texts) > 4:
            extra = np.zeros((len(texts) - 4, 4), dtype=np.float32)
            vecs = np.vstack([vecs, extra])
        return vecs[:len(texts)]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        vecs = np.zeros((len(queries), 4), dtype=np.float32)
        vecs[:, 0] = 1.0  # Always match first doc
        return vecs


class TestDenseRetriever:
    def _build_index(self, tmpdir: Path, docs: list[dict], title: str | None = None):
        embedder = StubEmbedder()
        builder = DenseIndexBuilder(embedder)
        builder.build_from_records(
            records=docs,
            citation_field="citation",
            text_field="text",
            title_field=title,
            output_dir=tmpdir,
        )
        return DenseIndex.load(tmpdir)

    def test_retrieve_returns_candidates(self):
        docs = [
            {"citation": "Art. 1 ZGB", "text": "Das Gesetz"},
            {"citation": "Art. 2 ZGB", "text": "Guter Glaube"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index = self._build_index(Path(tmpdir), docs)
            embedder = StubEmbedder()
            retriever = DenseRetriever(
                embedder=embedder,
                laws_index=index,
                courts_index=None,
            )
            results = retriever.retrieve("test query", top_k=5)
            assert len(results) > 0
            assert all(hasattr(c, "citation_raw") for c in results)

    def test_retrieve_includes_anchors(self):
        docs = [
            {"citation": "Art. 221 StPO", "text": "Haftgruende"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index = self._build_index(Path(tmpdir), docs)
            embedder = StubEmbedder()
            retriever = DenseRetriever(
                embedder=embedder,
                laws_index=index,
                courts_index=None,
            )
            results = retriever.retrieve(
                "Art. 221 Abs. 1 StPO applies here",
                top_k=10,
            )
            citations = [c.citation_raw for c in results]
            assert any("Art. 221" in c for c in citations)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval/test_dense_retriever.py -v
```

- [ ] **Step 3: Write the implementation**

File: `src/omnilex/retrieval/dense_retriever.py`

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_retrieval/test_dense_retriever.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omnilex/retrieval/dense_retriever.py tests/test_retrieval/test_dense_retriever.py
git commit -m "feat(retrieval): add DenseRetriever with anchor extraction and RRF fusion"
```

---

### Task 9: Corpus Embedding Script (`scripts/embed_corpus.py`)

CLI script for building dense indices from the raw CSV files. SLURM-friendly.

**Files:**
- Create: `scripts/embed_corpus.py`

- [ ] **Step 1: Write the script**

File: `scripts/embed_corpus.py`

```python
"""Build dense FAISS indices from raw corpus CSVs.

Usage:
    python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws
    python scripts/embed_corpus.py --corpus courts --output data/processed/bge_m3/courts
    python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws --limit 100
"""

import argparse
import csv
import sys
from pathlib import Path


def load_csv_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            records.append(dict(row))
    return records


def main():
    parser = argparse.ArgumentParser(description="Build dense FAISS index from corpus CSV")
    parser.add_argument("--corpus", choices=["laws", "courts"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()

    data_dir = Path("data")

    if args.corpus == "laws":
        csv_path = data_dir / "laws_de.csv"
        title_field = "title"
    else:
        csv_path = data_dir / "court_considerations.csv"
        title_field = None

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run from Omnilex-Agentic-Retrieval-Competition/")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    records = load_csv_records(csv_path, limit=args.limit)
    print(f"Loaded {len(records)} records")

    from omnilex.retrieval.models import BgeM3Embedder
    from omnilex.retrieval.dense_index import DenseIndexBuilder

    print(f"Loading model {args.model}...")
    embedder = BgeM3Embedder(model_name=args.model)

    builder = DenseIndexBuilder(embedder)
    print(f"Building index -> {args.output}")
    builder.build_from_records(
        records=records,
        citation_field="citation",
        text_field="text",
        title_field=title_field,
        output_dir=args.output,
        batch_size=args.batch_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a small sample (requires FlagEmbedding + faiss-cpu)**

```bash
cd Omnilex-Agentic-Retrieval-Competition
python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws --limit 10
```

- [ ] **Step 3: Commit**

```bash
git add scripts/embed_corpus.py
git commit -m "feat(scripts): add embed_corpus.py for building dense FAISS indices"
```

---

### Task 10: Evaluation Harness (`scripts/run_evaluation.py`)

Runs the full pipeline on val.csv and reports metrics at multiple top-k values.

**Files:**
- Create: `scripts/run_evaluation.py`

- [ ] **Step 1: Write the script**

File: `scripts/run_evaluation.py`

```python
"""Evaluate retrieval pipeline on val.csv.

Usage:
    python scripts/run_evaluation.py \
        --laws-index data/processed/bge_m3/laws \
        --val-csv data/val.csv \
        --top-k 2 3 5 10 \
        --output results/bgem3_laws_only.json
"""

import argparse
import csv
import json
from pathlib import Path

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.metrics import citation_f1, macro_f1
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.dense_retriever import DenseRetriever
from omnilex.retrieval.models import BgeM3Embedder


def load_val_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on val.csv")
    parser.add_argument("--laws-index", type=Path, default=None)
    parser.add_argument("--courts-index", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    normalizer = CitationNormalizer()

    print(f"Loading model {args.model}...")
    embedder = BgeM3Embedder(model_name=args.model)

    laws_index = DenseIndex.load(args.laws_index) if args.laws_index else None
    courts_index = DenseIndex.load(args.courts_index) if args.courts_index else None

    retriever = DenseRetriever(
        embedder=embedder,
        laws_index=laws_index,
        courts_index=courts_index,
    )

    queries = load_val_queries(args.val_csv)
    print(f"Evaluating on {len(queries)} queries")

    all_gold = []
    all_retrieved: dict[int, list] = {}

    for k in args.top_k:
        all_retrieved[k] = []

    for q in queries:
        gold_raw = q.get("gold_citations", "")
        gold_canonical = normalizer.canonicalize_list(
            [c.strip() for c in gold_raw.split(";") if c.strip()]
        )
        all_gold.append(gold_canonical)

        candidates = retriever.retrieve(q["query"], top_k=max(args.top_k), faiss_top_k=100)

        for k in args.top_k:
            top = candidates[:k]
            pred_canonical = normalizer.canonicalize_list([c.citation_raw for c in top])
            all_retrieved[k].append(pred_canonical)

    results = {"val_queries": len(queries)}

    for k in args.top_k:
        scores = macro_f1(all_retrieved[k], all_gold)
        results[f"top_{k}"] = scores
        print(f"  top-{k}: Macro F1={scores['macro_f1']:.4f}  "
              f"P={scores['macro_precision']:.4f}  R={scores['macro_recall']:.4f}")

    best_k = max(args.top_k, key=lambda k: results[f"top_{k}"]["macro_f1"])
    results["best_k"] = best_k
    results["per_query"] = []

    for i, q in enumerate(queries):
        pq = citation_f1(all_retrieved[best_k][i], all_gold[i])
        pq["query_id"] = q["query_id"]
        pq["query_preview"] = q["query"][:100]
        pq["num_gold"] = len(all_gold[i])
        pq["num_predicted"] = len(all_retrieved[best_k][i])
        results["per_query"].append(pq)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print(f"\nBest k={best_k} -> Macro F1={results[f'top_{best_k}']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat(scripts): add run_evaluation.py with multi-k comparison"
```

---

### Task 11: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new dependencies**

Append these lines to `requirements.txt`:

```
# Dense retrieval
FlagEmbedding>=1.2
faiss-cpu>=1.7.4
torch>=2.0
transformers>=4.40
```

- [ ] **Step 2: Install**

```bash
uv pip install -r requirements.txt
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add FlagEmbedding, faiss-cpu, torch, transformers"
```

---

## Chunk 4: Acceptance Tests and End-to-End Run

Verify the full pipeline before running the expensive 2.4M-passage embedding job.

---

### Task 12: Acceptance Tests (`tests/test_retrieval/test_acceptance.py`)

Tests from Section 10 of the spec — run on tiny fixtures before large indexing.

**Files:**
- Create: `tests/test_retrieval/test_acceptance.py`

- [ ] **Step 1: Write acceptance tests**

File: `tests/test_retrieval/test_acceptance.py`

```python
"""Acceptance tests — verify pipeline correctness on fixtures before large indexing."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.dense_index import DenseIndex, DenseIndexBuilder
from omnilex.retrieval.dense_retriever import DenseRetriever
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
```

- [ ] **Step 2: Run acceptance tests**

```bash
pytest tests/test_retrieval/test_acceptance.py -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval/test_acceptance.py
git commit -m "test(retrieval): add acceptance tests for pipeline correctness"
```

---

### Task 13: Run Full Pipeline End-to-End

This is the execution step — not code to write, but commands to run.

- [ ] **Step 1: Build laws index (~20 min)**

```bash
cd Omnilex-Agentic-Retrieval-Competition
python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws
```

- [ ] **Step 2: Verify laws index built correctly**

```bash
python -c "
from omnilex.retrieval.dense_index import DenseIndex
idx = DenseIndex.load('data/processed/bge_m3/laws')
print(f'Laws index: {len(idx.metadata)} passages, dim={idx.index.d}')
print(f'First citation: {idx.metadata[0][\"citation_raw\"]}')
"
```

- [ ] **Step 3: Run evaluation (laws only)**

```bash
python scripts/run_evaluation.py \
    --laws-index data/processed/bge_m3/laws \
    --val-csv data/val.csv \
    --top-k 2 3 5 10 \
    --output results/bgem3_laws_only.json
```

- [ ] **Step 4: Build courts index (~4-8 hours on GPU)**

On the SLURM cluster:

```bash
python scripts/embed_corpus.py --corpus courts --output data/processed/bge_m3/courts
```

Or test locally with a sample:

```bash
python scripts/embed_corpus.py --corpus courts --output data/processed/bge_m3/courts --limit 10000
```

- [ ] **Step 5: Run full evaluation (laws + courts)**

```bash
python scripts/run_evaluation.py \
    --laws-index data/processed/bge_m3/laws \
    --courts-index data/processed/bge_m3/courts \
    --val-csv data/val.csv \
    --top-k 2 3 5 10 \
    --output results/bgem3_full.json
```

- [ ] **Step 6: Compare with BM25 baseline**

Run the existing BM25 baseline on val.csv for comparison numbers.

- [ ] **Step 7: Commit results**

```bash
git add results/
git commit -m "results: BGE-M3 vs BM25 evaluation on val.csv"
```

---

## Summary

| Task | What it produces | Est. time |
|------|-----------------|-----------|
| 1 | `models.py` — EmbeddingModel/RerankerModel protocols | 15 min |
| 2 | `fusion.py` — RRF fusion + dedup | 30 min |
| 3 | `anchor_extractor.py` — citation extraction from queries | 30 min |
| 4 | `submission.py` — CSV generator | 20 min |
| 5 | `__init__.py` update | 5 min |
| 6 | `BgeM3Embedder` adapter | 20 min |
| 7 | `dense_index.py` — FAISS builder + loader | 45 min |
| 8 | `dense_retriever.py` — end-to-end retriever | 30 min |
| 9 | `embed_corpus.py` — CLI embedding script | 20 min |
| 10 | `run_evaluation.py` — evaluation harness | 30 min |
| 11 | requirements.txt update | 5 min |
| 12 | Acceptance tests | 30 min |
| 13 | Run the pipeline (laws: 20 min, courts: 4-8 hours) | variable |

**Total coding time:** ~4-5 hours
**Total compute time:** ~5-9 hours (dominated by court embedding)
