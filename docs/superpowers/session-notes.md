# BGE-M3 Pipeline — Session Implementation Notes

Running notes on what each session built, why decisions were made, and what the verification results mean. Updated at the end of each session.

**Reference plan:** `docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md`
**Progress tracker:** `docs/superpowers/plans/2026-05-18-session-split-execution.md`

---

## Background: The Problem

This is a **cross-lingual legal citation retrieval** task. Given an English legal query, the system must return the correct Swiss law citations from two corpora:

- **Federal laws** (`laws_de.csv`): 176K rows, ~99% German, articles prefixed `Art.`
- **Court decisions** (`court_considerations.csv`): 2.4M rows, 61% German / 32% French / 6% Italian, prefixed `BGE`

The core difficulty: queries are English, documents are German/French/Italian. Keyword search (BM25) fails because the words simply don't overlap. The solution is **BGE-M3**, a multilingual embedding model that maps text from 65+ languages into a shared 1024-dimensional vector space — so "pre-trial detention" (English) and "Untersuchungshaft" (German) end up as nearby vectors.

The competition metric is **Macro F1**: F1 is computed per query, then averaged. Each query gets equal weight regardless of how many citations it has.

---

## Session 1 — Pure Logic Foundation

**Date:** 2026-05-18
**Tasks:** 1 (model protocols), 2 (RRF fusion), 3 (anchor extractor), 4 (submission CSV), 5 (`__init__.py`), 11 (requirements.txt)
**Result:** 23 tests passing, no GPU required

### What was built

#### `src/omnilex/retrieval/models.py` — Protocol definitions

Defines the interface that any embedding or reranking model must implement. Uses Python Abstract Base Classes (ABCs):

```python
class EmbeddingModel(ABC):
    embedding_dim: int          # e.g. 1024 for BGE-M3
    encode_documents(texts)     # list[str] → float32 matrix (N × dim)
    encode_queries(queries)     # list[str] → float32 matrix (N × dim)

class RerankerModel(ABC):
    score_pairs(query, documents)  # str, list[str] → float32 array (N,)
```

Why this matters: all downstream code depends only on these interfaces, not on any specific model. Swapping BGE-M3 for Qwen3-Embedding means writing one new class — no other code changes.

#### `src/omnilex/retrieval/fusion.py` — Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from multiple retrieval channels (dense search, sparse search, anchor extraction) into one list. The score formula:

```
RRF_score(candidate) = Σ over channels:  1 / (k + rank_in_channel)
```

`k = 60` (from the original RRF paper, 2009). Rank is 1-indexed. A candidate that appears in multiple channels accumulates contributions from each — so cross-channel agreement is rewarded. The constant `k` dampens the advantage of being ranked #1 vs #2.

The `Candidate` dataclass stores: `citation_raw`, `score` (original retrieval score), `source` (which channel), `rrf_score` (fused score).

`deduplicate_candidates` keeps the highest-scoring occurrence of each citation string, preserving first-seen order.

#### `src/omnilex/retrieval/anchor_extractor.py` — Regex citation extraction

Scans query text for explicit citation references and returns them as high-confidence candidates. Two regex patterns:

- `Art. X [Abs. Y] [lit. z] BOOK` → federal law articles (`BOOK` = 2–10 uppercase letter abbreviation)
- `BGE VOL SECTION PAGE` → court decisions

Matched strings pass through `CitationNormalizer` to strip sub-paragraph elements (`lit.`, `Ziff.`) and produce canonical forms. These are free precision points: if the query names a citation explicitly, it is almost certainly relevant.

Example: `"Does Art. 221 Abs. 1 StPO apply? See also BGE 137 IV 122."` → `["Art. 221 Abs. 1 StPO", "BGE 137 IV 122"]`

#### `src/omnilex/retrieval/submission.py` — Competition CSV generator

Takes a `dict[query_id → list[Candidate]]`, selects top-k, normalizes via `CitationNormalizer`, and writes the required format:

```
query_id,predicted_citations
q1,Art. 1 ZGB;BGE 116 Ia 56;Art. 11 Abs. 2 OR
```

`CitationNormalizer` (pre-existing) strips `lit.`/`Ziff.` because gold labels stop at paragraph level — `Art. 221 Abs. 1 lit. b StPO` and `Art. 221 Abs. 1 StPO` must match.

### Session 1 smoke test output explained

```
Anchors: ['Art. 221 Abs. 1 StPO', 'BGE 137 IV 122']
```
Both citations extracted and normalized correctly from the query string.

```
Fused top-3: [('Art. 1 ZGB', 0.0325), ('Art. 221 StPO', 0.0164), ('BGE 137 IV 122', 0.0161)]
```
`Art. 1 ZGB` ranked first because it appeared in **both** the dense and sparse channels:

| Citation | Dense rank | Sparse rank | RRF contribution | Total |
|---|---|---|---|---|
| Art. 1 ZGB | 2nd → 0.0161 | 1st → 0.0164 | both channels | **0.0325** |
| Art. 221 StPO | 1st → 0.0164 | — | dense only | 0.0164 |
| BGE 137 IV 122 | — | 2nd → 0.0161 | sparse only | 0.0161 |

```
CSV:
query_id,predicted_citations
q1,Art. 1 ZGB;Art. 221 StPO;BGE 137 IV 122
```
Valid competition format. Ready to submit as-is.

---

## Session 2 — FAISS Dense Index

**Date:** 2026-05-18
**Tasks:** 6 (BgeM3Embedder), 7 (DenseIndexBuilder + DenseIndex)
**Result:** 27 tests passing, 4 skipped (BgeM3 integration tests — FlagEmbedding not installed in CI)

### What was built

#### `BgeM3Embedder` (appended to `models.py`)

A concrete `EmbeddingModel` that wraps `BGEM3FlagModel` from the `FlagEmbedding` library. Key design decisions:

- **Lazy import inside `__init__`** — `from FlagEmbedding import BGEM3FlagModel` is inside the constructor, not at module top level. This means importing `omnilex.retrieval.models` never fails even if `FlagEmbedding` is not installed. The failure only happens when you actually try to instantiate `BgeM3Embedder`.

- **`return_sparse=True` flag** — BGE-M3 can produce both dense vectors and sparse lexical weights in one forward pass. When `return_sparse=True`, the sparse weights (a dict mapping token IDs to weights) are stored in `self.last_sparse_weights`. This is the foundation for a future third RRF channel: sparse BGE-M3 retrieval, which outperforms BM25 on cross-lingual queries.

- **`use_fp16=True`** — uses 16-bit floating point for inference, halving GPU memory usage with negligible accuracy loss.

- **`embedding_dim = 1024`** — BGE-M3's fixed output dimension. All FAISS indices built with this embedder will have `d=1024`.

The `TestBgeM3Embedder` tests use `pytest.importorskip("FlagEmbedding")` so they skip gracefully when the library is not installed. They only run when `FlagEmbedding` is available and the model has been downloaded.

#### `src/omnilex/retrieval/dense_index.py` — FAISS builder and loader

**`DenseIndexBuilder.build_from_records`** — the indexing pipeline:

1. **Text composition**: concatenate `citation + title + text` per record (e.g. `"Art. 221 StPO Haft Untersuchungshaft wegen Kollusionsgefahr"`). Including the citation string itself helps the model align query mentions with document content.

2. **Batch encoding**: passes texts through `embedder.encode_documents()` in chunks of `batch_size=256`. Returns a float32 matrix of shape `(N, 1024)`.

3. **Save raw embeddings**: writes `dense.npy` — the un-normalized vectors. Kept for potential future re-indexing without re-embedding.

4. **L2-normalize**: calls `faiss.normalize_L2(embeddings)` on a copy. After normalization, inner product equals cosine similarity — the standard pattern for semantic similarity with FAISS.

5. **Build `IndexFlatIP`**: exact inner product search. Compares every query against every vector — no approximation. At 176K laws passages this takes ~100ms per query. `IndexHNSWFlat` (approximate) was explicitly rejected: it adds approximation error and tuning parameters for no practical gain at this scale.

6. **Write `metadata.jsonl`**: one JSON object per line, in the same row order as the FAISS index. Fields: `idx` (integer, matches FAISS row number), `citation_raw` (the raw citation string), `text_preview` (first 200 chars). The row order is the critical invariant — FAISS returns integer row indices and you need this file to resolve which citation each index refers to.

**`DenseIndex.load` + `.search`**:

- `load` reads all three files back from disk.
- `search` L2-normalizes the query vector, calls `index.search(qv, top_k)`, then uses returned integer indices to look up `metadata`. Returns a list of dicts with the metadata fields plus a `score` key.

### Session 2 smoke test output explained

```
Dense search top-2: [('Art. 221 StPO', 1.0), ('Art. 1 ZGB', 0.0)]
```

The `StubEmbedder` assigns orthogonal unit vectors by position:
- Doc 0 (Art. 221 StPO)  → `[1, 0, 0, 0]`
- Doc 1 (Art. 1 ZGB)     → `[0, 1, 0, 0]`
- Doc 2 (BGE 137 IV 122) → `[0, 0, 1, 0]`

The query vector is `[1, 0, 0, 0]`. After L2 normalization (already unit length), inner product = cosine similarity:
- vs Doc 0: `[1,0,0,0]·[1,0,0,0] = 1.0` — perfect match
- vs Doc 1: `[1,0,0,0]·[0,1,0,0] = 0.0` — orthogonal, no similarity

A score of **1.0** means the query and document vectors are identical after normalization. A score of **0.0** means they share no signal at all. In real usage with BGE-M3, scores will typically range from roughly 0.3 to 0.95.

This confirms the full round-trip is correct: build → save → load → search → metadata lookup.

---

## What Comes Next

### Session 3 — Retriever + Scripts (Tasks 8–10)

Creates `DenseRetriever`: the orchestrator that takes a raw query string and returns ranked `Candidate` objects by:
1. Extracting citation anchors from the query text
2. Encoding the query with the embedder
3. Searching the laws and/or courts FAISS indices
4. Running RRF fusion across all channels
5. Deduplicating and returning top-k

Also creates two CLI scripts:
- `scripts/embed_corpus.py` — builds FAISS indices from the raw CSV files (~20 min for laws on GPU, 4–8 hrs for courts)
- `scripts/run_evaluation.py` — runs the full pipeline on `val.csv` and reports Macro F1 at multiple top-k values

### Session 4 — Acceptance Tests + Laws Embedding (Task 12 + Task 13 steps 1–3)

Requires GPU. Writes `tests/test_retrieval/test_acceptance.py` (end-to-end pipeline correctness checks on small fixtures), then embeds all 176K law passages and runs the first real evaluation on the 10 validation queries to get an actual Macro F1 score.

### Session 5 — Courts Evaluation (Task 13 steps 4–7)

Embeds the 2.4M court decisions (4–8 hours unattended on GPU), then runs the full hybrid evaluation (laws + courts) and compares against the BM25 baseline to measure the uplift from dense retrieval.
