# Implementation Strategy: Cross-Lingual Legal Citation Retrieval

**Date:** 2026-05-17
**Project:** Omnilex Agentic Legal Information Retrieval (LUH Agentic AI Course)
**Team:** 3 people
**Timeline:** 2026-04-30 to 2026-07-16 (8 weeks remaining)

---

## 1. Strategy: Progressive Pipeline

Build incrementally, starting with BGE-M3 hybrid retrieval. Each week adds a measurable improvement.

### Rationale

- Minimizes risk for the first progress presentation (May 21)
- Each week produces a clear delta over the previous week
- BGE-M3 (0.6B params) is fast to embed, strong cross-lingually, and provides hybrid retrieval natively
- Larger models (Qwen3-8B) are introduced once the pipeline is proven

---

## 2. Week-by-Week Roadmap

| Week | Dates | Milestone | Deliverable |
|------|-------|-----------|-------------|
| 1-3 | Apr 30 - May 21 | BGE-M3 hybrid retrieval end-to-end | Presentation: BM25 vs. BGE-M3 F1 on val set |
| 3 | May 22 - May 28 | Add reranker (bge-reranker-v2-m3) | F1 improvement; ablation: dense vs. hybrid vs. hybrid+rerank |
| 4 | May 29 - Jun 4 | Add HyDE (Architecture C) | Compare with/without HyDE for conceptual queries |
| 5-6 | Jun 5 - Jun 18 | Upgrade to Qwen3-8B on cluster | BGE-M3 vs. Qwen3-8B comparison |
| 7 | Jun 19 - Jun 25 | Citation count calibration + query routing | Dynamic thresholding; route lexical vs. conceptual queries |
| 8 | Jun 26 - Jul 16 | Final polish, ablation study, final presentation | Complete pipeline diagram + ablation table |

---

## 3. Architecture: Week 1-2 (BGE-M3 Hybrid Retrieval)

### Pipeline

```
English Legal Query
    |
    +--> Citation-anchor extractor (regex for Art./BGE patterns)
    |    --> normalize via CitationNormalizer --> high-priority candidates
    |
    v
BGE-M3 Encoder (0.6B, BAAI/bge-m3)
    |-- Dense embedding (1024-dim)
    |-- Sparse neural weights (cross-lingual lexical)
    +-- ColBERT multi-vector tokens
    |
    v
FAISS Index (pre-built, two separate indices)
    |-- Laws index: 175K passages from laws_de.csv
    +-- Courts index: 2.4M passages from court_considerations.csv
    |
    v
Top-100 candidates per channel (dense, sparse, ColBERT each ranked independently)
    |
    v
Reciprocal Rank Fusion per corpus (score = sum of 1/(60 + rank) across channels)
    |
    v
Federated law/court merge with deduplication
    |
    v
Score threshold or fixed top-k --> predicted citations
    |
    v
CitationNormalizer (existing) --> submission.csv
    |
    v
Scorer (existing) --> Macro F1
```

### Why BGE-M3

- Single model produces three complementary retrieval signals
- Cross-lingual by design: English "detention" maps near German "Haft" in embedding space
- 0.6B params fits any GPU; embeds 2.4M passages in 4-8 hours
- Hybrid scoring combines semantic understanding (dense), cross-lingual lexical matching (sparse), and exact token matching (ColBERT)
- Well-supported via the `FlagEmbedding` Python library

### Why two separate indices

- Laws (175K rows, short German, ~242 chars) and courts (2.4M rows, multilingual, ~1,105 chars) have different characteristics
- Train gold: 70% laws / 30% courts -- separate indices allow differential weighting
- Embedding laws takes ~20 min; courts take 4-8 hours. Separate indices allow iteration on laws while courts embed
- Failure isolation: a crashed court embedding job doesn't lose the laws index

---

## 4. Data Pipeline & Storage

### Embedding storage

```
data/processed/bge_m3/
    laws_dense.npy          # (175K, 1024) fp16, ~340 MB
    laws_sparse.pkl         # sparse weight dicts, ~50-100 MB
    laws_faiss.index        # HNSW (M=32), ~500 MB
    laws_metadata.jsonl     # one JSON object per line, keyed by FAISS row index

    courts_dense.npy        # (2.4M, 1024) fp16, ~4.6 GB
    courts_sparse.pkl       # ~1-2 GB
    courts_faiss.index      # HNSW (M=32), ~7-8 GB
    courts_metadata.jsonl   # one JSON object per line, keyed by FAISS row index
```

### Metadata record format

Each line in `*_metadata.jsonl` corresponds to the same row index in the FAISS index and embeddings array:

```json
{"idx": 0, "citation_raw": "Art. 1 ZGB", "source_row": 42, "text_preview": "Das Recht findet auf alle..."}
```

Fields:
- `idx`: FAISS row index (0-based, matches position in `*_dense.npy`)
- `citation_raw`: the original citation string from the corpus, passed to `CitationNormalizer` at scoring time
- `source_row`: row number in the source CSV (`laws_de.csv` or `court_considerations.csv`)
- `text_preview`: first 200 chars of the passage text (for debugging / presentation examples)
```

**Total storage: ~15 GB** (without ColBERT).

### ColBERT handling

- ColBERT stores a vector per token per passage. For 2.4M passages this exceeds available memory.
- Decision: skip ColBERT for courts initially. Use `0.6 * dense + 0.4 * sparse` for courts.
- Use full hybrid (dense + sparse + ColBERT) for laws (175K passages, manageable).
- ColBERT can be applied at rerank time on top-100 candidates in later weeks.

### FAISS index type

- IndexHNSWFlat (M=32): ~98% recall, no training needed, fast queries
- Before committing to HNSWFlat for courts: measure actual RAM on a 100K sample and extrapolate. If memory exceeds cluster allocation, fall back to IndexIVFFlat or int8 quantization
- For potential Kaggle submission: quantize to int8

---

## 5. New Code Structure

```
src/omnilex/retrieval/
    bm25_index.py          # existing
    tools.py               # existing
    models.py              # NEW: EmbeddingModel / RerankerModel protocols + adapters
    dense_index.py         # NEW: embedding + FAISS index builder (model-agnostic)
    dense_retriever.py     # NEW: query encoding + search
    fusion.py              # NEW: RRF fusion + candidate deduplication
    anchor_extractor.py    # NEW: citation-anchor extraction from query text
    submission.py          # NEW: top-k -> normalize -> CSV

scripts/
    embed_corpus.py        # NEW: SLURM-friendly batch embedding script
    run_evaluation.py      # NEW: comparison evaluation harness

results/
    bm25_baseline.json     # evaluation output
    bgem3_hybrid.json      # evaluation output
```

### Component responsibilities

**`models.py`** -- Model protocols and adapters
- `EmbeddingModel` protocol: `encode_documents(texts) -> vectors`, `encode_queries(queries) -> vectors`
- `RerankerModel` protocol: `score_pairs(query, documents) -> probabilities`
- `BgeM3Embedder` adapter (first implementation)
- Qwen3 adapters added in week 5-6 behind the same interface

**`dense_index.py`** -- `DenseIndexBuilder`
- Accepts any `EmbeddingModel` (not hard-coded to BGE-M3)
- Compose indexed text as `citation + " " + title + " " + text` (laws) or `citation + " " + text` (courts)
- Batch-encode passages (configurable batch size for GPU memory)
- Build FAISS IndexHNSWFlat from dense embeddings
- Save embeddings, sparse weights, and FAISS index to disk
- Resumable: checkpoint after every N batches

**`dense_retriever.py`** -- `DenseRetriever`
- Load pre-built indices from disk
- Encode query via `EmbeddingModel`
- Search both FAISS indices (laws + courts)
- Return per-channel candidate lists (dense, sparse ranked independently)

**`fusion.py`** -- `RRFFusion`
- Reciprocal Rank Fusion: `score = sum(1 / (60 + rank))` across channels
- Per-corpus fusion first, then federated law/court merge
- Deduplicate by normalized citation string
- Return unified ranked candidate list

**`anchor_extractor.py`** -- `AnchorExtractor`
- Regex extraction of `Art. X [Abs. Y] BOOK` and `BGE VOL SEC PAGE` patterns from query text
- Normalize via existing `CitationNormalizer`
- Return as high-priority candidate list (injected into fusion at rank 0)

**`submission.py`** -- `SubmissionGenerator`
- Accept retrieval results
- Apply score threshold or fixed top-k (test k=2,3,5,10 on val from v1)
- Normalize citations via CitationNormalizer
- Write submission CSV

**`embed_corpus.py`** -- SLURM script
- Parse corpus CSV files
- Batch embed with progress tracking
- Save intermediate checkpoints
- SLURM-compatible: handles job time limits gracefully

**`run_evaluation.py`** -- Evaluation harness
- Load val.csv gold labels
- Run specified retrieval pipeline on each query
- Compute Macro F1, Micro F1, Precision, Recall
- Output per-query breakdown + aggregate (label as calibration check, not robust evidence)
- Save to results/ as JSON

---

## 6. Task Division (Week 1-2, 3 People)

| Person | Task | Depends on | Output |
|--------|------|------------|--------|
| A | Embed laws_de.csv + build FAISS index | BGE-M3 model download | `laws_dense.*`, `laws_faiss.index` |
| B | Embed court_considerations.csv on SLURM | BGE-M3 model download, SLURM setup | `courts_dense.*`, `courts_faiss.index` |
| C | Build query pipeline + evaluation harness | A's output (laws index first) | `dense_retriever.py`, `run_evaluation.py`, results/ |

Person A finishes first (~20 min for 175K passages), then helps C test the pipeline. Person B runs the large embedding job on SLURM (4-8 hours). Integration happens once courts index is ready.

---

## 7. Evaluation Plan

### Metrics

- **Macro F1** (primary, competition metric)
- Micro F1 (secondary)
- Per-query Precision, Recall, F1

### Comparisons for May 21 presentation

| Method | Expected Macro F1 |
|--------|-------------------|
| BM25 baseline (English query, no translation) | ~0.02 (near zero for conceptual queries) |
| BGE-M3 dense only | 0.08-0.15 |
| BGE-M3 dense + sparse | 0.10-0.20 |
| BGE-M3 hybrid (laws: full, courts: dense+sparse) | 0.15-0.25 |

These are estimates. The key story is the relative improvement over BM25.

### Presentation outline (May 21)

1. The problem: English queries vs. German/French/Italian corpus (1 slide)
2. Why BM25 fails cross-lingually (1 slide)
3. Our approach: BGE-M3 hybrid retrieval (2 slides)
4. Results: BM25 vs. BGE-M3 comparison table (1-2 slides)
5. Example: one query end-to-end walkthrough (1 slide)
6. Roadmap: reranker, HyDE, Qwen3-8B, calibration (1 slide)

---

## 8. Dependencies

### Python packages (add to requirements.txt)

```
FlagEmbedding>=1.2
faiss-cpu>=1.7.4       # or faiss-gpu on cluster
torch>=2.0
transformers>=4.40
```

### External resources

- BGE-M3 model: `BAAI/bge-m3` from HuggingFace (~1.2 GB download)
- GWDG/KISSKI cluster account with SLURM access
- Cluster GPU: any NVIDIA GPU with 8+ GB VRAM

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SLURM queue delays (courts embedding takes too long to start) | Medium | Start job early; test with 10K sample first; have CPU fallback |
| BGE-M3 cross-lingual quality disappointing on legal text | Low | Dense-only mode still works; upgrade to Qwen3-8B in week 5 |
| Disk quota exceeded on cluster | Low | Use lean config (~9 GB); compress embeddings to int8 |
| val.csv too small (10 queries) for reliable evaluation | High (inherent) | Report confidence intervals; use per-query breakdown; supplement with train subset |
| Citation count threshold uncalibrated until Week 7 | High | Use train median (k=2) as initial fixed-k; test k=3,5,10 on val set early; full calibration in Week 7 |

---

## 10. Acceptance Tests (Before Large Indexing)

Before running the full 2.4M court embedding job, verify on a 100-row sample:

- CSV row id to FAISS id alignment (metadata idx matches embedding row)
- Metadata lookup by FAISS id returns correct `citation_raw`
- `CitationNormalizer` successfully parses retrieved raw citations
- Duplicate citation collapse works (same citation from different passages)
- RRF fusion is deterministic (same input -> same output)
- Submission CSV validates via existing `validate_submission.py`
- Fixed top-k bounds work (k=2,3,5,10 all produce valid output)

---

## 11. Architecture Review Reference

This spec incorporates findings from `2026-05-17-architecture-review-findings-for-claude.md`. Key changes adopted from that review:

1. RRF fusion instead of raw weighted score fusion
2. Citation-anchor extraction as a first-class candidate channel
3. Model-pluggable `EmbeddingModel` / `RerankerModel` protocols
4. Text composition: `citation + title + text` for indexed passages
5. Dynamic k tested from v1 (not deferred to Week 7)
6. Acceptance tests before large indexing jobs

Deferred to post-presentation: SQLite metadata, CSR sparse storage, compressed FAISS, chunked MaxSim courts. These will be evaluated once measurements prove they are needed.

---

## 12. Future Weeks (Preview)

- **Week 3:** Add `bge-reranker-v2-m3` -- rerank top-100 to top-k. Expected +0.03-0.05 F1.
- **Week 4:** HyDE -- LLM generates synthetic German law passage, embed that instead of English query. Expected +0.02-0.05 F1 on conceptual queries.
- **Week 5-6:** Qwen3-Embedding-8B on cluster -- compare with BGE-M3 head-to-head.
- **Week 7:** Citation count calibration -- dynamic threshold instead of fixed top-k. Query routing: lexical (BM25) vs. conceptual (dense) based on whether query contains article numbers.
- **Week 8:** Final ablation study showing each component's contribution.
