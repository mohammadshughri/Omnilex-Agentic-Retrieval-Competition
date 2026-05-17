# Task Assignment — BGE-M3 Pipeline (May 21 deadline)

## Parallel Timeline

```
Hour 0    A: Task 1 ──► Task 11 ──► Task 6 ──► Task 7
          B: Task 2 ──► Task 3 ──────────────► Task 4 ──► Task 5

Hour 2    C: ─────────────────────────────────────────────► Task 8 ──► Task 9 ──► Task 10
          A: ─────────────────────────────────────────────► embed laws (GPU ~20 min)
          B: ─────────────────────────────────────────────► Task 12 ──► submit SLURM

Hour 4    C: ─────────────────────────────────────────────► laws-only val.csv eval (first F1)

Hour 12+  C: ─────────────────────────────────────────────► full eval once courts job finishes
```

> C is unblocked once **A:Task 7** and **B:Task 2 + B:Task 3** are committed.

## Person A — Model + FAISS Infrastructure

**Start here:** Create `src/omnilex/retrieval/models.py` (new file) with the `EmbeddingModel` and `RerankerModel` abstract base classes. Everything downstream depends on this interface existing.

| # | File | Notes |
|---|------|-------|
| 1 | `src/omnilex/retrieval/models.py` | EmbeddingModel / RerankerModel abstract protocols — do this first |
| 11 | `requirements.txt` | Append: FlagEmbedding>=1.2, faiss-cpu>=1.7.4, scipy, torch>=2.0, transformers>=4.40 |
| 6 | `src/omnilex/retrieval/models.py` (extend) | Add BgeM3Embedder class; call `encode()` with `return_sparse=True`; store `last_sparse_weights` |
| 7 | `src/omnilex/retrieval/dense_index.py` | Use `IndexHNSWFlat(dim, 32, METRIC_INNER_PRODUCT)` not IndexFlatIP; save sparse weights via `scipy.sparse.save_npz` to `sparse.npz` |
| **Run** | `scripts/embed_corpus.py --corpus laws` | Needs GPU; ~20 min; single pass produces `faiss.index` + `sparse.npz` + `metadata.jsonl` |

## Person B — Fusion + Submission + Tests

**Start here:** Create `src/omnilex/retrieval/fusion.py` (new file) with the `Candidate` dataclass and `rrf_fuse()` function. This has no dependencies and the `Candidate` type is used everywhere downstream.

| # | File | Notes |
|---|------|-------|
| 2 | `src/omnilex/retrieval/fusion.py` | `Candidate` dataclass + `rrf_fuse()` + `deduplicate_candidates()` — do this first |
| 3 | `src/omnilex/retrieval/anchor_extractor.py` | Regex for `Art. X BOOK` and `BGE VOL SEC PAGE`; normalize via CitationNormalizer |
| 4 | `src/omnilex/retrieval/submission.py` | `select_top_k()` + `generate_submission()` → writes `query_id,predicted_citations` CSV |
| 5 | `src/omnilex/retrieval/__init__.py` | Add exports for fusion, anchor_extractor, submission, models, dense_index, dense_retriever |
| 12 | `tests/test_retrieval/test_acceptance.py` | Pipeline correctness tests; add one sparse round-trip test (load sparse.npz, score a query) |
| **Run** | `scripts/embed_corpus.py --corpus courts` on SLURM | Needs GPU; 4-8 hr; submit once Task 12 passes |

## Person C — Retriever + Evaluation

**Start here:** Wait for A:Task 7 and B:Task 2 + B:Task 3 to be committed (~2 hrs), then create `src/omnilex/retrieval/dense_retriever.py`. While waiting, read the plan doc and the spec to understand the 3-channel RRF flow.

| # | File | Notes |
|---|------|-------|
| 8 | `src/omnilex/retrieval/dense_retriever.py` | Load `faiss.index` + `sparse.npz`; encode query for dense + sparse; add anchor channel; `rrf_fuse()` all three — **Option 2 core change** |
| 9 | `scripts/embed_corpus.py` | CLI: `--corpus laws\|courts`, `--output`, `--limit` (for smoke tests), `--batch-size` |
| 10 | `scripts/run_evaluation.py` | Load val.csv; run retriever; compute macro_f1 at k=2,3,5,10; save JSON to `results/` |
| **Run** | `python scripts/run_evaluation.py --laws-index data/processed/bge_m3/laws --val-csv data/raw/lexam/val.csv` | Laws-only F1 — first signal |
| **Run** | Add `--courts-index` once B's SLURM job finishes | Full hybrid F1 for May 21 presentation |

## Option 2 changes vs. original plan
- `dense_index.py` — HNSW swap + scipy CSR sparse save
- `models.py` — BgeM3Embedder enables sparse in same embedding pass
- `dense_retriever.py` — loads sparse.npz, adds sparse channel before RRF

## Implementation reference
Full code for each task: `docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md`
