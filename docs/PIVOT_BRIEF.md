# Omnilex Competition — Team Pivot Brief
**Date:** May 31, 2026 · **Status:** Active

---

## 1. Previous Approach — What We Built

### What it was
We built a **dense retrieval pipeline** (Architecture A) based on BGE-M3, a state-of-the-art multilingual embedding model. The pipeline embeds both the query and every document in the two corpora (Swiss federal laws + court decisions) into a shared vector space, then retrieves the top-k closest documents using FAISS.

### Components implemented (still in the codebase)
| Component | File | Status |
|-----------|------|--------|
| BGE-M3 embedder | `src/omnilex/retrieval/models.py` | Built ✓ |
| FAISS dense index | `src/omnilex/retrieval/dense_index.py` | Built ✓ |
| Dense retriever | `src/omnilex/retrieval/dense_retriever.py` | Built ✓ |
| Reciprocal Rank Fusion | `src/omnilex/retrieval/fusion.py` | Built ✓ |
| Regex citation extractor | `src/omnilex/retrieval/anchor_extractor.py` | Built (weak) |
| Submission generator | `src/omnilex/retrieval/submission.py` | Built ✓ |
| Evaluation scorer | `src/omnilex/evaluation/scorer.py` | Built ✓ |

### What it scored
- **Macro F1 on validation set:** 0.0435 (at k=10)
- **Leaderboard position:** Bottom tier

### Why it was not enough
The pipeline is the best pure retrieval approach possible, but retrieval has a hard ceiling:

| Approach | Max achievable F1 |
|----------|-------------------|
| Any retrieval method (empirically measured) | ~0.065 |
| LLM-based approaches (leaderboard leaders) | 0.336 – 0.359 |
| Oracle at k=25 (perfect retrieval, upper bound) | 0.779 |

The root cause is that this is **not a document retrieval problem**. The task requires predicting which Swiss legal citations apply to a scenario — like a lawyer reasoning from a set of facts, not a search engine finding similar documents.

Additional structural problem: training queries are in German (mean 4.1 citations/query) while val/test queries are in English (mean 25.1 citations/query). All training-based signals — TF-IDF, co-citation graphs, frequency statistics — massively underpredict on val/test because of this 6× distribution shift.

---

## 2. New Investigation — What We Are Now Doing

### Core hypothesis
> LLM reasoning, not retrieval ranking, is what separates the top teams from the baseline. The winning approach is to let an LLM read the query and reason about which Swiss laws and BGE decisions logically apply — the same way a Swiss lawyer would.

This is confirmed by two pieces of evidence:
- "BetterCallAgent" (3rd place, 0.336) — the name explicitly signals an agentic LLM approach
- The leaderboard jump from 0.065 (retrieval ceiling, all public notebooks) to 0.359 happened exactly when LLM-based submissions appeared

### New architecture: 4-layer pipeline

```
Query
  │
  ├─► [Layer 1] Regex Extraction
  │     Extract Art. X BOOK / BGE citations directly mentioned in query text.
  │     Apply FR→DE abbreviation mapping (LAI→IVG, CPC→ZPO, etc.)
  │     Current recall: 1.2% — needs improvement.
  │
  ├─► [Layer 2] Dense Retrieval at k=25
  │     BGE-M3 on laws + courts corpora (existing pipeline, already built).
  │     Key change: k=10 → k=25.
  │     Oracle F1 improves from 0.644 to 0.779 with this one change.
  │
  ├─► [Layer 3] LLM Reasoning  ← THE DIFFERENTIATOR
  │     Input: query + Layer 1 hits + Layer 2 top-25 candidates.
  │     LLM reasons about the legal scenario and predicts the full citation list.
  │     Uses university-provided API (BLOCKED until API docs are received).
  │
  └─► [Layer 4] RRF Fusion + Adaptive k
        Merge all three layers with Reciprocal Rank Fusion (already built).
        Adaptive k: predict citation count per query.
        Val set mean = 25.1 citations, range 10–47. Fixed k=10 misses ~60% on average.
```

### Verification rule
We do not build the next layer until the current one is empirically verified on `data/val.csv`. No guessing, no wasted work.

| Step | Gate to pass before proceeding |
|------|-------------------------------|
| Layer 1 | Recall > 1.2% on val gold citations |
| Layer 2 | Macro F1 at k=25 > Macro F1 at k=10 (currently 0.0435) |
| Layer 3 | At least one val query returns citations that match gold |
| Layer 4 | Full pipeline Macro F1 > 0.15 on val.csv |

---

## 3. Task Assignments

> Replace the blanks with actual team member names.

### Layer 1 — Regex Extraction v2
**Goal:** Replace the current weak extractor (`anchor_extractor.py`, 1.2% recall) with an improved version that handles French abbreviations and bare article patterns.

| # | Task | File | Details |
|---|------|------|---------|
| L1-1 | Study reference implementation | `notebooks_research/dasdasdada__hybrid.../` | Read the FR→DE mapping dict and the bare-article regex patterns in the reference notebook |
| L1-2 | Implement improved extractor | `src/omnilex/retrieval/anchor_extractor.py` | (1) Bare article + nearest abbreviation within 150 chars, (2) FR→DE mapping (40+ rules), (3) BGE citation pattern, (4) case number pattern |
| L1-3 | Verify on val.csv | — | Run extractor against `data/val.csv` gold. Report: precision, recall, F1. Must beat 1.2% recall to proceed. |

**Assigned to:** ___________
**Deliverable:** Updated `anchor_extractor.py` + verification numbers reported to team.

---

### Layer 2 — Dense Retrieval at k=25
**Goal:** Re-run the existing BGE-M3 pipeline with k=25 and measure the improvement. This requires GPU access.

| # | Task | File | Details |
|---|------|------|---------|
| L2-1 | Embed laws corpus | `scripts/embed_corpus.py` | `--corpus laws --output data/processed/bge_m3/laws` — ~20 min on a T4 GPU |
| L2-2 | Embed courts corpus | `scripts/embed_corpus.py` | `--corpus courts --output data/processed/bge_m3/courts` — 4–8 hrs on GPU (submit as Kaggle notebook or SLURM job) |
| L2-3 | Evaluate k=10 vs k=25 | `scripts/run_evaluation.py` | Run at both k values. Report Macro F1 for each. |
| L2-4 | Confirm oracle ceiling | — | Check oracle F1 at k=25 ≥ 0.779. If lower, re-check embedding quality. |

**Assigned to:** ___________
**Deliverable:** Macro F1 numbers at k=10 and k=25 on val.csv.
**Note:** Use Kaggle notebook (free T4 GPU) if no local GPU is available.

---

### Layer 3 — LLM Reasoning via University API
**Goal:** Wire up the university-provided LLM to reason over query + candidate list and predict citations.

| # | Task | File | Details |
|---|------|------|---------|
| L3-0 | **Share API docs** | — | **Critical blocker.** Share the API endpoint, auth method, and model name with the whole team before anything else. |
| L3-1 | Build prompt template | `src/omnilex/llm/prompts.py` | System + user prompt that gives the LLM: (1) the legal query, (2) candidate citations from Layer 2, (3) instruction to output a structured citation list |
| L3-2 | Single-query smoke test | — | Call the API with one val query. Verify it returns parseable citations. |
| L3-3 | Full val evaluation | `scripts/run_evaluation.py` | Run all 10 val queries through the LLM layer. Report Macro F1. |
| L3-4 | Implement in pipeline | `src/omnilex/llm/citation_predictor.py` | API call + response parser + citation normalization via `CitationNormalizer` |

**Assigned to:** ___________
**Deliverable:** `citation_predictor.py` + Macro F1 on val.csv with LLM layer active.
**Status: BLOCKED** — waiting for API documentation from university.

---

### Layer 4 — Adaptive k + Final Fusion
**Goal:** Replace fixed k with per-query citation count prediction, then fuse all layers for final submission.

| # | Task | File | Details |
|---|------|------|---------|
| L4-1 | Add adaptive k to submission | `src/omnilex/retrieval/submission.py` | Accept a per-query k dict instead of a global fixed k |
| L4-2 | LLM-predicted citation count | `src/omnilex/llm/citation_predictor.py` | Ask the LLM to also estimate how many citations apply; use that as k for each query |
| L4-3 | Full pipeline fusion | `src/omnilex/retrieval/fusion.py` | Wire L1 + L2 + L3 into existing `rrf_fuse()`. Run on val.csv. |
| L4-4 | Final submission | `submission.csv` | Generate, validate with `utils/validate_submission.py`, then submit to Kaggle |

**Assigned to:** ___________
**Deliverable:** Validated `submission.csv` + final Macro F1 on val.csv.
**Note:** Do not start L4-3 until Layer 3 is verified independently.

---

### Cross-cutting — Evaluation & Integration
**Goal:** Keep the team unblocked and track all results in one place.

| # | Task | Details |
|---|------|---------|
| X-1 | Maintain results log | After each verification step, record F1 numbers in `docs/results_log.md` |
| X-2 | Validate before upload | Run `python utils/validate_submission.py submission.csv` before every Kaggle submission |
| X-3 | Keep architecture doc updated | If any layer's design changes from experimental results, update `docs/ARCHITECTURE_V2.md` |

**Assigned to:** ___________

---

## 4. Summary

### What we keep from the old approach
| Component | Why we keep it |
|-----------|---------------|
| BGE-M3 pipeline | Still one of the best multilingual models; becomes Layer 2 |
| FAISS indices | Just need to re-run at k=25 |
| RRF fusion | Already works — just needs more layers as input |
| Evaluation scorer | Used to verify every step |
| `CitationNormalizer` | Canonical citation parser — do not modify |

### What changes
- k=10 → k=25 across the board
- Regex extractor completely rewritten with FR→DE mapping
- LLM reasoning added as the primary citation prediction layer
- Adaptive k replaces fixed k in the final submission

### Expected performance trajectory
| Milestone | Expected Macro F1 |
|-----------|-------------------|
| Current baseline (BGE-M3, k=10) | 0.0435 |
| After k=25 fix (Layer 2 only) | 0.06 – 0.08 |
| After LLM layer added (Layers 1–3) | 0.20 – 0.30 |
| Full pipeline + adaptive k (all 4 layers) | 0.25 – 0.35 |
| Current leaderboard leader | 0.359 |

The architecture is proven correct by the leaderboard. The gap closes with execution.
