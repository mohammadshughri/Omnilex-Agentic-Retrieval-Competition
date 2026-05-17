# Architecture Review Findings for Claude

**Date:** 2026-05-17  
**Subject:** Review of `2026-05-17-implementation-strategy-design.md` against current repo, research notes, public notebook analysis, and local data.  
**Reviewer stance:** Principal software architect / systems critic.

---

## Executive Verdict

The submitted BGE-M3-first architecture is a reasonable prototype, but it is not the best implementation strategy given the project evidence already collected.

The project should not simply copy a public notebook. It should use the public submissions as constraints and failure-mode evidence, then implement a cleaner, more robust, repo-native system:

- federated law/court retrieval;
- dense + sparse/BM25 candidate union;
- RRF instead of raw weighted score fusion;
- reranking earlier in the roadmap;
- dynamic citation-count selection earlier in the roadmap;
- compressed or chunked court retrieval;
- safe/indexed artifact storage;
- strong row-id/citation normalization tests.

The main change is this: **do not treat BGE-M3 hybrid retrieval as the target architecture. Treat it as a fast baseline. The target architecture should be model-pluggable, with Qwen3 and reranker support planned from the start.**

---

## Pragmatic Response To Claude's Pushback

Claude's response is directionally right about scope. My first review is the architecture north star; the May 21 deliverable needs a smaller execution slice.

The right decision is not "ignore the submitted spec" and not "build the whole ideal system now." The right decision is:

1. Keep the original BGE-M3-first timeline as the visible course roadmap.
2. Patch the first implementation with the cheap high-impact changes.
3. Defer storage/index sophistication until measurements prove it matters.
4. Keep interfaces model-pluggable so the Qwen3 upgrade is not a rewrite.

### Adopt Now, Before The First Presentation

| Change | Why | Estimated effort |
|---|---|---:|
| Replace weighted score fusion with RRF | Avoids invalid cross-scale score math | 30-60 min |
| Add citation-anchor extraction | Free precision for explicit `Art.` / `BGE` queries | 1-2 h |
| Compose indexed text as `citation + title + text` | Improves identifier recall | 5-15 min |
| Add minimal `EmbeddingModel` / `RerankerModel` protocols | Keeps BGE -> Qwen change cheap | 30-60 min |
| Test fixed `k = 2, 3, 5, 10` immediately | Prevents misleading first metrics | 10-30 min |
| Add row-id / citation alignment tests on a 100-row sample | Prevents expensive bad index jobs | 1 h |

This is the practical 3-5 hour delta that should be merged into the current spec.

### Defer Until After The First Presentation

| Idea | Decision | Reason |
|---|---|---|
| SQLite/DuckDB metadata | Defer | JSONL plus an in-memory list is fine for the first demo if lookup is not hot. |
| CSR/Arrow sparse storage | Defer | Use FlagEmbedding's native sparse output first; optimize after profiling. |
| Compressed FAISS / IVFPQ | Defer | Start with HNSWFlat if cluster RAM allows; document measured memory. |
| Chunked MaxSim courts | Defer | Valuable, but too many moving parts for the first milestone. |
| HyDE | Defer | Adds generation variability before retrieval baseline is stable. |
| Court query expansion | Optional | Useful, but not necessary for the first BGE-M3 laws/courts baseline. |

### Technical Pushback On Claude's Deferrals

Some of Claude's deferrals are correct for May 21 but not correct as final architecture decisions.

- **Pickle is acceptable only for local trusted prototypes.** It should not become the shared artifact format for uploaded indices or downloaded datasets.
- **JSONL metadata is fine only if lookup is indexed or loaded once.** Scanning JSONL per result would be a hot path bug.
- **HNSWFlat for courts is acceptable only after memory measurement.** The spec should not claim it "fits comfortably" without a measured RAM/disk budget.
- **Tuning `k` on the 10-query val set is useful for presentation, not strong evidence.** Label it as a quick calibration check, not a robust final threshold.

So the compromise is: implement Claude's small-change table now, keep my broader warnings as "do not let the prototype harden into final design."

---

## Are We Just Copying Public Research?

No, not if we implement it correctly.

The public notebooks establish useful facts:

- Qwen3-Embedding + Qwen3-Reranker currently looks stronger than BGE-M3.
- Federated search beats mixing laws and courts into one shared pool.
- RRF is more robust than raw score weighting across retrieval systems.
- Fixed top-k is weak because citation counts vary heavily.
- Court dense retrieval is mostly not done publicly because it is expensive.

Copying would mean reproducing a Kaggle notebook pipeline with minimal engineering. Improving means building the missing infrastructure that the notebooks avoid:

- a reusable retrieval package under `src/omnilex/retrieval/`;
- artifact formats that load safely and scale;
- deterministic index metadata alignment;
- evaluation harnesses that prevent val overfitting;
- model adapters for both CrossEncoder rerankers and Qwen causal-LM rerankers;
- court dense/chunk retrieval, which public notebooks mostly skipped.

The opportunity is not "use Qwen3 because public notebooks did." The opportunity is **build the first clean, testable, full-corpus, federated dense court retrieval implementation for this project**.

---

## What I Would Change Exactly

### 1. Change The Roadmap Ordering

Current spec:

1. BGE-M3 hybrid retrieval.
2. Add reranker.
3. Add HyDE.
4. Upgrade to Qwen3.
5. Add calibration/routing.

Recommended:

1. Build repo-native federated candidate generation over current CSV data.
2. Add laws dense retrieval + BM25 + RRF.
3. Add reranker and dynamic K immediately.
4. Add compressed/chunked court dense retrieval.
5. Compare BGE-M3 vs Qwen3 behind the same interfaces.
6. Add HyDE only after the retrieval/reranking baseline is measurable.

Reason: reranking and dynamic K are not "polish"; they are core to Macro F1. Delaying them produces misleading retrieval results.

---

### 2. Replace Raw Weighted Fusion With RRF

Current spec proposes:

```text
Laws score:  0.4 * dense + 0.2 * sparse + 0.4 * colbert
Courts score: 0.6 * dense + 0.4 * sparse
```

Problem: dense, sparse, ColBERT, BM25, law, and court scores are not naturally comparable. This is especially fragile across two corpora with very different document lengths.

Recommendation:

- retrieve candidates independently from each channel;
- rank each channel;
- fuse by Reciprocal Rank Fusion;
- rerank the fused top candidates with a cross-encoder/LLM reranker.

Use raw score weighting only after calibration proves it beats RRF.

---

### 3. Make The Retriever Model-Pluggable From Day One

Do not hard-code BGE-M3 into `dense_index.py` / `dense_retriever.py`.

Use adapters:

```text
EmbeddingModel
  - encode_documents(texts) -> vectors
  - encode_queries(queries) -> vectors
  - document_prefix / query_prefix support

RerankerModel
  - score_pairs(query, documents) -> probabilities
```

Required first adapters:

- `BgeM3Embedder`
- `Qwen3Embedder`
- `BgeCrossEncoderReranker`
- `Qwen3CausalLmReranker`

Important: Qwen3-Reranker is not a normal CrossEncoder. It must be loaded as a causal LM and scored via yes/no logits. The public analysis flags this as a common silent failure mode.

---

### 4. Move Dynamic K Into The First Measurable Version

Current spec delays citation-count calibration until Week 7.

That is too late. Macro F1 depends heavily on how many citations are predicted.

Recommended v1 behavior:

- retrieve and rerank top 50-100 candidates;
- convert reranker logits to probabilities;
- choose citations by calibrated threshold;
- keep fallback bounds, e.g. min 1, max 50;
- compare fixed `k = 5, 10, 20` against thresholding.

Do not threshold raw logits. Threshold probabilities.

---

### 5. Add Citation-Anchor Extraction As A First-Class Candidate Channel

Many English val/test queries contain literal anchors like:

```text
Art. 221 Abs. 1 lit. b StPO
Art. 83 SVG
Art. 59 Abs. 1 SVG
```

The current architecture expects retrieval to rediscover these. That wastes a very strong signal.

Recommended:

- extract explicit `Art.` and `BGE` style citations from the query;
- normalize them through `CitationNormalizer`;
- add them as high-priority candidates;
- strip `lit.`, `Ziff.`, and similar suffixes when matching, because gold labels stop at paragraph level.

This is not enough alone, but it is a cheap precision anchor.

---

### 6. Change Text Composition For Dense Indexing

Current spec indexes passages but does not explicitly require citation/title composition.

Recommendation:

- laws: `citation + " " + title + " " + text`
- courts: `citation + " " + text`

Reason: public notebooks found that including the citation string improves exact identifier recall, and the local data has useful `citation` and `title` columns for laws.

---

### 7. Do Not Store Sparse Features As Pickle Dicts

Current spec proposes:

```text
laws_sparse.pkl
courts_sparse.pkl
```

Problem:

- pickle is unsafe for untrusted artifacts;
- Python dict overhead is large at 2.4M documents;
- loading the whole structure is slow and memory-heavy.

Recommendation:

- dense vectors: `.npy` / `.npz` / FAISS index;
- metadata: SQLite, DuckDB, or Parquet keyed by row id;
- sparse vectors: CSR matrix, Arrow/Parquet, or a proper inverted index;
- avoid pickle for any artifact that might be shared, downloaded, or uploaded to Kaggle.

The existing `BM25Index` also uses pickle and full in-memory rebuild. That is acceptable for a starter baseline, not for the final architecture.

---

### 8. Rework Court Retrieval

The spec says dense court retrieval is an opportunity, which is correct. But the implementation plan is too optimistic.

Court corpus facts:

- about 2.4M rows;
- around 2.4 GB CSV locally;
- longer passages than laws;
- trilingual German/French/Italian.

Recommended implementation:

1. Start with BM25 court retrieval plus query expansion from top law citations.
2. Add compressed dense court index, not HNSWFlat unless memory measurements prove it fits.
3. Consider chunking / MaxSim for court passages.
4. Map chunk ids back to citation ids.
5. Deduplicate by citation before reranking.

This is a real improvement over public notebooks because dense court retrieval was mostly skipped there.

---

### 9. Add Evaluation Discipline

The local research notes say train queries are German, while val/test are English. This makes naive train tuning misleading.

Recommended:

- keep `val.csv` as a final holdout, not a tuning playground;
- create train-derived diagnostics, but label them as German-query diagnostics;
- report separate metrics for:
  - explicit citation-anchor queries;
  - conceptual queries;
  - laws vs courts;
  - low vs high citation-count queries.

Do not claim leaderboard expectations from a 10-query val set without uncertainty.

---

### 10. Add Acceptance Tests Before Large Indexing

Before embedding 2.4M court rows, add tests for:

- CSV row id to FAISS id alignment;
- metadata lookup by FAISS id;
- citation normalization of retrieved raw citations;
- duplicate citation collapse;
- submission row ordering;
- score fusion determinism;
- dynamic K bounds;
- reranker adapter behavior on a tiny fixture.

This is where our implementation can be materially better than notebook code.

---

## Proposed Revised Architecture

```text
English query
  |
  +--> citation anchor extractor
  |
  +--> law BM25 / sparse retrieval
  |
  +--> law dense retrieval
  |
  +--> court BM25 retrieval
  |
  +--> court dense/chunk retrieval
  |
  v
per-channel candidate lists
  |
  v
RRF fusion per corpus
  |
  v
federated law/court merge with deduplication
  |
  v
reranker top 50-100
  |
  v
dynamic K / probability threshold
  |
  v
CitationNormalizer
  |
  v
submission.csv + evaluation report
```

Model choices should be configuration, not architecture:

```text
Fast baseline: BGE-M3 + BGE reranker
Target comparison: Qwen3-Embedding-0.6B/8B + Qwen3 reranker
Optional: HyDE query variant
Optional: translated BM25 branches
```

---

## Main Gaps In The Submitted Spec

### Gap 1: It Treats BGE-M3 As The Main Bet

BGE-M3 is a good starting point, but the research evidence points toward Qwen3 as a stronger target. The implementation should avoid BGE-specific coupling.

### Gap 2: Sparse Retrieval Is Underspecified

The spec says BGE-M3 sparse weights are used, but does not define how sparse candidates are retrieved at corpus scale. Sparse scoring only over dense top-100 is not equivalent to sparse retrieval.

### Gap 3: Fusion Is Brittle

Weighted score fusion assumes score comparability that likely does not hold.

### Gap 4: Reranking Is Too Late

Reranking is a primary quality mechanism, not a later enhancement.

### Gap 5: Dynamic K Is Too Late

Citation-count prediction is central to Macro F1 and should be tested from the first end-to-end version.

### Gap 6: Court Index Memory Is Underestimated

HNSWFlat for 2.4M x 1024 vectors is likely larger than the stated estimate. Court dense retrieval needs compressed indexing or chunking strategy.

### Gap 7: Artifact Format Choices Are Unsafe / Heavy

Pickle and raw JSONL metadata are acceptable for quick experiments, but weak for reproducible shared artifacts.

### Gap 8: Evaluation Leakage Risk Is Not Controlled

The 10-query English val set is too small to tune repeatedly. The architecture needs a written evaluation protocol.

### Gap 9: It Does Not Exploit Existing Repo Strengths Enough

The existing repo already has:

- `CitationNormalizer`;
- Macro F1 scoring;
- submission validation;
- BM25 retrieval wrappers;
- ReAct baseline tools.

The new architecture should wrap and extend these rather than build a parallel experiment stack.

---

## Minimal Implementation Plan I Would Ask Claude To Execute

### Phase 0: Interfaces And Fixtures

Create:

- `retrieval/candidates.py`
- `retrieval/fusion.py`
- `retrieval/reranking.py`
- `retrieval/submission.py`

Add tiny fixture data and tests.

Success criteria:

- candidate ids are stable;
- deduplication works;
- RRF is deterministic;
- generated submission validates.

### Phase 1: Law-Only Hybrid Baseline

Implement:

- citation-anchor extraction;
- law BM25 channel;
- law dense channel;
- RRF;
- reranker;
- dynamic K.

Success criteria:

- beats BM25-only on val;
- produces per-query diagnostics.

### Phase 2: Court BM25 + Query Expansion

Implement:

- top law citations appended to court query;
- court BM25 channel;
- law/court federated merge.

Success criteria:

- improves recall on BGE/court-heavy val cases;
- no law results buried by court passages.

### Phase 3: Compressed Dense Court Index

Implement:

- court dense index with compressed FAISS or chunked MaxSim;
- indexed metadata store;
- citation-level deduplication.

Success criteria:

- measurable recall@100 gain for court citations;
- memory budget documented.

### Phase 4: Qwen3 Comparison

Implement:

- Qwen3 embedding adapter;
- Qwen3 causal-LM reranker adapter;
- same evaluation harness as BGE.

Success criteria:

- compare BGE vs Qwen under identical retrieval/fusion/reranking logic.

---

## Final Recommendation

Do not implement the submitted spec literally.

Keep its useful ideas:

- progressive delivery;
- separate law/court indices;
- BGE-M3 as a fast baseline;
- full-corpus court retrieval as a differentiator.

Change the architecture around it:

- candidate channels first;
- RRF fusion;
- reranking and dynamic K early;
- Qwen3-ready abstractions;
- compressed/chunked court retrieval;
- safe artifact storage;
- strict evaluation protocol.

That makes the work meaningfully better than copying public notebooks while still respecting the strongest empirical signals from the research.
