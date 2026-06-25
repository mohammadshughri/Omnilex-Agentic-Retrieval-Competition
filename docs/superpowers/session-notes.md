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

---

## Session 6 — Measurement Fix + API Setup + Smoke Tests (Person A & B)

**Date:** 2026-06-06
**Reference doc:** `2026-06-06-diagnostics-and-task-split.md`

---

### Key Finding: Measurement Bug Fixed (Person A)

`CitationNormalizer` was silently dropping 33 of 251 val.csv gold citations (13.1%) because docket-style Federal Tribunal case references (`1B_210/2023 E. 4.1`, `7B_496/2025 E. 3.2`, etc.) were not recognised. All prior evaluation metrics were computed against 218 citations — the denominator was wrong, making measured recall artificially high.

**Fix:** Added `DOCKET_PATTERN` regex and `_parse_docket()` method to `CitationNormalizer`. Also updated `anchor_extractor.py` with a matching docket pattern.

| | Before | After |
|---|---|---|
| val.csv gold citations parsed | 218 | **251** |
| Unparseable docket citations | 33 | **0** |
| Prior BGE-M3 F1 (0.0435) | Based on 218 gold | **Needs recomputation** |

---

### Corrected Baselines (Person A — `results/baselines_corrected.json`)

| Metric | Value |
|---|---|
| val total citations | 251 (149 Art + 69 BGE + 33 docket) |
| val mean citations/query | 25.1 |
| Anchor-only Macro F1 | **0.0237** |
| Oracle F1 @ k=10 | **0.6444** |
| Oracle F1 @ k=25 | **0.7788** |
| Oracle F1 @ k=30 | 0.7612 (drops — overshoot) |

---

### GWDG / KISSKI API Setup (Person B — `scripts/test_gwdg_api.py`)

**Endpoint:** `https://chat-ai.academiccloud.de/v1` (OpenAI-compatible). Credentials in `.env` as `KISSKI_API_KEY`.

**Available chat models:**

| Model | Active Params | Recommended for |
|---|---|---|
| `meta-llama-3.1-8b-instruct` | 8B | Fast preprocessing |
| `teuken-7b-instruct-research` | 7B | Native German — law text |
| `qwen3-30b-a3b-instruct-2507` | 3B active | Current default |
| `deepseek-r1-distill-llama-70b` | 70B | Chain-of-thought reasoning |
| `qwen3.5-122b-a10b` | 10B active | **Recommended next** — strong + fast (MoE) |
| `qwen3.5-397b-a17b` | 17B active | Strongest if 122b insufficient |
| `mistral-large-3-675b-instruct-2512` | 675B | Most powerful, slowest |

**Embedding decision:** `multilingual-e5-large` is NOT deployed on the API. Use it **locally** via sentence-transformers (560M, CPU-feasible, 1024-dim, cross-lingual EN→DE). The API's `e5-mistral-7b-instruct` (4096-dim) is EN-primary and weaker for German laws.

---

### LLM Smoke Test — val_001 (Person B — `scripts/llm_smoke_test.py`)

Model: `qwen3-30b-a3b-instruct-2507`

| Metric | Before fix | After fix |
|---|---|---|
| Gold citations (parseable) | 30 / 42 raw | **42 / 42** |
| LLM predicted (parseable) | 8 (4 unparseable) | **12 (0 unparseable)** |
| Hits | 3 | **4** |

**Multi-model results:**

| Model | Hits | Precision | Recall | Gate |
|---|---|---|---|---|
| `qwen3-30b-a3b-instruct-2507` | 4 | 0.333 | 0.095 | Borderline |
| `qwen3.5-122b-a10b` | 1 | 0.111 | 0.024 | Fail |

**Gate verdict: DEPRIORITIZE LLM reasoning. Focus on retrieval improvements.**

The 30B model outperformed the 122B — the larger model hallucinated 8 citations vs the smaller model's 4 hits. Neither crossed the ≥5 threshold. Note: `qwen3.5-122b-a10b` uses thinking mode (`msg.reasoning`, not `msg.content`) and requires `max_tokens=8000`. Fixed in `scripts/llm_smoke_test.py`.

---

### Dense Retrieval Pipeline — Validated (Person B — `scripts/eval_dense_baseline.py`)

Supports `--embedder local` (multilingual-e5-large) or `--embedder api` (e5-mistral), `--sample N`, `--no-cache`. Pipeline validated at 10/100/1000 rows. F1=0 on small samples is expected — gold citations are spread across 175K laws. Full run: ~3h local (CPU) or ~20 min on Kaggle GPU with BGE-M3.

---

### Decisions & What NOT to Do Yet

- ✅ Normalizer fix merged — all future evaluations use 251-citation gold set
- ✅ `multilingual-e5-large` chosen as embedding model (cross-lingual, local, no rate limits)
- ⏳ LLM pipeline decision gated on smoke test re-run with `qwen3.5-122b-a10b`
- ❌ Do NOT use prior 0.0435 BGE-M3 number — measured against wrong 218-citation gold set
- ❌ Do NOT build XGBoost, HyDE, or query translation yet — need correct metrics first
- ❌ Courts corpus embedding deferred — measure laws-only first

---

## Session 7 — BGE-M3 Laws Eval + Hybrid RRF (Person A)

**Date:** 2026-06-09
**Notebook:** `dense-retrieval-bge-m3-laws-eval.ipynb` (Kaggle T4 GPU)

---

### BGE-M3 Dense Retrieval — Laws Only (Full Run)

Embedded all 175,933 laws with `BAAI/bge-m3` on Kaggle T4 GPU (~18 min). Evaluated on val.csv with corrected 251-citation gold set.

| k | Macro F1 | Mean Precision | Mean Recall |
|---|---|---|---|
| 10 | 0.0120 | 0.0300 | 0.0075 |
| 15 | 0.0183 | 0.0325 | 0.0128 |
| 20 | 0.0197 | 0.0287 | 0.0152 |
| 25 | **0.0216** | 0.0274 | 0.0180 |

**Key finding:** Dense-only (laws) is BELOW anchor-only baseline (0.0237). Root cause: 102/251 gold citations are BGE/docket court decisions — absent from `laws_de.csv`. 8/10 queries retrieved zero true positives.

---

### Anchor Extraction Analysis

Ran `extract_citation_anchors()` on all 10 val queries. Results:

| Query | Anchors found |
|---|---|
| val_001 | 1 (`Art. 221 Abs. 1 StPO`) |
| val_006 | 3 (`Art. 364 OR`, `Art. 248 OR`, `Art. 41 OR`) |
| val_007 | 2 (`Art. 934 ZGB`, `Art. 936 ZGB`) |
| val_002–005, 008–010 | 0 |

**Finding:** Most queries describe legal scenarios in plain English without citing specific articles or case numbers. Anchor extraction has reached its ceiling at 3/10 queries. BGE/docket citations in the gold set come from court decisions that ruled on similar cases — they are not mentioned in the query text itself.

---

### Hybrid Retrieval: Anchor + Dense via RRF

Combined anchor channel and dense (laws) channel using Reciprocal Rank Fusion (RRF_K=60, DENSE_FETCH=100).

| k | Hybrid F1 | Dense F1 | Delta |
|---|---|---|---|
| 10 | **0.0332** | 0.0120 | +0.0212 |
| 15 | 0.0289 | 0.0183 | +0.0106 |
| 20 | 0.0325 | 0.0197 | +0.0128 |
| 25 | 0.0294 | 0.0216 | +0.0078 |

**Best: k=10, Hybrid F1 = 0.0332 (+40% over anchor-only baseline of 0.0237)**

Per-query at k=10:

| Query | Gold | Anchors | TP | F1 |
|---|---|---|---|---|
| val_001 | 42 | 1 | 2 | 0.077 |
| val_002 | 36 | 0 | 1 | 0.043 |
| val_006 | 18 | 3 | 2 | 0.143 |
| val_007 | 19 | 2 | 1 | 0.069 |
| val_003–005, 008–010 | — | 0 | 0 | 0.000 |

---

### Updated Scoreboard

| Method | Macro F1 | Notes |
|---|---|---|
| Anchor-only | 0.0237 | Regex extraction only |
| BGE-M3 dense, laws k=25 | 0.0216 | Below anchor baseline |
| **Hybrid anchor+dense k=10** | **0.0332** | Current best |
| Oracle k=25 | 0.7788 | Ceiling |
| Leaderboard top | 0.3590 | Target |

---

### Decisions & Next Steps

- ✅ Hybrid RRF pipeline validated — current best is 0.0332
- ✅ Anchor extraction ceiling confirmed — no further gains without courts corpus
- ✅ Courts corpus embedded — 2.4M rows, 3.2h on T4
- ❌ Do NOT further tune anchor regex — queries don't contain explicit BGE/docket citations
- ❌ Do NOT deprioritize courts corpus — 102/251 gold citations are unreachable without it

---

## Session 8 — LLM Metadata Filtering + Citation Bridge (Person A)

**Date:** 2026-06-09
**Notebook:** `dense-retrieval-bge-m3-laws-eval.ipynb` (Kaggle T4 GPU)
**Reference:** Public solution analysis — [PrudhvirajuChekuri/swiss-legal-information-retrieval](https://github.com/PrudhvirajuChekuri/swiss-legal-information-retrieval)

---

### Key Discovery: Top Solution Architecture

Reverse-engineered the approach behind the leaderboard top score (0.3590). The critical insight is **filter first, search less**:

1. **LLM predicts Swiss law codes** (ZGB, OR, StGB, StPO...) from the English query before any vector search
2. **Filter corpus** to only matching articles — reduces laws search space by ~96%
3. **Citation bridge** — precomputed mapping `law article → court decisions that cite it`, bypasses dense search on 2.4M courts entirely
4. **HyDE** — generate hypothetical German documents from English queries for better cross-lingual embeddings
5. **Three separate pipelines** — laws, BGE, docket — each tuned independently

---

### Courts Dense Retrieval — Post-Mortem

Embedded full courts corpus (2.4M rows, 3.2h on T4). Result: **0 hits** in top-100 for val_001 gold BGE/docket citations.

Root cause confirmed: dense retrieval finds *topically similar* court cases but not the *specific cases cited* in gold. With 2.4M documents, gold citations rank at 10,000+ — unreachable at any practical k.

**Decision: do NOT use courts dense retrieval. Use citation bridge instead.**

---

### LLM Metadata Filtering — Results

Used KISSKI `qwen3-30b-a3b-instruct-2507` to predict relevant Swiss law codes per query, then filtered laws corpus before dense search.

**Evaluation results (anchor + filtered dense via RRF):**

| k | Filtered F1 | 2-Way F1 | Delta |
|---|---|---|---|
| 10 | 0.0485 | 0.0332 | +0.0153 |
| **15** | **0.0600** | 0.0289 | **+0.0311** |
| 20 | 0.0529 | 0.0325 | +0.0204 |
| 25 | 0.0502 | 0.0294 | +0.0208 |

**Best: k=15, F1 = 0.0600 (+81% over previous best of 0.0332)**

Per-query at k=15 — val_002 and val_009 jumped from 0 TP to 3 TP and 2 TP respectively after filtering.

| Query | TP | F1 |
|---|---|---|
| val_001 | 3 | 0.105 |
| val_002 | 3 | 0.118 |
| val_006 | 2 | 0.121 |
| val_007 | 2 | 0.118 |
| val_009 | 2 | 0.138 |
| val_003–005, 008, 010 | 0 | 0.000 |

---

### Updated Scoreboard

| Method | Best F1 | Notes |
|---|---|---|
| Anchor-only | 0.0237 | Regex extraction only |
| BGE-M3 dense, laws k=25 | 0.0216 | Below anchor baseline |
| Hybrid anchor+dense k=10 | 0.0332 | 2-way RRF |
| **Filtered anchor+dense k=15** | **0.0600** | LLM code prediction + RRF — current best |
| Oracle k=25 | 0.7788 | Ceiling |
| Leaderboard top | 0.3590 | Target |

---

### Citation Bridge — In Progress

Building precomputed mapping: `normalized Art. citation → [court decisions that cite it]`.

- Source: `court_considerations.csv` (2.4M rows)
- Method: regex extract Art. patterns from each court decision text
- Estimated build time: ~30 min one-time, cached to `citation_bridge.json`

When complete: retrieved law articles will automatically expand to linked court decisions via the bridge, covering the 102/251 BGE/docket gold citations currently unreachable by dense retrieval.

---

### Decisions & Next Steps

- ✅ LLM metadata filtering validated — +81% F1 improvement, new best = 0.0600
- ✅ Courts dense retrieval abandoned — 0 hits confirmed dead end
- ⏳ **Citation bridge build in progress** — expected to push F1 toward 0.10+
- ⏳ **HyDE** — next after bridge: generate German hypothetical docs from English queries
- ❌ Do NOT tune RRF weights yet — citation bridge will change the signal mix
- ❌ Do NOT use 3-way RRF with courts dense index — confirmed hurts performance

---

## Session 9 — LLM Article Prediction Paradigm Shift (Person A)

**Date:** 2026-06-09
**Notebook:** `dense-retrieval-bge-m3-laws-eval.ipynb` (Kaggle T4 GPU)

---

### Citation Bridge — Abandoned

Built and tested the precomputed mapping (`law article → court decisions that cite it`). Result: **F1 = 0.0445 vs 0.0600 baseline — hurts performance**.

Root cause: the bridge relies on accurate law article retrieval to produce meaningful court links. At current retrieval quality, the law articles feeding the bridge are mostly wrong, so the bridge expands noise rather than finding gold BGE citations.

**Decision: abandon citation bridge. Focus on improving law article precision first.**

---

### HyDE — Tested, Minor Contribution as 3rd RRF Channel

Generated hypothetical German legal documents from English queries using `qwen3-30b-a3b-instruct-2507`, then embedded the German text with BGE-M3.

- HyDE alone as query replacement: **worse** than filtered dense (0.0517 vs 0.0600 at k=15)
- HyDE as a **3rd RRF channel** (anchor + raw query + HyDE) at k=30: **F1 = 0.0678** (+13% over filtered 2-way)

**Why HyDE fails as the primary signal:** val_004 diagnostic showed HyDE generated the correct article number but still got 0 TP. Embedding a full German paragraph does not match short law corpus entry style — format mismatch, not a language problem.

---

### Diagnostic: Why Dense Retrieval Fails on Most Queries

Detailed diagnosis on val_004 (inheritance/will case, 10 gold citations):

1. **All 9 law article gold citations ARE in the laws corpus** — dense retrieval simply cannot find them.
2. **BGE citations are structurally unreachable** from `laws_de.csv` — `BGE 131 III 601 E. 3.1` is a court decision. Every query with BGE gold citations has a hard recall ceiling from laws-only retrieval.
3. **The gap is legal reasoning, not language** — finding that val_004 requires `Art. 469 ZGB` (undue influence on a will) from a plain-English case description requires legal expertise. Dense embeddings cannot bridge this.

---

### LLM Article Prediction — Major Paradigm Shift

**Core idea:** Instead of embedding the query and searching, ask the LLM directly: *"Given this case, which specific Swiss law articles would a court cite?"*

**Iteration history:**

| Version | Approach | val_004 TP | Macro F1 |
|---|---|---|---|
| v1 | Naive "list 20 articles" | 0 (sequential list 490–509) | — |
| v2 | Chain-of-thought: identify issues first, then articles | 4–5 TP | **0.0926** |
| v3 | v2 + French→German abbreviation fix (LAI→IVG, LACI→AVIG) | varies | 0.0882 combined |
| Ensemble | Union of 3 runs | 1 TP, 25 articles | 0.0781 |

**Best result: LLM v2 + anchor union = Macro F1 = 0.0926 (+37% over filtered dense 0.0600)**

Per-query breakdown at best run:

| Query | Gold | LLM_TP | F1 |
|---|---|---|---|
| val_001 | 42 | 2 | 0.111 |
| val_002 | 36 | 0 | 0.000 |
| val_003 | 47 | 3 | 0.102 |
| val_004 | 10 | 5 | 0.323 |
| val_005 | 11 | 1 | 0.062 |
| val_006 | 18 | 2 | 0.174 |
| val_007 | 19 | 0 | 0.000 |
| val_008 | 29 | 1 | 0.048 |
| val_009 | 14 | 1 | 0.053 |
| val_010 | 25 | 1 | 0.054 |

**Important bugs found and fixed:**

- **French abbreviations:** KISSKI API returns LAI (French) instead of IVG (German), LACI instead of AVIG. Fix: `FR_TO_DE` dict applied to LLM output before regex extraction. val_002 went from 0 predictions to valid IVG/AVIG articles.
- **Parent citation mismatch:** Gold has `Art. 467 ZGB` (no Abs.), LLM predicts `Art. 467 Abs. 1 ZGB`. Fix: strip Abs. from predicted articles and add parent form to prediction set.

**Key negative findings:**

- **Ensemble (3-run union) hurts:** val_007 ballooned to 72 articles with 0 TP → F1=0.000. More predictions = more FP = lower precision = lower F1. Do NOT ensemble.
- **Dense retrieval on top of LLM hurts:** Combined LLM+anchor+dense = 0.0882 (< 0.0926 LLM+anchor alone) because dense adds noise without enough new TPs. The LLM is also non-deterministic at temperature=0.1 — val_004 ranged from 5 TP to 0 TP between runs.

---

### Updated Scoreboard

| Method | Best F1 | Notes |
|---|---|---|
| Anchor-only | 0.0237 | Regex extraction only |
| BGE-M3 dense laws k=25 | 0.0216 | Below anchor baseline |
| Hybrid anchor+dense k=10 | 0.0332 | 2-way RRF |
| Filtered anchor+dense k=15 | 0.0600 | LLM code prediction + RRF |
| 3-way RRF (anchor+raw+HyDE) k=30 | 0.0678 | HyDE as 3rd channel |
| **LLM article prediction + anchor** | **0.0926** | **Current best** |
| Oracle k=25 | 0.7788 | Ceiling |
| Leaderboard top | 0.3590 | Target |

---

### Open Problems

1. **LLM stochasticity** — val_004 varies 0–5 TP between runs at temperature=0.1. Fix in progress: `predict_articles_v4` with `temperature=0.0` and max 10 articles, "one citation per line" output format.
2. **BGE/docket citations unreachable from laws corpus** — 102/251 gold citations are court decisions. Laws corpus alone caps recall at ~59%.
3. **val_007 consistently 0 TP** — LLM predicts wrong articles across all runs. Needs diagnosis.
4. **Low recall on high-citation queries** — val_001 (42 gold), val_002 (36), val_003 (47) have too many gold citations for LLM to exhaust.

---

### Decisions & Next Steps

- ✅ LLM article prediction validated — paradigm shift confirmed, best F1 = 0.0926
- ✅ Citation bridge abandoned — hurts performance at current retrieval quality
- ✅ Dense retrieval confirmed as noise source — LLM+anchor alone beats LLM+anchor+dense
- ✅ Ensemble approach confirmed harmful — over-prediction hurts F1
- ⏳ **`predict_articles_v4` in progress** — temperature=0, max 10 articles, stable output
- ⏳ **Diagnose val_007** — 0 TP despite 11+ predictions across all runs
- ❌ Do NOT use ensemble (union of N runs) — inflates FP, hurts F1
- ❌ Do NOT add dense retrieval on top of LLM predictions — adds noise, no net gain
- ❌ Do NOT use citation bridge yet — needs accurate law retrieval first

---

## Session 10 — Push A: v4 determinism + calibration sweep (2026-06-11)

**Date:** 2026-06-11
**Plan:** `docs/superpowers/plans/2026-06-10-llm-v4-and-bge-pipeline.md` Tasks 2–4
**Kernel version pushed:** v8 (after 7 debug pushes to fix K_VALUES guard, KISSKI secret, import re)
**Push A result:** GATE FAILED — v4 regressed vs v2 baseline

---

### What was built

Added three task cells to `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb`:
- `predict_articles_v4`: temperature=0.0, ISSUES+CITATIONS chain-of-thought, no law-code filtering
- `expand_parents`: adds Abs.-stripped parent for every Abs. prediction (Session 9 fix)
- Truncation sweep at N=[5,8,10,12,15,20,25]
- val_007 diagnostic (book overlap + near-miss analysis)

Also fixed 7 unguarded cells that caused K_VALUES NameError (cells 35, 37, 38, 39, 41, 42 + bridge builder cell 30).

---

### Push A Results

#### Determinism (Task 2 gate)
- Run lengths: [25, 25, 25] — model returns 25 articles every time
- **Identical across runs: False** — model is NOT deterministic at temperature=0.0
- All 3 runs on val_004: 0 TP each (val_004 gold has 10 citations, all missed)
- Plan verdict: "record variance, proceed anyway — KISSKI backend nondeterminism out of our control"

#### Full val evaluation (Task 2)
| Query | Gold | Pred | TP | F1 |
|---|---|---|---|---|
| val_001 | 42 | 27 | 3 | 0.087 |
| val_002 | 36 | 49 | 3 | 0.071 |
| val_003 | 47 | 50 | 2 | 0.041 |
| val_004 | 10 | 26 | 0 | 0.000 |
| val_005 | 11 | 50 | 0 | 0.000 |
| val_006 | 18 | 27 | 2 | 0.089 |
| val_007 | 19 | 26 | 0 | 0.000 |
| val_008 | 29 | 27 | 1 | 0.036 |
| val_009 | 14 | 25 | 0 | 0.000 |
| val_010 | 25 | 25 | 0 | 0.000 |

**v4 (n=25) + anchor: Macro F1 = 0.0323** (vs v2 baseline 0.0926)

#### Truncation sweep (Task 3 gate)
| N | Macro F1 | Mean pred |
|---|---|---|
| 5 | 0.0453 | 7.4 |
| 8 | 0.0412 | 11.3 |
| 10 | 0.0389 | 13.9 |
| 12 | 0.0369 | 16.5 |
| 15 | 0.0404 | 20.4 |
| 20 | 0.0359 | 26.9 |
| 25 | 0.0323 | 33.2 |

**Best N=5, Macro F1=0.0453 — GATE FAILED (< 0.0926). Entire curve is below baseline.**

---

### val_007 Diagnostic (Task 4)

**Query summary:** Heirship claim to a vintage pocket chronometer. Ms. Barnes died 2010, allegedly donated watch to Mr. Collins in 2006 (only photocopy of deed). Mr. Collins sold it 2010 via Ms. Ortega to Eastbridge LLC, then to Alpine Trading AG.

**Gold citations (19):** Art. 933, 934, 940 ZGB (possession/good-faith acquisition), Art. 8, 16 ZGB (burden of proof), Art. 197, 641 ZGB (ownership), Art. 245 OR (donations), Art. 292 StGB (document offence), Art. 100 IPRG + Art. 98 IPRG (private international law), Art. 3, 15 OR, Art. 16 ZGB, plus 4 BGE court decisions.

**v4 predictions (24):** Art. 221–240 ZGB (donation/gift chapter), Art. 934, 936 ZGB (from anchors), Art. 100–101 ZPO. Book overlap with gold: only ZGB. Zero IPRG, zero BGE.

**Hypothesis for 0 TP:** The LLM correctly identified the gift/donation legal dimension (Art. 221–240 ZGB) but the gold focuses on *property recovery* (Art. 933–940 ZGB) and *private international law* (IPRG). The model reframed "was the donation valid?" instead of "can the heir recover the chattel in a cross-border context?" without law-code filtering to constrain the search space to the relevant books. The 2 anchors (Art. 934, 936 ZGB) provided the only correct ZGB hits, but with expand_parents they didn't match the gold's Abs.-qualified forms exactly.

---

### Root Cause of v4 Regression

v2/v3 pipeline = **two-step**: (1) `predict_law_codes` → 3–5 relevant law books, (2) predict articles *within those books only*. The code constraint stops the model from wandering into adjacent areas.

v4 = **one-step**: predict articles from all Swiss law directly. Without the law-book constraint, predictions drift: for val_007 the model explored ZGB donation law (Art. 221–240) and ZPO; for val_004 it returned 0 TP entirely (gold is inheritance law, Art. 469–475 ZGB, but model didn't hit them despite being in that neighbourhood for another query).

The confidence-ordering instruction ("most confident to least") did not help — the model's confident predictions are simply wrong books/articles.

---

### Fix Required Before Continuing

v4 must accept `predicted_codes` (from the already-running `predict_law_codes` cell) and restrict article candidates to those books, as v3 did. This preserves:
- temperature=0 determinism (still fails due to backend, but effort is correct)
- confidence ordering (quality improvement goal)
- law-code filtering (the actual source of v2/v3's score)

Alternatively, incorporate law-code prediction directly into the v4 system prompt (single LLM call, two-step instruction: "first identify relevant law codes, then list articles from those codes only").

---

### Status
- Task 1: COMPLETE (kernel snapshotted, GPU disabled, dense cells guarded)
- Task 2: GATE FAILED — v4 F1 = 0.0323, non-deterministic; root cause identified (missing code filter)
- Task 3: GATE FAILED — best sweep F1 = 0.0453 < 0.0926
- Task 4: COMPLETE — val_007 hypothesis documented above
- Tasks 5, 6, 7: Blocked pending Task 2/3 fix

---

## Session 11 — KISSKI Model Comparison + Anchor Fix (Person A)

**Date:** 2026-06-13 / 2026-06-14
**Notebook:** `omnilex-llm-retrieval-v2.ipynb` (Kaggle T4 GPU)

---

### Anchor Fix: Strict Abs. Filter

Discovered that val_007 anchors (`Art. 934 ZGB`, `Art. 936 ZGB`) are NOT in the gold set — gold only contains `Art. 934 Abs. 1 ZGB` and `Art. 934 Abs. 2 ZGB`. Plain parent citations without `Abs.` are false positives.

**Fix:** Only keep anchors that contain `Abs.` — these are specific enough to match gold format.

```python
anchor_lists_strict = [
    [a for a in anchors if 'Abs.' in a]
    for anchors in anchor_lists
]
```

Effect:
- val_007: 2 false-positive anchors removed ✓
- val_006: `Art. 41 OR` dropped, `Art. 364 Abs. 1 OR` and `Art. 248 Abs. 1 OR` kept ✓
- val_001: `Art. 221 Abs. 1 StPO` kept ✓

---

### KISSKI Model Survey (updated 2026-06-14)

Full model list available at `https://chat-ai.academiccloud.de/v1`:

`apertus-70b-instruct-2509`, `deepseek-r1-distill-llama-70b`, `devstral-2-123b-instruct-2512`, `gemma-4-31b-it`, `glm-4.7`, `internvl3.5-30b-a3b`, `medgemma-27b-it`, `meta-llama-3.1-8b-instruct`, `mistral-large-3-675b-instruct-2512`, `openai-gpt-oss-120b`, `qwen3-30b-a3b-instruct-2507`, `qwen3.5-122b-a10b`, `qwen3.5-397b-a17b`, `teuken-7b-instruct-research`, `llama-3.1-sauerkrautlm-70b-instruct`

---

### Model Comparison — predict_articles_v4

All models used the same `V4_PROMPT` (temp=0, max 10 articles, law code metadata) + strict anchors on all 10 val queries.

| Model | Macro F1 | Delta vs qwen3-30b |
|---|---|---|
| apertus-70b-instruct-2509 | 0.0832 | -0.0049 |
| qwen3-30b-a3b-instruct-2507 | 0.0881 | baseline |
| teuken-7b-instruct-research | 0.0954 | +0.0073 |
| **llama-3.1-sauerkrautlm-70b-instruct** | **0.1225** | **+0.0344** |

**Winner: `llama-3.1-sauerkrautlm-70b-instruct`** — German fine-tune of LLaMA 70B, best F1 by a wide margin.

**Surprising finding: apertus-70b (Swiss-native, ETH/EPFL/CSCS) underperforms sauerkraut-70b.** Being trained on Swiss multilingual corpus does not translate to better citation prediction. SauerkrautLM's German legal text fine-tuning is more relevant for this task.

---

### Combination Approaches — All Hurt F1

| Approach | F1 | Delta vs sauerkraut standalone |
|---|---|---|
| Sauerkraut standalone | 0.1225 | — |
| + Code-prior boosting (top-2 citations per predicted law code) | 0.0920 | -0.0305 |
| Sauerkraut generate → qwen3-30b filter | 0.0551 | -0.0674 |
| qwen3-30b generate → sauerkraut filter | 0.0557 | -0.0668 |
| RRF fusion: sauerkraut + qwen3-30b | 0.0917 | -0.0308 |

**Pattern:** Every combination tried hurts F1. Adding a second model or filtering step inflates false positives without sufficient recall gain. Sauerkraut standalone is the optimal configuration.

---

### Top Leaderboard Architecture Analysis

Analyzed `bettercallagent_legal_rag_qwen3_fast_johny.py` (public leaderboard top, F1=0.3590):

**Their pipeline:**
1. `Qwen3-Embedding-8B` (8B param embedder — 14x larger than our BGE-M3)
2. Retrieve top-500 → RRF fuse to 300
3. `Qwen3-4B` as **reranker** (scores candidates, does NOT generate from scratch)
4. Keep top-15 above `threshold=9.0`, `fallback-k=3`, `fallback-min-votes=2`

**Key architectural difference:** They use LLM as a *reranker* over a large retrieved candidate pool. We use LLM as a *generator* from scratch. Their approach requires a much stronger embedding model to produce quality candidates first. LLM-as-filter with our current retrieval quality does not replicate their gains (confirmed by experiments above).

---

### Updated Scoreboard

| Method | Best F1 | Notes |
|---|---|---|
| Anchor-only | 0.0237 | Regex extraction only |
| BGE-M3 dense laws k=25 | 0.0216 | Below anchor baseline |
| Hybrid anchor+dense k=10 | 0.0332 | 2-way RRF |
| Filtered anchor+dense k=15 | 0.0600 | LLM code prediction + RRF |
| 3-way RRF (anchor+raw+HyDE) k=30 | 0.0678 | HyDE as 3rd channel |
| LLM v2 + anchor (qwen3-30b) | 0.0926 | Previous best |
| **sauerkraut-70b v4 + strict anchors** | **0.1225** | **Current best** |
| Oracle k=25 | 0.7788 | Ceiling |
| Leaderboard top | 0.3590 | Target |

---

### Decisions & Next Steps

- ✅ `llama-3.1-sauerkrautlm-70b-instruct` confirmed as best model — use for all future runs and submission
- ✅ Strict anchor filter (`Abs.` required) — removes val_007 false positives, applied to all future runs
- ✅ LLM-as-filter approach abandoned — too aggressive, destroys recall
- ✅ Code-prior boosting abandoned — generic priors add noise, not signal
- ✅ RRF fusion with second model abandoned — dilutes sauerkraut signal
- ⏳ **Submission pending** — sauerkraut-70b on test.csv (40 queries), rate limit issues addressed with retry logic + 2s sleep between calls
- ⏳ **LoRA fine-tuning** — Kaggle T4 unstable with vanilla PEFT/TRL (CUDA illegal memory access, training hangs). Next attempt: **Unsloth** (2-5x faster, 50% less VRAM) or Google Colab A100
- ❌ Do NOT retry apertus-70b — confirmed worse than sauerkraut despite Swiss-native training
- ❌ Do NOT combine sauerkraut with any second model — standalone is optimal
- ❌ Do NOT use plain parent citations as anchors (no `Abs.`) — val_007 false positives confirmed

### Untested KISSKI Models Worth Trying Next

- `qwen3.5-397b-a17b` — largest available MoE (17B active params), may have stronger legal knowledge
- `mistral-large-3-675b-instruct-2512` — most powerful available, slowest
- `deepseek-r1-distill-llama-70b` — reasoning model, chain-of-thought may help for legal citation prediction
