# BGE-M3 Pipeline — Multi-Session Execution Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full BGE-M3 hybrid retrieval pipeline (Tasks 1–13) across 5 focused Claude Sonnet 4.6 sessions, each scoped to a single dependency layer, with GPU-blocking work isolated between sessions.

**Architecture:** Foundation (pure Python) → FAISS Index → Retriever + Scripts → Acceptance Tests + Laws Eval → (GPU wait) → Courts Eval. All sessions run on one branch.

**Tech Stack:** Python 3.10+, FlagEmbedding (BGE-M3), faiss-cpu, scipy, numpy, pandas, pytest.

**Reference implementation:** All code for every task is in `docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md`. Sessions reference it by task number — never rewrite code from scratch.

**Working directory for all sessions:** `Omnilex-Agentic-Retrieval-Competition/`

---

## Implementation Guidelines

These rules apply to every session and every task. Read before starting.

### Branch discipline
- **Never commit to `main`.** All work happens on `feat/bgem3-hybrid-retrieval` for all five sessions.
- Create the branch once before Session 1:
  ```bash
  git checkout -b feat/bgem3-hybrid-retrieval
  ```
- Do not merge between sessions. Open a PR whenever you want a review — merging is optional and non-blocking.

### Progress tracking
- This plan file is the single source of truth for progress. Mark each checkbox as you complete it.
- At the end of each session, commit the updated plan file alongside the code:
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session N tasks complete in progress plan"
  ```
- If a task is skipped or blocked, note it inline next to the checkbox so the next session knows why.

### TDD — Iron Law
```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```
For every task that produces library code:
1. Write the test file
2. Run it — confirm it **fails** with `ModuleNotFoundError` or `ImportError`
3. Write the minimal implementation
4. Run it — confirm it **passes**
5. Commit

If a test passes immediately after writing it (before you wrote the implementation), the test is wrong. Fix it.

**Legitimate TDD exceptions** (CLI scripts only, Tasks 9 and 10): these require real 176K-row CSV files that don't exist in CI. Compensate by: (a) verifying the script imports cleanly, (b) running with `--limit 10` as a smoke test once data is available.

### Karpathy: Simplicity First
- **IndexFlatIP, not IndexHNSWFlat.** At 176K laws rows, exact search is ~100 ms on GPU — fast enough. HNSW adds approximation error and tuning parameters for no practical gain at this scale.
- Write the minimum code that makes the test pass. No extra parameters, no "flexibility" hooks.
- If an implementation is 200 lines and could be 50, rewrite it.

### Karpathy: Surgical Changes
- Each session touches only the files listed in its checklist. Do not refactor adjacent code.
- Do not "improve" existing files (style, formatting, comments) while in a task.
- If you notice unrelated dead code, note it in the PR description — do not delete it.

### Karpathy: Goal-Driven Execution
- Each task has a binary gate: `pytest -v` passes, or it doesn't. Don't mark a task done until the gate passes.
- Each session has a session gate: the full test suite runs clean. Don't commit the progress update until it does.
- If a verification fails twice in a row, stop and ask — do not push through.

### PR workflow
- Open a PR from `feat/bgem3-hybrid-retrieval` to `main` after any session you want reviewed.
- PRs are for review and visibility — merging is **not required** before continuing to the next session.
- A single final PR after Session 5 is also fine if you prefer to review everything at once.

---

## Session Overview

| Session | Tasks | Requires GPU? | Est. tokens | Est. time |
|---------|-------|--------------|-------------|-----------|
| 1 | 1, 2, 3, 4, 5, 11 | No | ~150k | 1.5–2 hrs |
| 2 | 6, 7 | No (stub tests) | ~150k | 1–1.5 hrs |
| 3 | 8, 9, 10 | No | ~150k | 1.5 hrs |
| 4 | 12 + Task 13 steps 1–3 | Yes (laws embed) | ~100k | 1 hr + 20 min GPU |
| 5 | Task 13 steps 4–7 | Yes (courts result) | ~50k | 30 min |

**Total:** ~600k tokens, ~6–7 hrs coding, ~5–9 hrs GPU wait (runs unattended).

---

## Chunk 1: Session 1 — Foundation

Pure Python. No ML libraries required. All tests pass before `faiss` or `FlagEmbedding` are installed.

**Tasks:** 1 (models protocols), 2 (fusion), 3 (anchor extractor), 4 (submission), 5 (\_\_init\_\_.py), 11 (requirements.txt)

---

### How to start Session 1

Open a new Claude Code session in `Omnilex-Agentic-Retrieval-Competition/` and paste this prompt:

```
I need you to implement Tasks 1, 2, 3, 4, 5, and 11 from the BGE-M3 pipeline plan.

Reference plan (all code): docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md
Guidelines: docs/superpowers/plans/2026-05-18-session-split-execution.md (read the
"Implementation Guidelines" section first)

Working directory: Omnilex-Agentic-Retrieval-Competition/
Branch: feat/bgem3-hybrid-retrieval (switch to it if not already on it)

For each task, follow TDD strictly:
1. Write the test, run it (expect fail)
2. Write the minimal implementation, run tests (expect pass)
3. Commit

Do NOT install FlagEmbedding or faiss — these tasks are pure Python.

After all 6 tasks: pytest tests/test_retrieval/ -v — all must pass.
Then mark the Session 1 checkboxes in the progress plan and commit it.
```

---

### Session 1 checklist

- [ ] **Confirm branch:** `git branch` shows `feat/bgem3-hybrid-retrieval`

- [ ] **Task 1** — `src/omnilex/retrieval/models.py` (EmbeddingModel + RerankerModel protocols)
  - Tests: `tests/test_retrieval/test_models.py` (4 tests — watch each fail first)
  - Commit: `feat(retrieval): add EmbeddingModel and RerankerModel protocols`

- [ ] **Task 2** — `src/omnilex/retrieval/fusion.py` (Candidate dataclass + rrf\_fuse + deduplicate\_candidates)
  - Tests: `tests/test_retrieval/test_fusion.py` (7 tests)
  - Commit: `feat(retrieval): add RRF fusion and candidate deduplication`

- [ ] **Task 3** — `src/omnilex/retrieval/anchor_extractor.py`
  - Tests: `tests/test_retrieval/test_anchor_extractor.py` (7 tests)
  - Commit: `feat(retrieval): add citation-anchor extraction from query text`

- [ ] **Task 4** — `src/omnilex/retrieval/submission.py`
  - Tests: `tests/test_retrieval/test_submission.py` (5 tests)
  - Commit: `feat(retrieval): add submission CSV generator with top-k selection`

- [ ] **Task 5** — `src/omnilex/retrieval/__init__.py` (add exports for all new modules)
  - No new tests; run full suite to verify nothing broke
  - Commit: `feat(retrieval): export fusion, anchor, and submission modules`

- [ ] **Task 11** — `requirements.txt` (append FlagEmbedding>=1.2, faiss-cpu>=1.7.4, torch>=2.0, transformers>=4.40)
  - Commit: `deps: add FlagEmbedding, faiss-cpu, torch, transformers`

- [ ] **Session 1 gate:** `pytest tests/test_retrieval/ -v` — all tests green

- [ ] **Commit progress update:**
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session 1 tasks complete in progress plan"
  ```

---

### Session 1 exit state

- 4 new files committed: `models.py`, `fusion.py`, `anchor_extractor.py`, `submission.py`
- 2 modified files committed: `__init__.py`, `requirements.txt`
- `tests/test_retrieval/__init__.py` created
- All tests green, progress plan updated

---

## Chunk 2: Session 2 — FAISS Index

Adds the BGE-M3 adapter and FAISS index builder. Tests use stub embedders — GPU not required.

**Tasks:** 6 (BgeM3Embedder), 7 (dense\_index.py)

**Prerequisite:** Session 1 complete (checkboxes marked).

---

### How to start Session 2

Paste this prompt in a new Claude Code session:

```
I need you to implement Tasks 6 and 7 from the BGE-M3 pipeline plan.

Reference plan (all code): docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md
Guidelines: docs/superpowers/plans/2026-05-18-session-split-execution.md

Working directory: Omnilex-Agentic-Retrieval-Competition/
Branch: feat/bgem3-hybrid-retrieval

First: uv pip install -r requirements.txt

Follow TDD for both tasks. Commit each separately.

Task 6: Extend src/omnilex/retrieval/models.py with BgeM3Embedder.
TestBgeM3Embedder uses pytest.importorskip — it may SKIP without the model downloaded.
All existing protocol tests must still PASS.

Task 7: Create src/omnilex/retrieval/dense_index.py.
Use IndexFlatIP (not HNSW) — exact search is sufficient at this corpus size.
Tests use StubEmbedder, no GPU needed.

After both tasks: pytest tests/test_retrieval/ -v — all pass (BgeM3 may skip).
Then mark the Session 2 checkboxes in the progress plan and commit it.
```

---

### Session 2 checklist

- [x] Install deps: `uv pip install -r requirements.txt`

- [x] **Task 6** — Extend `src/omnilex/retrieval/models.py` with `BgeM3Embedder`
  - Tests: append `TestBgeM3Embedder` to `tests/test_retrieval/test_models.py` (may skip without model)
  - Existing `TestEmbeddingModel` and `TestRerankerModel` must still pass
  - Commit: `feat(retrieval): add BgeM3Embedder adapter wrapping FlagEmbedding`

- [x] **Task 7** — Create `src/omnilex/retrieval/dense_index.py` (DenseIndexBuilder + DenseIndex, IndexFlatIP)
  - Tests: `tests/test_retrieval/test_dense_index.py` (4 tests, all use StubEmbedder)
  - Commit: `feat(retrieval): add DenseIndexBuilder and DenseIndex with FAISS`

- [x] **Session 2 gate:** `pytest tests/test_retrieval/ -v` — 27 passed, 4 skipped (BgeM3 skipped — FlagEmbedding not installed)

- [x] **Commit progress update:**
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session 2 tasks complete in progress plan"
  ```

---

### Session 2 exit state

- 1 modified file committed: `models.py` (BgeM3Embedder appended)
- 1 new file committed: `dense_index.py`
- All tests passing (BgeM3Embedder tests may skip), progress plan updated

---

## Chunk 3: Session 3 — Retriever + Scripts

Wires the full pipeline together: retriever, embedding CLI, evaluation harness.

**Tasks:** 8 (dense\_retriever.py), 9 (embed\_corpus.py), 10 (run\_evaluation.py)

**Prerequisite:** Session 2 complete (checkboxes marked).

---

### How to start Session 3

```
I need you to implement Tasks 8, 9, and 10 from the BGE-M3 pipeline plan.

Reference plan (all code): docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md
Guidelines: docs/superpowers/plans/2026-05-18-session-split-execution.md

Working directory: Omnilex-Agentic-Retrieval-Competition/
Branch: feat/bgem3-hybrid-retrieval

Task 8: Create src/omnilex/retrieval/dense_retriever.py.
Full TDD — tests use StubEmbedder and a temp FAISS index, no GPU needed.

Task 9: Create scripts/embed_corpus.py.
TDD exception (requires real CSV data). Verify it imports cleanly:
  python -c "import runpy; runpy.run_path('scripts/embed_corpus.py')" 2>&1 | head -5
Commit it.

Task 10: Create scripts/run_evaluation.py.
TDD exception (requires real indices). Verify it imports cleanly.
Commit it.

Also update src/omnilex/retrieval/__init__.py to export DenseIndex, DenseRetriever, BgeM3Embedder.

After all tasks: pytest tests/test_retrieval/ -v — all must pass.
Then mark the Session 3 checkboxes in the progress plan and commit it.
```

---

### Session 3 checklist

- [ ] **Task 8** — Create `src/omnilex/retrieval/dense_retriever.py`
  - Tests: `tests/test_retrieval/test_dense_retriever.py` (2 tests, full TDD)
  - Commit: `feat(retrieval): add DenseRetriever with anchor extraction and RRF fusion`

- [ ] **Task 9** — Create `scripts/embed_corpus.py`
  - TDD exception — verify clean import, smoke-test with `--limit 10` when data available
  - Commit: `feat(scripts): add embed_corpus.py for building dense FAISS indices`

- [ ] **Task 10** — Create `scripts/run_evaluation.py`
  - TDD exception — verify clean import
  - Commit: `feat(scripts): add run_evaluation.py with multi-k comparison`

- [ ] Update `src/omnilex/retrieval/__init__.py` to export `DenseIndex`, `DenseRetriever`, `BgeM3Embedder`
  - Commit: `feat(retrieval): export dense index, retriever, and model classes`

- [ ] **Session 3 gate:** `pytest tests/test_retrieval/ -v` — all tests green

- [ ] **Commit progress update:**
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session 3 tasks complete in progress plan"
  ```

---

### Session 3 exit state

- 3 new files committed: `dense_retriever.py`, `scripts/embed_corpus.py`, `scripts/run_evaluation.py`
- `__init__.py` updated, full test suite green, progress plan updated

---

## Chunk 4: Session 4 — Acceptance Tests + Laws Embedding

Write acceptance tests, kick off the laws embedding job, and get the first F1 signal.

**Tasks:** 12 (acceptance tests), Task 13 steps 1–3

**Prerequisite:** Session 3 complete (checkboxes marked). GPU access available.

---

### How to start Session 4

```
I need you to implement Task 12 and run Task 13 steps 1-3 from the BGE-M3 pipeline plan.

Reference plan (all code): docs/superpowers/plans/2026-05-17-bgem3-retrieval-pipeline.md
Guidelines: docs/superpowers/plans/2026-05-18-session-split-execution.md

Working directory: Omnilex-Agentic-Retrieval-Competition/
Branch: feat/bgem3-hybrid-retrieval

Task 12: Create tests/test_retrieval/test_acceptance.py.
Full TDD: write each test class, watch it fail, write the fixture, watch it pass.
All 5 test classes must pass before continuing.

Task 13 steps 1-3 (GPU required):
  Step 1: python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws
  Step 2: Verify the index using the verification command in the reference plan
  Step 3: python scripts/run_evaluation.py \
            --laws-index data/processed/bge_m3/laws \
            --val-csv data/raw/lexam/val.csv \
            --top-k 2 3 5 10 \
            --output results/bgem3_laws_only.json

Print Macro F1 at each k. Commit results.
Then kick off the courts embedding job and mark Session 4 checkboxes in the progress plan.
```

---

### Session 4 checklist

- [ ] **Task 12** — Create `tests/test_retrieval/test_acceptance.py`
  - 5 test classes: FAISS alignment, normalization roundtrip, RRF determinism, submission format, top-k bounds
  - Watch each test class fail before writing fixtures, then watch them pass
  - Run: `pytest tests/test_retrieval/test_acceptance.py -v` — all pass
  - Commit: `test(retrieval): add acceptance tests for pipeline correctness`

- [ ] **Task 13 — Step 1:** Build laws index (~20 min on GPU)
  ```bash
  python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws
  ```

- [ ] **Task 13 — Step 2:** Verify laws index
  ```bash
  python -c "
  from omnilex.retrieval.dense_index import DenseIndex
  idx = DenseIndex.load('data/processed/bge_m3/laws')
  print(f'Laws index: {len(idx.metadata)} passages, dim={idx.index.d}')
  print(f'First citation: {idx.metadata[0][\"citation_raw\"]}')
  "
  ```

- [ ] **Task 13 — Step 3:** Laws-only evaluation
  ```bash
  python scripts/run_evaluation.py \
      --laws-index data/processed/bge_m3/laws \
      --val-csv data/raw/lexam/val.csv \
      --top-k 2 3 5 10 \
      --output results/bgem3_laws_only.json
  ```

- [ ] Commit results: `git add results/ && git commit -m "results: BGE-M3 laws-only eval on val.csv"`

- [ ] **Start courts job** (runs unattended, 4–8 hours):
  ```bash
  python scripts/embed_corpus.py --corpus courts --output data/processed/bge_m3/courts
  ```

- [ ] **Session 4 gate:** `pytest tests/test_retrieval/ -v` — all tests green

- [ ] **Commit progress update** (fill in the results table below before committing):
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session 4 complete, add laws-only F1 results"
  ```

  Laws-only Macro F1 results:

  | top-k | Macro F1 | Precision | Recall |
  |-------|----------|-----------|--------|
  | 2     |          |           |        |
  | 3     |          |           |        |
  | 5     |          |           |        |
  | 10    |          |           |        |

---

### Session 4 exit state

- `test_acceptance.py` committed and passing
- `results/bgem3_laws_only.json` committed
- Courts embedding job running in background
- Progress plan updated with F1 results

---

## Chunk 5: Session 5 — Courts Evaluation (post-GPU)

Short session. Run after the courts embedding job completes.

**Prerequisite:** `data/processed/bge_m3/courts/faiss.index` exists on disk.

---

### How to start Session 5

```
The BGE-M3 courts embedding job has finished.
Working directory: Omnilex-Agentic-Retrieval-Competition/
Branch: feat/bgem3-hybrid-retrieval

Run the full hybrid evaluation:
python scripts/run_evaluation.py \
    --laws-index data/processed/bge_m3/laws \
    --courts-index data/processed/bge_m3/courts \
    --val-csv data/raw/lexam/val.csv \
    --top-k 2 3 5 10 \
    --output results/bgem3_full.json

Compare to the BM25 baseline using scripts/evaluate_submission.py on val.csv.
Print a side-by-side table: BM25 vs BGE-M3 laws-only vs BGE-M3 full.
Note the best top-k for the final Kaggle submission.

Commit results. Mark the Session 5 checkboxes in the progress plan, fill in the
results table, and commit the plan.
```

---

### Session 5 checklist

- [ ] Run full hybrid evaluation (laws + courts)

- [ ] Compare BM25 baseline vs BGE-M3 laws-only vs BGE-M3 full

- [ ] Commit: `git add results/bgem3_full.json && git commit -m "results: BGE-M3 full hybrid eval (laws + courts)"`

- [ ] Note best top-k for Kaggle submission

- [ ] **Commit progress update with final results table:**
  ```bash
  git add docs/superpowers/plans/2026-05-18-session-split-execution.md
  git commit -m "chore: mark Session 5 complete, add final eval results"
  ```

  Final comparison:

  | System | top-k | Macro F1 |
  |--------|-------|----------|
  | BM25 baseline | — | |
  | BGE-M3 laws-only | | |
  | BGE-M3 full hybrid | | |

- [ ] **Open PR when ready for review** (optional):
  ```bash
  gh pr create \
    --title "feat: BGE-M3 hybrid retrieval pipeline" \
    --body "Implements Tasks 1-13. See docs/superpowers/plans/2026-05-18-session-split-execution.md for results."
  ```

---

## Dependency Graph

```
Session 1 (pure Python, no GPU)
    └── Session 2 (FAISS index, no GPU)
            └── Session 3 (retriever + scripts, no GPU)
                    └── Session 4 (acceptance tests + laws eval, GPU ~20 min)
                                │
                                └──► courts job starts (GPU 4-8 hrs, unattended)
                                                │
                                         Session 5 (courts eval, ~30 min)
```

All sessions run on `feat/bgem3-hybrid-retrieval`. Progress tracked via checkboxes in this file.

---

## Troubleshooting

**`ModuleNotFoundError: omnilex`** — activate venv first: `source .venv/bin/activate`

**Wrong branch** — `git checkout feat/bgem3-hybrid-retrieval` before writing any code.

**FAISS shape mismatch** — `normalize_L2` must be called before both `index.add()` and `index.search()`.

**`canonicalize_list`** — confirmed at `src/omnilex/citations/normalizer.py:145`.

**Courts embedding OOM** — add `--batch-size 64` to the `embed_corpus.py` call.

**val.csv column name mismatch** — check the header row; adjust `gold_citations`/`query_id` in `run_evaluation.py` if needed.

**Test passes immediately (before implementation)** — the test is wrong; fix it before proceeding.

**Verification fails twice** — stop, do not push through. Ask for help.