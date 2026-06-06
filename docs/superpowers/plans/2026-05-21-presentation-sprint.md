# Presentation Sprint Plan — May 21 → May 22, 10 AM

**Deadline:** 2026-05-22 10:00 AM  
**Starting point:** 2026-05-21 6:49 PM  
**Branch:** `feat/bgem3-hybrid-retrieval`

Progress legend: ⬜ pending | 🔄 in progress | ✅ done | ❌ blocked

---

## Quick Status

| Item | Status | Notes |
|------|--------|-------|
| Pipeline code (Sessions 1–4) | ✅ | All committed |
| Acceptance tests | ✅ | 34 passed, 4 skipped |
| GitHub push (5 commits) | ✅ | Pushed to `feat/bgem3-hybrid-retrieval` |
| Kaggle dataset push | 🔄 | Upload in progress |
| BM25 baseline eval | ⬜ | Script to run locally |
| Laws embedding (Kaggle) | ⬜ | User runs notebook |
| Laws-only F1 results | ⬜ | Waiting on notebook |
| Courts embedding (~4-8 hrs) | ⬜ | Starts with notebook |
| Full hybrid F1 results | ⬜ | After courts job |
| Presentation slides | ⬜ | To create tonight |

---

## Block 1 — Infrastructure (Claude) ~15 min

### Task 1: Push code to Kaggle dataset ✅→🔄
**Files:** `src/omnilex/retrieval/__init__.py`, `dense_index.py`, `tests/test_retrieval/test_acceptance.py`  
**Command:** `kaggle datasets version -p . --dir-mode zip -m "Session 4: lazy-import faiss, per-record progress bar, acceptance tests"`  
**Verify:** `kaggle datasets files moeghri/omnilex-retrieval-code` shows `src/`

### Task 2: Push 5 commits to GitHub ✅
**Command:** `git push origin feat/bgem3-hybrid-retrieval`  
**Result:** Pushed successfully (3f36403..01e5001)

### Task 3: BM25 baseline evaluation (local) ⬜
**Script:** `scripts/bm25_eval.py` (to be created)  
**Data:** `data/laws_de.csv` + `data/court_considerations.csv`, scored against `data/val.csv`  
**Output:** `results/bm25_baseline.json`  
**Commit:** `results: BM25 baseline eval on val.csv`

---

## Block 2 — Kaggle Notebook (User) ~25 min

### Task 4: Run Kaggle notebook ⬜
**URL:** https://www.kaggle.com/code/moeghri/omnilex-bgem3-embed  
**Settings:** `SKIP_LAWS_EMBED = False`, `EMBED_COURTS = True`  
**Prerequisite:** Task 1 (Kaggle dataset) must be complete first  
**Output to retrieve:** `results/bgem3_laws_only.json` from Output tab

---

## Block 3 — Presentation (Claude) ~1 hr

### Task 5: Create 7-slide deck ⬜
**Output:** `docs/presentation/2026-05-21-progress-presentation.md` (Marp format)  
**Slides:**
1. The Problem — cross-lingual gap, corpus sizes
2. Why BM25 Fails — token mismatch, near-zero F1
3. Our Approach: BGE-M3 — 3 signals, cross-lingual by design
4. Our Approach: Pipeline — citation anchors → FAISS → RRF → normalize
5. Results — BM25 vs BGE-M3 laws-only vs full (fill after eval)
6. Query Walkthrough — one real val query end-to-end
7. Roadmap — reranker, HyDE, Qwen3-8B, calibration

---

## Block 4 — Results Commit (Claude, after user provides Kaggle output)

### Task 6: Commit laws-only F1 results ⬜
1. Write `results/bgem3_laws_only.json`
2. Fill Session 4 results table in `docs/superpowers/plans/2026-05-18-session-split-execution.md`
3. Update slide 5 with actual numbers
4. Commit: `results: BGE-M3 laws-only eval on val.csv`

---

## Block 5 — Full Hybrid Eval (Tomorrow morning, if courts done)

### Task 7: Session 5 — full hybrid evaluation ⬜
**Prerequisite:** Courts embedding job complete (check Kaggle output for `courts/faiss.index`)  
**Script:** `python scripts/run_evaluation.py --laws-index ... --courts-index ... --val-csv data/val.csv --top-k 2 3 5 10 --output results/bgem3_full.json`  
**Fallback:** If courts not done by 9 AM → present with laws-only + note "courts running"

---

## Block 6 — Final Polish (Tomorrow 8–9 AM)

### Task 8: Pre-presentation checklist ⬜
- [ ] Slide 5 has actual F1 numbers
- [ ] Slide 6 has a real query example
- [ ] `results/` committed and pushed
- [ ] GitHub branch up to date
- [ ] Session 4 + 5 checkboxes marked in progress plan

---

## Results Table (fill as evals complete)

| System | top-k | Macro F1 | Precision | Recall |
|--------|-------|----------|-----------|--------|
| BM25 baseline | — | TBD | TBD | TBD |
| BGE-M3 laws-only | 2 | TBD | TBD | TBD |
| BGE-M3 laws-only | 3 | TBD | TBD | TBD |
| BGE-M3 laws-only | 5 | TBD | TBD | TBD |
| BGE-M3 laws-only | 10 | TBD | TBD | TBD |
| BGE-M3 full hybrid | best-k | TBD | TBD | TBD |

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Kaggle dataset missing `src/` again | Verify with `kaggle datasets files` after push |
| Courts job not done by 9 AM | Present laws-only — still compelling vs BM25 |
| val.csv only 10 queries (noisy) | Acknowledge in presentation; show per-query breakdown |
| BM25 courts index too slow to build (2.4GB) | Run laws-only baseline; courts BM25 is expected to be poor anyway |
