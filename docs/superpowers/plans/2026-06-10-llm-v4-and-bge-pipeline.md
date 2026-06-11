# LLM Article Prediction v4 + BGE/Docket Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. On start, copy this plan to `docs/superpowers/plans/2026-06-10-llm-v4-and-bge-pipeline.md` and check off steps there.

**Goal:** Raise val Macro F1 from the current best 0.0926 (LLM article prediction v2 + anchor) toward ≥ 0.12 by (a) making LLM prediction deterministic and count-calibrated, and (b) opening the BGE/docket channel that currently caps recall (102/251 gold citations are court decisions, unreachable from the laws corpus).

**Architecture:** All work happens in the Kaggle notebook `moeghri/dense-retrieval-bge-m3-laws-eval` (the Sessions 7–9 code lives ONLY there — it is not in the local repo). The pipeline is LLM-first: KISSKI API predicts law articles directly from the English query; regex anchors are unioned in; a citation bridge (law article → court decisions citing it) and a direct-BGE-prediction experiment attack the court-decision gap. Dense retrieval is OFF (confirmed noise source, Session 9). No GPU is required for any task in this plan.

**Tech Stack:** Kaggle notebook + kaggle CLI push cycles, KISSKI API (`qwen3-30b-a3b-instruct-2507` via OpenAI SDK), `omnilex.citations.normalizer.CitationNormalizer`, `omnilex.evaluation.metrics.citation_f1`.

---

## Context

**Where we are** (from `docs/superpowers/session-notes.md`, Sessions 7–9, dated 2026-06-09):

| Method | Macro F1 | Status |
|---|---|---|
| Anchor-only (regex) | 0.0237 | floor |
| BGE-M3 dense, laws k=25 | 0.0216 | below floor — dense abandoned as primary |
| Filtered anchor+dense k=15 | 0.0600 | superseded |
| **LLM article prediction v2 + anchor** | **0.0926** | **current best** |
| Oracle k=25 | 0.7788 | ceiling |
| Leaderboard top | 0.3590 | target |

**Open problems this plan addresses:**
1. **LLM stochasticity** — val_004 varies 0–5 TP between runs at temperature=0.1. Session 9 started `predict_articles_v4` (temperature=0, line-based output) but it is unfinished.
2. **Count calibration** — gold citations per query range 10–47 (mean 25.1); capping predictions at 10 caps recall. Need a confidence-ordered prediction list + truncation sweep.
3. **val_007 = 0 TP** across all runs despite 2 correct anchors. Undiagnosed.
4. **BGE/docket gap** — 102/251 gold citations (41%) are court decisions. Courts dense retrieval is a confirmed dead end (0 hits in top-100 at 2.4M docs). The leaderboard-top solution uses a citation bridge instead.

**Hard constraints (empirically confirmed — do NOT revisit):**
- ❌ NO ensemble / union of multiple LLM runs (inflates FP, F1 drops — Session 9)
- ❌ NO dense retrieval on top of LLM predictions (0.0882 < 0.0926 — Session 9)
- ❌ NO courts-corpus dense search (0 hits — Session 8)
- ❌ NO naive citation bridge on the full prediction set (0.0445 < 0.0600 — Session 9; this plan retries it ONLY with high-confidence seeds)
- ✅ ALWAYS evaluate against the 251-citation gold set via `CitationNormalizer` (docket-aware). Never compare with old 218-gold numbers.

---

## Operational Workflow (read before Task 1)

**Kernel:** `moeghri/dense-retrieval-bge-m3-laws-eval` · **push dir:** `notebooks/bge_m3_push/` · **code dataset:** `moeghri/omnilex-retrieval-code` (v3, has docket-aware normalizer)

Every push triggers a full "Save & Run All" on Kaggle. The iteration loop:

```bash
# from repo root: Omnilex-Agentic-Retrieval-Competition/
kaggle kernels push -p notebooks/bge_m3_push
kaggle kernels status moeghri/dense-retrieval-bge-m3-laws-eval     # poll every ~90s until "complete" / "error"
kaggle kernels output moeghri/dense-retrieval-bge-m3-laws-eval -p results/kaggle_runs/<run-name>
```

- **Batch pushes:** Push A = Tasks 2+3+4 together (one run). Push B = Tasks 5+6 together. Two push cycles total, not six.
- **Runtime budget:** with dense skipped, a run is ~5–15 min (LLM API calls only). Task 5's bridge build adds ~30–40 min on its first run only.
- **GPU:** set `"enable_gpu": false` in `kernel-metadata.json` (Task 1). Nothing in this plan needs GPU, and this sidesteps the P100/sm_60 `model.half()` crash entirely.
- **KISSKI key:** the pulled notebook already reads it via Kaggle secrets (Sessions 8–9 pattern, `UserSecretsClient`). Reuse that cell verbatim. NEVER hardcode the key.
- **On run error:** download output anyway (`kernels output` includes the log), read the traceback, fix locally, re-push. Apply systematic-debugging: read the actual error before changing anything.

---

### Task 1: Snapshot the Kaggle notebook + prep the run config

The Sessions 7–9 code (LLM filtering, HyDE, `predict_articles_v1–v3`, FR_TO_DE dict, bridge prototype) exists only in the Kaggle kernel. First job: get it into git, then make runs cheap.

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (replaced by pulled latest version)
- Modify: `notebooks/bge_m3_push/kernel-metadata.json`
- Create: `docs/superpowers/plans/2026-06-10-llm-v4-and-bge-pipeline.md` (copy of this plan)

- [ ] **Step 1: Pull the latest kernel version**

```bash
kaggle kernels pull moeghri/dense-retrieval-bge-m3-laws-eval -p notebooks/bge_m3_push -m
```

Expected: downloads the notebook + metadata. **Check the downloaded filename** — if Kaggle saves it as `dense-retrieval-bge-m3-laws-eval.ipynb`, either rename it to `03_dense_retrieval_bge_m3.ipynb` or update `code_file` in `kernel-metadata.json` to match. The `id` field must stay `moeghri/dense-retrieval-bge-m3-laws-eval`.

- [ ] **Step 2: Verify the pulled notebook contains Sessions 7–9 work**

Search the pulled `.ipynb` for the strings `predict_articles`, `FR_TO_DE`, and `bridge`. All three must be present. If any is missing, STOP and tell the user — the pull got a stale version and pushing would destroy the Session 9 work.

- [ ] **Step 3: Commit the snapshot**

```bash
git add notebooks/bge_m3_push/
git commit -m "chore: snapshot Kaggle kernel state after Session 9 (LLM v1-v3, bridge prototype)"
```

This is the safety net: every later push can be diffed against this commit.

- [ ] **Step 4: Disable GPU and skip dense cells**

In `kernel-metadata.json` set `"enable_gpu": false`. In the notebook, add a config cell near the top:

```python
RUN_DENSE = False   # dense retrieval confirmed as noise (Session 9); skip embedding entirely
```

and guard the model-load / embed / FAISS cells with:

```python
if not RUN_DENSE:
    print("RUN_DENSE=False — skipping dense retrieval cells")
else:
    ...existing cell body, indented...
```

(If a guarded cell defines names later cells need — e.g. `faiss_index` — guard those consumers the same way. The anchor, LLM, and eval cells must NOT depend on dense outputs; the Session 9 best pipeline is LLM+anchor only, so they shouldn't.)

- [ ] **Step 5: Copy this plan into the repo and commit**

```bash
git add docs/superpowers/plans/2026-06-10-llm-v4-and-bge-pipeline.md
git commit -m "docs: add LLM v4 + BGE pipeline implementation plan"
```

---

### Task 2: `predict_articles_v4` — deterministic, confidence-ordered prediction

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (new cells appended after the Session 9 LLM cells)

- [ ] **Step 1: Add the v4 prediction cell**

Reuse the existing `client` (KISSKI OpenAI client) and `FR_TO_DE` dict from the pulled notebook. If the existing `FR_TO_DE` lacks entries, extend it — do not replace it. Add this cell:

```python
import re

LLM_MODEL = "qwen3-30b-a3b-instruct-2507"   # Session 9: outperformed the 122b on this task
MAX_ARTICLES_REQUEST = 25                    # request many, truncate later (Task 3 calibrates)

V4_SYSTEM = """You are a Swiss legal expert. Given a case description, determine which \
Swiss federal law articles a court would cite when deciding this case.

Work in two steps:

ISSUES:
List the distinct legal issues in the case, one line each.

CITATIONS:
List the specific applicable articles, ONE PER LINE, format:
Art. <number> [Abs. <number>] <GERMAN abbreviation>
Example: Art. 221 Abs. 1 StPO

Rules:
- Use GERMAN law abbreviations only (ZGB, OR, StGB, StPO, ZPO, SchKG, BGG, BV, IVG, AVIG, UVG, KVG, ATSG, ...). Never French (CC, CO, CP, CPP, LAI, LACI) or Italian.
- Order citations from MOST confident to LEAST confident.
- List at most {n} citations. Quality over quantity.
- After the citation list, output nothing else."""


def predict_articles_v4(query: str, n: int = MAX_ARTICLES_REQUEST) -> list[str]:
    """Deterministic article prediction. Returns canonical citations, confidence-ordered."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": V4_SYSTEM.format(n=n)},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "CITATIONS" in text:
        text = text.split("CITATIONS", 1)[1]   # discard the ISSUES reasoning block

    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip(" -*•:\t")
        if "Art." not in line:
            continue
        line = line[line.index("Art."):]
        for fr, de in FR_TO_DE.items():        # French→German abbreviation fix (Session 9)
            line = re.sub(rf"\b{re.escape(fr)}\b", de, line)
        canon = normalizer.canonicalize(line)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def expand_parents(citations: list[str]) -> list[str]:
    """Session 9 fix: gold often has 'Art. 467 ZGB' where LLM says 'Art. 467 Abs. 1 ZGB'.
    Add the Abs.-stripped parent form for every prediction that has an Abs."""
    out, seen = [], set()
    for c in citations:
        for form in (c, re.sub(r"\s+Abs\.\s+\d+", "", c)):
            if form not in seen:
                seen.add(form)
                out.append(form)
    return out
```

- [ ] **Step 2: Add the shared evaluation helper cell**

One canonical eval function used by every later task (reuses `citation_f1` from `omnilex.evaluation.metrics`, same as the Session 7–9 cells):

```python
def macro_f1_for(preds_by_qid: dict) -> tuple[float, list[dict]]:
    """preds_by_qid: query_id -> iterable of canonical citation strings."""
    rows, f1s = [], []
    for q in val_queries:
        qid = q["query_id"]
        gold = gold_sets_by_qid[qid]           # already built in the Session 7 eval cells
        pred = set(preds_by_qid.get(qid, []))
        s = citation_f1(list(pred), list(gold))
        f1s.append(s["f1"])
        rows.append({"query_id": qid, "gold": len(gold), "pred": len(pred),
                     "tp": len(pred & gold), "f1": round(s["f1"], 4)})
    return sum(f1s) / len(f1s), rows
```

(If the pulled notebook names the gold-set dict differently — e.g. `gold_sets` indexed by position — adapt the lookup to the existing structure rather than rebuilding it.)

- [ ] **Step 3: Add the determinism-check cell**

```python
val_004 = next(q for q in val_queries if q["query_id"] == "val_004")
runs = [predict_articles_v4(val_004["query"]) for _ in range(3)]
print("Run lengths:", [len(r) for r in runs])
print("Identical across runs:", runs[0] == runs[1] == runs[2])
gold4 = gold_sets_by_qid["val_004"]
for i, r in enumerate(runs):
    tp = len(set(expand_parents(r)) & gold4)
    print(f"  run {i}: {tp} TP")
```

- [ ] **Step 4: Add the v4 full-val prediction + baseline-comparison cell**

```python
v4_preds_ordered = {}                       # keep ordered lists — Task 3 truncates these
for q in tqdm(val_queries, desc="v4 predict"):
    v4_preds_ordered[q["query_id"]] = predict_articles_v4(q["query"])

# anchors_by_qid: from the Session 7 anchor cell (3/10 queries have anchors)
v4_full = {qid: set(expand_parents(lst)) | set(anchors_by_qid.get(qid, []))
           for qid, lst in v4_preds_ordered.items()}
macro, rows = macro_f1_for(v4_full)
print(f"v4 (untruncated, n=25) + anchor: Macro F1 = {macro:.4f}   [v2 baseline: 0.0926]")
for r in rows: print(r)
```

- [ ] **Step 5: Verification gate (checked after Push A completes, see Task 4 Step 3)**

Expected: identical prediction lists across the 3 determinism runs (temperature=0). v4+anchor Macro F1 in the 0.08–0.13 range. **Gate:** if v4 is deterministic AND ≥ 0.0926 → v4 becomes the new base. If deterministic but < 0.0926, Task 3's truncation sweep decides (untruncated n=25 may over-predict — that's expected and fine). If NOT deterministic, record the variance per run in session notes and proceed anyway (KISSKI backend nondeterminism is out of our control).

---

### Task 3: Citation-count calibration — truncation sweep

Gold counts per query: 10–47 (mean 25.1). One LLM call per query (already made in Task 2 Step 4) supports the whole sweep — truncation is free, no extra API calls.

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (one new cell)

- [ ] **Step 1: Add the truncation sweep cell**

```python
print(f"{'N':>4}  {'Macro F1':>9}  {'mean pred':>9}")
sweep = {}
for N in [5, 8, 10, 12, 15, 20, 25]:
    preds = {qid: set(expand_parents(lst[:N])) | set(anchors_by_qid.get(qid, []))
             for qid, lst in v4_preds_ordered.items()}
    macro, rows = macro_f1_for(preds)
    sweep[N] = macro
    mean_pred = sum(r["pred"] for r in rows) / len(rows)
    print(f"{N:>4}  {macro:>9.4f}  {mean_pred:>9.1f}")

BEST_N = max(sweep, key=sweep.get)
print(f"\nBest N = {BEST_N}, Macro F1 = {sweep[BEST_N]:.4f}  [previous best: 0.0926]")

# Freeze the best config for downstream tasks
v4_best = {qid: set(expand_parents(lst[:BEST_N])) | set(anchors_by_qid.get(qid, []))
           for qid, lst in v4_preds_ordered.items()}
```

- [ ] **Step 2: Verification gate (after Push A)**

Expected: a clear F1-vs-N curve, peak typically at N=10–15 given precision ~0.2–0.3. **Success: `sweep[BEST_N] ≥ 0.0926`.** If the whole curve is below 0.0926, v4's prompt regressed vs v2 — compare v4 predictions against the v2 predictions still in the notebook for 2–3 queries, identify the difference (likely the confidence-ordering instruction diluting quality), and report findings rather than blindly iterating prompts.

---

### Task 4: val_007 diagnostic

val_007 (19 gold: 15 Art + 4 BGE) gets 0 LLM TP across all runs despite 2 correct anchors (`Art. 934 ZGB`, `Art. 936 ZGB` — possession/good-faith acquisition). One diagnostic cell, no open-ended prompt tuning.

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (one new cell)

- [ ] **Step 1: Add the diagnostic cell**

```python
q7 = next(q for q in val_queries if q["query_id"] == "val_007")
gold7 = sorted(gold_sets_by_qid["val_007"])
pred7 = v4_preds_ordered["val_007"]

print("QUERY:\n", q7["query"][:600], "\n")
print(f"GOLD ({len(gold7)}):")
for g in gold7: print("  ", g)
print(f"\nLLM v4 PREDICTED ({len(pred7)}):")
for p in pred7: print("  ", p)

# Near-miss analysis: same law book? adjacent article numbers?
def book(c):
    m = re.search(r"([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+)$", c)
    return m.group(1) if m else "?"

def artno(c):
    m = re.search(r"Art\.\s+(\d+)", c)
    return int(m.group(1)) if m else None

gold_books = {book(g) for g in gold7 if g.startswith("Art.")}
pred_books = {book(p) for p in pred7}
print(f"\nGold books     : {sorted(gold_books)}")
print(f"Predicted books: {sorted(pred_books)}")
print(f"Book overlap   : {sorted(gold_books & pred_books)}")
near = [(p, g) for p in pred7 for g in gold7
        if g.startswith("Art.") and book(p) == book(g)
        and artno(p) and artno(g) and abs(artno(p) - artno(g)) <= 3]
print(f"Near misses (same book, ±3 articles): {near}")
```

- [ ] **Step 2: Push A — push Tasks 2+3+4 together and run**

```bash
kaggle kernels push -p notebooks/bge_m3_push
# poll:
kaggle kernels status moeghri/dense-retrieval-bge-m3-laws-eval
# when complete:
kaggle kernels output moeghri/dense-retrieval-bge-m3-laws-eval -p results/kaggle_runs/push_a_v4_calibration
```

- [ ] **Step 3: Verify Push A results against the gates in Task 2 Step 5 and Task 3 Step 2**

Read the downloaded log. Record: determinism result, v4 F1, sweep table, BEST_N, val_007 diagnosis. For val_007, the deliverable is a **written hypothesis** in session notes (e.g. "LLM predicts SchKG debt-collection articles but gold is ZGB property law — query framing misleads the issue-spotting step"), not a fix. If the diagnosis suggests a one-line prompt tweak with a plausible mechanism, note it for Push B; otherwise move on.

- [ ] **Step 4: Commit**

```bash
git add notebooks/bge_m3_push/ results/kaggle_runs/push_a_v4_calibration/
git commit -m "feat: v4 deterministic LLM article prediction + count calibration (Push A)"
```

---

### Task 5: Citation bridge with high-confidence seeds (BGE/docket channel)

Session 9 abandoned the bridge because feeding it the *full* (mostly wrong) prediction set expanded noise: 0.0445 < 0.0600. The retry hypothesis: seed the bridge ONLY with high-confidence articles — **anchors + LLM top-3 per query** — and cap expansion tightly. Anchors are near-certain; LLM rank-1–3 articles are the most reliable (confidence-ordered v4 output).

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (bridge cells — the Session 8/9 bridge-build code already exists in the pulled notebook; reuse its build function, replace its usage)

- [ ] **Step 1: Locate and reuse the existing bridge build**

The pulled notebook has Session 8's bridge construction (regex-extract `Art.` patterns from each of 2.4M court-decision texts → `dict[canonical Art. citation, list[court citation_raw]]`, cached to `citation_bridge.json`). Keep that build cell as-is, but make it cache-aware:

```python
BRIDGE_FILE = Path("/kaggle/working/citation_bridge.json")
BRIDGE_DATASET_COPY = Path("/kaggle/input/omnilex-retrieval-code/citation_bridge.json")

if BRIDGE_FILE.exists():
    bridge = json.load(open(BRIDGE_FILE, encoding="utf-8"))
elif BRIDGE_DATASET_COPY.exists():
    bridge = json.load(open(BRIDGE_DATASET_COPY, encoding="utf-8"))
    print(f"Bridge loaded from dataset: {len(bridge):,} articles")
else:
    bridge = build_citation_bridge()        # existing Session 8 function, ~30-40 min
    json.dump(bridge, open(BRIDGE_FILE, "w", encoding="utf-8"))
```

- [ ] **Step 2: Add the seeded-bridge expansion + sweep cell**

```python
def bridge_candidates(qid: str, seed_top: int = 3, max_courts: int = 5) -> list[str]:
    """High-confidence seeds only: anchors + LLM top-`seed_top`. Returns canonical
    court citations ranked by how many seeds cite-link to them."""
    seeds = list(anchors_by_qid.get(qid, [])) + v4_preds_ordered[qid][:seed_top]
    counts = {}
    for seed in seeds:
        for court_raw in bridge.get(seed, []):
            c = normalizer.canonicalize(court_raw)
            if c:
                counts[c] = counts.get(c, 0) + 1
    ranked = sorted(counts, key=lambda c: -counts[c])
    return ranked[:max_courts]

print(f"{'seed_top':>8} {'max_courts':>10} {'Macro F1':>9}  (laws-only best: {sweep[BEST_N]:.4f})")
bridge_sweep = {}
for seed_top in [2, 3, 5]:
    for max_courts in [2, 3, 5]:
        preds = {qid: v4_best[qid] | set(bridge_candidates(qid, seed_top, max_courts))
                 for qid in v4_best}
        macro, _ = macro_f1_for(preds)
        bridge_sweep[(seed_top, max_courts)] = macro
        print(f"{seed_top:>8} {max_courts:>10} {macro:>9.4f}")

best_cfg = max(bridge_sweep, key=bridge_sweep.get)
print(f"\nBest bridge config {best_cfg}: {bridge_sweep[best_cfg]:.4f}")

# BGE-specific recall: does the bridge reach ANY of the 102 court-decision gold citations?
bge_gold_hit = 0
for qid in v4_best:
    cands = set(bridge_candidates(qid, *best_cfg))
    court_gold = {g for g in gold_sets_by_qid[qid] if not g.startswith("Art.")}
    hits = cands & court_gold
    if hits:
        bge_gold_hit += len(hits)
        print(f"{qid}: bridge hit {sorted(hits)}")
print(f"Total court-decision gold hits via bridge: {bge_gold_hit} / 102")
```

- [ ] **Step 3: Verification gate (after Push B)**

**Success: `bridge_sweep[best_cfg] > sweep[BEST_N]`** (bridge adds net F1) AND `bge_gold_hit > 0`. If the bridge hits 0 court-gold even from correct seeds (the 3 anchor queries have verified-correct seed articles), that's a **decisive negative**: gold court decisions are not strongly cite-linked to the gold articles, and the bridge approach should be marked dead in session notes — don't iterate on caps.

---

### Task 6: Direct LLM BGE prediction with corpus-existence filter (cheap experiment)

Can the LLM name landmark BGE decisions directly? Raw output will hallucinate, but the courts corpus gives us a free hallucination filter: only keep predicted BGE ids that actually exist as corpus citations. ~10 API calls, one cell.

**Files:**
- Modify: `notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb` (two new cells)

- [ ] **Step 1: Build the valid-BGE lookup (page-level → full corpus citations)**

```python
# Map page-level id ("BGE 137 IV 122") -> all corpus citation strings starting with it
# (corpus/gold citations carry E. parts: "BGE 137 IV 122 E. 6.2")
from collections import defaultdict
COURTS_CSV = DATA_DIR / "court_considerations.csv"

valid_bge = defaultdict(list)
with open(COURTS_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = normalizer.canonicalize(row["citation"])
        if c and c.startswith("BGE"):
            page_id = " ".join(c.split()[:4])      # "BGE 137 IV 122"
            valid_bge[page_id].append(c)
print(f"Distinct BGE page-level ids in corpus: {len(valid_bge):,}")
```

(If a citation_bridge-style cache makes this scan redundant — e.g. the bridge values already enumerate corpus court citations — derive `valid_bge` from that instead of re-scanning 2.4M rows.)

- [ ] **Step 2: Add the BGE prediction + filter + eval cell**

```python
BGE_SYSTEM = """You are a Swiss legal expert. Given a case description, name the leading \
published decisions of the Swiss Federal Supreme Court (BGE) that govern this kind of case.

Output ONE per line, format: BGE <volume> <section> <page>   (e.g. BGE 137 IV 122)
- Only list decisions you are confident actually exist. Do not guess numbers.
- At most 5 decisions. Fewer is better than wrong.
- Output nothing else."""

def predict_bge_v1(query: str) -> list[str]:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": BGE_SYSTEM},
                  {"role": "user", "content": query}],
        temperature=0.0, max_tokens=500,
    )
    text = (resp.choices[0].message.content or "").strip()
    ids = []
    for line in text.splitlines():
        m = re.search(r"BGE\s+\d+\s+[IVX]+[a-b]?\s+\d+", line)
        if m: ids.append(m.group(0))
    return ids

survived_total, hallucinated_total = 0, 0
bge_preds = {}
for q in tqdm(val_queries, desc="BGE predict"):
    raw_ids = predict_bge_v1(q["query"])
    kept = [pid for pid in raw_ids if pid in valid_bge]
    hallucinated_total += len(raw_ids) - len(kept)
    survived_total += len(kept)
    # expand each surviving page id to its single most frequent corpus citation (with E.)
    expanded = []
    for pid in kept:
        forms = valid_bge[pid]
        expanded.append(max(set(forms), key=forms.count))
    bge_preds[q["query_id"]] = expanded

print(f"Survived corpus filter: {survived_total}, hallucinated: {hallucinated_total}")
preds = {qid: v4_best[qid] | set(bge_preds.get(qid, [])) for qid in v4_best}
macro, rows = macro_f1_for(preds)
print(f"v4_best + direct BGE: Macro F1 = {macro:.4f}  (v4_best alone: {sweep[BEST_N]:.4f})")
for r in rows: print(r)
```

- [ ] **Step 3: Push B — push Tasks 5+6 together and run**

```bash
kaggle kernels push -p notebooks/bge_m3_push
kaggle kernels status moeghri/dense-retrieval-bge-m3-laws-eval
kaggle kernels output moeghri/dense-retrieval-bge-m3-laws-eval -p results/kaggle_runs/push_b_bge_channel
```

Note: this run includes the one-time ~30–40 min bridge build → expect ~60 min total.

- [ ] **Step 4: Verify both BGE channels against their gates; pick at most ONE**

Bridge gate (Task 5 Step 3) and direct-BGE gate (**macro improves over `sweep[BEST_N]`**). If both pass, keep the better one — do NOT union both channels (over-prediction lesson from Session 9). If neither passes, the laws-track best from Push A stands, and the BGE gap is documented as requiring a different approach (e.g. court-decision metadata search or fine-tuning — out of scope here).

- [ ] **Step 5: Preserve the bridge cache for future runs**

Download `citation_bridge.json` from the kernel output and upload it into the code dataset so future runs skip the 30-min rebuild. Stage a temp dir that mirrors the existing dataset contents plus `citation_bridge.json` (reuse the Session 6 `kagglehub.dataset_upload` pattern with `ignore_patterns`):

```python
# local, run once:
import kagglehub
kagglehub.dataset_upload(
    "moeghri/omnilex-retrieval-code",
    "<staging-dir-containing-existing-dataset-files-plus-citation_bridge.json>",
    version_notes="add citation_bridge.json cache",
)
```

Do NOT accidentally upload the whole `results/` tree or any data CSVs.

- [ ] **Step 6: Commit**

```bash
git add notebooks/bge_m3_push/ results/kaggle_runs/push_b_bge_channel/
git commit -m "feat: seeded citation bridge + direct BGE prediction experiments (Push B)"
```

---

### Task 7: Record results — scoreboard, session notes, baselines

**Files:**
- Modify: `docs/superpowers/session-notes.md` (append "## Session 10" section)
- Modify: `results/baselines_corrected.json` (add `llm_v4` and winning-config entries)

- [ ] **Step 1: Append Session 10 to session notes**

Follow the exact structure of Sessions 7–9: date, what was run, results tables (determinism check, truncation sweep, val_007 diagnosis, bridge sweep, direct-BGE result), updated scoreboard table including all prior rows, and a "Decisions & Next Steps" list with explicit ✅/⏳/❌ markers. Every number copied from the downloaded Kaggle logs — no recollection from memory.

- [ ] **Step 2: Add the new results to `results/baselines_corrected.json`**

```json
"llm_v4_anchor": {"macro_f1": "<best sweep value>", "best_n": "<BEST_N>",
                   "model": "qwen3-30b-a3b-instruct-2507", "temperature": 0.0,
                   "date": "2026-06-10"},
"llm_v4_plus_bge_channel": {"macro_f1": "<value>", "channel": "<bridge|direct_bge|none>",
                             "config": "<best_cfg or n/a>", "date": "2026-06-10"}
```

(Numeric values as numbers, not strings — placeholders shown quoted only for JSON validity here.)

- [ ] **Step 3: Final commit**

```bash
git add docs/superpowers/session-notes.md results/baselines_corrected.json
git commit -m "docs: Session 10 results - v4 calibration and BGE channel experiments"
```

---

## Verification Summary

| Gate | Where | Pass condition |
|---|---|---|
| Pull integrity | Task 1 Step 2 | `predict_articles`, `FR_TO_DE`, `bridge` all present in pulled notebook |
| Determinism | Task 2 / Push A | 3 identical runs on val_004 at temperature=0 |
| v4 quality | Task 3 / Push A | `sweep[BEST_N] ≥ 0.0926` |
| val_007 | Task 4 / Push A | written hypothesis in session notes (diagnosis, not fix) |
| Bridge | Task 5 / Push B | macro > laws-only best AND ≥1 court-gold hit |
| Direct BGE | Task 6 / Push B | macro > laws-only best after corpus filter |
| Bookkeeping | Task 7 | scoreboard + baselines JSON updated from logs, committed |

**Overall success:** deterministic, calibrated laws pipeline ≥ 0.0926 (expected 0.10–0.13), plus a definitive keep/kill verdict on each of the two BGE channels with numbers recorded. **Stretch:** ≥ 0.12 with a working BGE channel.

## Known Risks

- **Pull may not reflect Session 9's final state** if the last interactive run wasn't saved as a version. Task 1 Step 2 catches this — stop and ask the user, never push over it.
- **Notebook variable names** (`anchors_by_qid`, `gold_sets_by_qid`, `val_queries`) are inferred from session notes; the pulled notebook may use different names. Adapt the new cells to the existing names — never rebuild existing structures.
- **KISSKI rate limits / nondeterminism**: temperature=0 reduces but may not eliminate variance (backend batching). The determinism check measures this honestly.
- **Each push re-runs everything**: keep `RUN_DENSE=False` and the bridge cache in place, or iteration cost explodes.
