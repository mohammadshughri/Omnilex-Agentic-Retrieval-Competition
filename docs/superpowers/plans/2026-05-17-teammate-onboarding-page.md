# Teammate Onboarding HTML Page — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained HTML page that fully onboards a new teammate onto the Omnilex cross-lingual legal retrieval project — covering context, concepts, environment setup, architecture, rules, and reasoning.

**Architecture:** One fully self-contained `docs/onboarding.html` file (no CDN dependencies, no build step) using inline CSS and JS. Sections are linked by an in-page sticky navigation bar. Concepts are explained inline with expandable "Learn More" details blocks.

**Tech Stack:** Vanilla HTML5, inline CSS (CSS variables for theming), minimal vanilla JS for interactivity (smooth scroll, collapsibles, copy-to-clipboard for commands).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `docs/onboarding.html` | The entire onboarding page — single deliverable, no dependencies |

---

## Page Sections (Content Map)

| # | Section | Anchor | Content |
|---|---------|--------|---------|
| 1 | Hero / Overview | `#overview` | Project name, course context, competition link, goal, first milestone, **First Day Checklist**, **Quick Links** |
| 2 | The Problem | `#problem` | Cross-lingual challenge, why BM25 fails, data table |
| 3 | Key Concepts | `#concepts` | BM25, Dense embeddings, BGE-M3, FAISS, RRF, HyDE, Macro F1, Reranker + **external reading link** |
| 4 | Architecture | `#architecture` | Pipeline diagram, why two indices, v1→v2 evolution, model pluggability, **Top 3 Gotchas** |
| 5 | Environment Setup | `#setup` | uv setup, data download, index build, common errors |
| 6 | Project Structure | `#structure` | Annotated file tree with "★ NEW" markers |
| 7 | Roadmap | `#roadmap` | **Horizontal CSS timeline** + week-by-week table with F1 estimates, May 21 presentation outline |
| 8 | Team & Tasks | `#team` | Person A/B/C assignments, dependencies, outputs |
| 9 | Rules | `#rules` | Package mgmt, eval discipline, data rules, artifact safety, citation normalization |
| 10 | Glossary | `#glossary` | Alphabetical quick-reference table |

---

## Design Spec

- **Color scheme:** Dark navy (`#0d1117`) background, white text, teal accent (`#58a6ff`) — GitHub dark palette
- **Font:** System font stack; monospace for code
- **Nav:** Sticky top bar; active section highlighted on scroll via `IntersectionObserver`
- **Concept cards:** `<details>`/`<summary>` collapsibles — summary = one-line definition, expanded = full explanation
- **Code blocks:** Monospace, dark bg, copy-to-clipboard button per block
- **Pipeline diagram:** ASCII art in styled `<pre>` with colored `<span>` annotations
- **Responsive:** 900px max-width content column, readable at desktop widths

---

## Task 1: HTML Skeleton + Navigation

**Files:**
- Create: `docs/onboarding.html`

- [ ] **Step 1: Create the HTML shell with all 10 section placeholders and sticky nav**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Omnilex Project — Teammate Onboarding</title>
  <style>/* Task 2 */</style>
</head>
<body>
  <nav id="main-nav">
    <a href="#overview">Overview</a>
    <a href="#problem">The Problem</a>
    <a href="#concepts">Concepts</a>
    <a href="#architecture">Architecture</a>
    <a href="#setup">Setup</a>
    <a href="#structure">Structure</a>
    <a href="#roadmap">Roadmap</a>
    <a href="#team">Team</a>
    <a href="#rules">Rules</a>
    <a href="#glossary">Glossary</a>
    <a href="#references">References</a>
  </nav>
  <main>
    <section id="overview"></section>
    <section id="problem"></section>
    <section id="concepts"></section>
    <section id="architecture"></section>
    <section id="setup"></section>
    <section id="structure"></section>
    <section id="roadmap"></section>
    <section id="team"></section>
    <section id="rules"></section>
    <section id="glossary"></section>
    <section id="references"></section>
  </main>
  <script>/* Task 13 */</script>
</body>
</html>
```

- [ ] **Step 2: Open in browser — verify page loads (blank sections expected)**

---

## Task 2: CSS Styling

**Files:**
- Modify: `docs/onboarding.html` — replace `<style>/* Task 2 */</style>`

- [ ] **Step 1: Add all CSS — variables, reset, nav, sections, cards, code blocks, tables, badges**

Key rules:
```css
:root {
  --bg: #0d1117; --bg-card: #161b22; --bg-code: #1e2430;
  --border: #30363d; --text: #e6edf3; --text-muted: #8b949e;
  --accent: #58a6ff; --accent-green: #3fb950;
  --accent-orange: #d29922; --accent-red: #f85149;
  --radius: 6px; --nav-height: 56px;
}
/* sticky nav, section max-width:900px, card, details/summary,
   code-block with copy-btn, table zebra stripes, badge, callout */
```

- [ ] **Step 2: Open in browser — dark theme renders, nav is visible at top**

---

## Task 3: Section 1 — Hero / Overview

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="overview">`

- [ ] **Step 1: Write content**

Include:
- Title: "Omnilex Legal Retrieval — Project Onboarding"
- Subtitle: "LUH Agentic AI Course · Semester 6 · Apr 30 – Jul 16, 2026"
- One-sentence goal: "Given an English legal query, retrieve the correct set of Swiss law citations from two corpora (176K federal law articles + 2.4M court decisions)."
- Team: 3 people
- First milestone: May 21, 2026 — progress presentation

**First Day Checklist** (ordered `<ol>` with checkboxes, prominently placed at top):
1. Clone the repo (see Setup section)
2. Run `uv venv .venv && uv pip install -r requirements.txt && uv pip install -e .`
3. Run `pytest` — all tests should pass
4. Read **The Problem** section (Section 2)
5. Read **Key Concepts** section (Section 3) — especially BM25, Dense, and RRF
6. Pick a Week 1-2 task from **Team & Tasks** (Section 8)
7. Read the **Rules** section before writing any code

**Quick Links card** (`.card` with icon links):
- Kaggle competition page
- Implementation strategy doc (`docs/superpowers/specs/2026-05-17-implementation-strategy-design.md`)
- Architecture review doc (`docs/superpowers/specs/2026-05-17-architecture-review-findings-for-claude.md`)
- Further reading: Dense vs. Sparse vs. Hybrid — RRF article (Medium)

---

## Task 4: Section 2 — The Problem

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="problem">`

- [ ] **Step 1: Write cross-lingual gap explanation**

Plain-language explanation: Train queries are German; val/test queries are English; corpus is German/French/Italian. BM25 scores documents by word overlap — "detention" has zero overlap with "Haft". We need semantic retrieval that maps across languages.

- [ ] **Step 2: Add data files table**

| File | Language | Rows | Role |
|------|----------|------|------|
| `train.csv` | German queries | 1,139 | Gold labels — but DIFFERENT language from test! Cannot use for cross-lingual tuning |
| `val.csv` | English queries | 10 | Our only English eval signal — treat as holdout |
| `test.csv` | English queries | 40 | Kaggle test set (no labels) |
| `laws_de.csv` | 99% German | 175,933 | Federal law articles, ~242 chars each |
| `court_considerations.csv` | 61% DE / 32% FR / 6% IT | ~2.4M | Court decisions, ~1,105 chars each |

- [ ] **Step 3: Add "Why BM25 fails" one-liner callout**

> "English 'detention' ≠ German 'Haft' — no shared tokens, BM25 score = 0 for conceptual queries."

- [ ] **Step 4: Add citation count stats** (mean 4.1, median 2, max 44 — explains why dynamic K matters)

---

## Task 5: Section 3 — Key Concepts

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="concepts">`

- [ ] **Step 1: Write 8 concept cards as `<details>` collapsibles**

Each card structure:
```html
<div class="card">
  <details>
    <summary>BM25 — keyword ranking (what we replaced)</summary>
    <p>...</p>
  </details>
</div>
```

Concepts and their key points:

**BM25:** TF-IDF-based keyword matching. Fails cross-lingually: zero token overlap between English queries and German corpus. Still used as one retrieval channel for queries that explicitly mention article numbers.

**Dense Embeddings:** Transformer encodes text to a fixed-size vector. Semantically similar texts land near each other in vector space regardless of language. Requires pre-building an index (FAISS).

**BGE-M3:** BAAI/bge-m3, 0.6B params. Produces three retrieval signals from one model: dense (1024-dim), sparse (neural lexical weights), ColBERT (multi-vector). Cross-lingual by design. Fits 8GB GPU. First model we use; Qwen3-8B planned for Week 5-6.

**FAISS:** Facebook AI Similarity Search. Indexes dense vectors for fast approximate nearest-neighbor queries. We use `IndexHNSWFlat` (M=32): ~98% recall, no training needed. Laws index ~500MB, Courts index ~7-8GB.

**RRF — Reciprocal Rank Fusion:** Combines ranked lists from multiple channels. Formula: `score = Σ 1/(60 + rank)`. Why not weighted scores: dense, sparse, and BM25 scores are incompatible scales — you cannot add them directly. RRF only uses rank positions, which are always comparable.

**HyDE (Week 4):** Instead of embedding the English query, use an LLM to generate a hypothetical German law passage that would answer it, then embed that. Bridges the cross-lingual gap without explicit translation. Deferred until retrieval baseline is stable.

**Macro F1:** Competition metric. Average F1 (precision × recall harmonic mean) across all queries. Each query has equal weight regardless of citation count. Implication: predicting the right *number* of citations matters as much as which ones — hence dynamic K.

**Reranker (Week 3):** Cross-encoder model that scores (query, document) pairs. First-stage retrieval (FAISS) optimizes recall; reranker optimizes precision over top-100 candidates. We use `bge-reranker-v2-m3`, then Qwen3-Reranker in Week 5-6.

- [ ] **Step 2: Add "Further Reading" banner below all concept cards**

After the last concept card, add a highlighted callout linking to the external article:

```html
<div class="callout" style="border-color: var(--accent);">
  <strong>📖 Further Reading:</strong>
  <a href="https://medium.com/@robertdennyson/dense-vs-sparse-vs-hybrid-rrf-which-rag-technique-actually-works-1228c0ae3f69"
     target="_blank" rel="noopener">
    Dense vs. Sparse vs. Hybrid + RRF — Which RAG Technique Actually Works?
  </a>
  <p style="margin-top:8px; color: var(--text-muted); font-size:0.9rem;">
    Practical comparison of the three retrieval modes and RRF fusion with real benchmark results.
    Read this after the concept cards above to see how these techniques compare in practice.
  </p>
</div>
```

- [ ] **Step 3: Verify all 8 cards expand/collapse in browser and the further reading link opens correctly**

---

## Task 6: Section 4 — Architecture

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="architecture">`

- [ ] **Step 1: Write ASCII pipeline diagram in styled `<pre>` block**

```
English Legal Query
    │
    ├──► Citation Anchor Extractor    ← regex: "Art. 221 StPO" found literally in query
    │    └──► CitationNormalizer       ← canonical form (existing module)
    │         └──► High-priority candidates (injected at rank 0 in fusion)
    │
    ├──► BGE-M3 Encoder (0.6B params, BAAI/bge-m3)
    │    ├── Dense embeddings (1024-dim)
    │    ├── Sparse neural weights (cross-lingual lexical)
    │    └── ColBERT tokens          ← laws only (175K); too large for 2.4M courts
    │
    ▼
 FAISS Index (pre-built, loaded from disk)
    ├── Laws Index   (175K passages)    ~500 MB
    └── Courts Index (2.4M passages)   ~7-8 GB  [dense+sparse only]
    │
    ▼
 Top-100 candidates per channel
    │
    ▼
 RRF Fusion (per corpus)
 score = Σ 1/(60 + rank)              ← rank-based, no score-scale issues
    │
    ▼
 Federated Law + Court Merge
 deduplicate by normalized citation
    │
    ▼
 Reranker: bge-reranker-v2-m3         ← Week 3+
    │
    ▼
 Dynamic K / Score Threshold          ← test k=2,3,5,10 vs threshold on val
    │
    ▼
 CitationNormalizer → submission.csv
```

- [ ] **Step 2: Write "Why two separate indices" explanation**

Laws (175K, short, German) vs. Courts (2.4M, long, trilingual). Different embedding time (~20 min vs. 4-8 hrs). Train gold: 70% laws / 30% courts — separate weights. Failure isolation on cluster.

- [ ] **Step 3: Write "Architecture Evolution (v1 → v2)" callout**

The first design proposed weighted score fusion: `0.4×dense + 0.2×sparse + 0.4×ColBERT`. An architecture review rejected this: dense, sparse, BM25, law, and court scores are not on the same scale. **RRF was adopted instead.** Do not reintroduce score weighting without calibration proving it beats RRF.

- [ ] **Step 4: Write "Model Pluggability" subsection**

`EmbeddingModel` protocol with `encode_documents()` / `encode_queries()`. `BgeM3Embedder` is the first adapter. Qwen3-Embedding-8B slots in during Week 5-6 without touching retrieval or fusion code. Same pattern for rerankers: `RerankerModel` protocol → `BgeCrossEncoderReranker`, then `Qwen3CausalLmReranker`. (Note: Qwen3-Reranker must be loaded as a causal LM and scored via yes/no logits — not a standard CrossEncoder.)

- [ ] **Step 5: Add "Top 3 Gotchas" warning box**

Prominent `.callout` with orange border, immediately after the architecture diagram:

```html
<div class="callout">
  <strong>⚠️ Top 3 Silent Failure Modes</strong>
  <ol style="margin-top: 10px; padding-left: 20px;">
    <li>
      <strong>Weighted scores instead of RRF.</strong>
      Dense, sparse, BM25, law, and court scores are on incompatible scales.
      Adding them directly produces nonsense rankings. Always use RRF.
    </li>
    <li>
      <strong>Comparing raw citation strings without CitationNormalizer.</strong>
      "Art. 1 Abs. 1 ZGB" and "Art. 1 ZGB" are the same citation but won't match as strings.
      Always normalize before comparing. F1 will silently underreport if you skip this.
    </li>
    <li>
      <strong>Tuning hyperparameters against val.csv more than once.</strong>
      val.csv has only 10 English queries. Repeated tuning against it is effectively
      memorization, not generalization. Use it as a final sanity check, not an
      optimization target.
    </li>
  </ol>
</div>
```

---

## Task 7: Section 5 — Environment Setup

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="setup">`

- [ ] **Step 1: Write step-by-step setup with copy buttons**

Each command wrapped in `.code-block` with `.copy-btn`:

```bash
# 1. Clone
git clone https://github.com/Omnilex-AI/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# 2. Create venv with uv (NOT pip — project rule)
uv venv .venv

# 3. Activate
source .venv/bin/activate        # Linux/macOS/WSL
# .venv\Scripts\activate         # Windows PowerShell

# 4. Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# 5. Install package in dev mode
uv pip install -e .

# 6. Download data (from Kaggle — place in data/raw/lexam/ and data/raw/swiss_citations/)
python utils/download_data.py

# 7. Build BM25 indices
python utils/build_indices.py

# 8. Verify — run tests
pytest
```

- [ ] **Step 2: Add "Why uv?" note**

`uv` is faster than pip, produces reproducible lockfiles, and is the project convention (see Rules section). Always use `uv pip install`, never plain `pip install`.

- [ ] **Step 3: Add Common Errors table**

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: omnilex` | Run `uv pip install -e .` from repo root |
| `FileNotFoundError: laws_de.csv` | Download from Kaggle → `data/raw/swiss_citations/` |
| `pickle.UnpicklingError` | Rebuild indices: `python utils/build_indices.py` |
| `CUDA out of memory` | Reduce batch size in embed script; add `--device cpu` flag |
| Tests fail with `CitationNormalizer` errors | Verify `data/abbrev-translations.json` exists |

---

## Task 8: Section 6 — Project Structure

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="structure">`

- [ ] **Step 1: Write annotated file tree**

```
Omnilex-Agentic-Retrieval-Competition/
├── src/omnilex/
│   ├── citations/
│   │   ├── normalizer.py      ★ GOD NODE — called everywhere; Art. + BGE parser
│   │   ├── abbreviations.py   — 4,362 Swiss law abbreviations (DE/FR/IT)
│   │   └── types.py           — Citation dataclass, CitationType enum
│   ├── evaluation/
│   │   ├── metrics.py         — macro_f1, micro_f1, MAP, NDCG@k
│   │   └── scorer.py          — load CSVs, normalize, compute metrics
│   ├── retrieval/
│   │   ├── bm25_index.py      — existing BM25 baseline (rank-bm25 wrapper)
│   │   ├── tools.py           — LLM-compatible tool wrappers (ReAct agent)
│   │   ├── models.py          ★ NEW — EmbeddingModel / RerankerModel protocols
│   │   ├── dense_index.py     ★ NEW — FAISS index builder (model-agnostic)
│   │   ├── dense_retriever.py ★ NEW — query encoding + FAISS search
│   │   ├── fusion.py          ★ NEW — RRF fusion + deduplication
│   │   ├── anchor_extractor.py★ NEW — citation-anchor extraction from query
│   │   └── submission.py      ★ NEW — top-k → normalize → CSV
│   └── llm/
│       ├── loader.py          — llama-cpp-python wrapper + GPU auto-detect
│       └── prompts.py         — ReAct agent prompt templates
├── scripts/
│   ├── embed_corpus.py        ★ NEW — SLURM batch embedding with checkpoints
│   └── run_evaluation.py      ★ NEW — comparison evaluation harness
├── notebooks/
│   ├── 01_direct_generation_baseline.ipynb
│   └── 02_agentic_retrieval_baseline.ipynb
├── data/
│   ├── raw/lexam/             — train.csv, val.csv, test.csv (gitignored)
│   ├── raw/swiss_citations/   — laws_de.csv, court_considerations.csv (gitignored)
│   └── processed/bge_m3/      ★ NEW — embeddings + FAISS indices (~15GB, gitignored)
│       ├── laws_dense.npy / laws_faiss.index / laws_metadata.jsonl
│       └── courts_dense.npy / courts_faiss.index / courts_metadata.jsonl
├── results/                   — JSON evaluation outputs (bm25_baseline.json, etc.)
└── tests/                     — pytest test suite
```

---

## Task 9: Section 7 — Roadmap

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="roadmap">`

- [ ] **Step 1: Build horizontal CSS timeline**

A visual lane above the table showing the 8 weeks as connected nodes with milestone labels. Pure CSS — no JS needed.

```css
.timeline {
  display: flex; align-items: flex-start; gap: 0;
  margin: 24px 0; overflow-x: auto; padding-bottom: 8px;
}
.timeline-item {
  flex: 1; min-width: 100px; text-align: center; position: relative;
}
.timeline-item::before {
  content: ''; position: absolute; top: 14px; left: 50%; right: -50%;
  height: 2px; background: var(--border); z-index: 0;
}
.timeline-item:last-child::before { display: none; }
.timeline-dot {
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--bg-card); border: 2px solid var(--accent);
  margin: 0 auto 8px; display: flex; align-items: center;
  justify-content: center; font-size: 0.7rem; font-weight: 700;
  color: var(--accent); position: relative; z-index: 1;
}
.timeline-dot.active { background: var(--accent); color: var(--bg); }
.timeline-label { font-size: 0.72rem; color: var(--text-muted); line-height: 1.3; }
.timeline-date { font-size: 0.65rem; color: var(--text-muted); margin-top: 3px; }
```

Milestones: W1-3 (current, active dot), W3 reranker, W4 HyDE, W5-6 Qwen3, W7 Calibration, W8 Final.

- [ ] **Step 2: Write week-by-week roadmap table below the timeline**

| Week | Dates | Goal | Key Deliverable | Est. Macro F1 |
|------|-------|------|-----------------|---------------|
| 1-3 | Apr 30 – May 21 | BGE-M3 hybrid retrieval | BM25 vs BGE-M3 F1 comparison | 0.15–0.25 |
| 3 | May 22 – May 28 | Add reranker | Ablation: dense vs hybrid vs hybrid+rerank | +0.03–0.05 |
| 4 | May 29 – Jun 4 | HyDE | With/without HyDE on conceptual queries | +0.02–0.05 |
| 5-6 | Jun 5 – Jun 18 | Qwen3-8B on cluster | BGE-M3 vs Qwen3 head-to-head | 0.20–0.30 |
| 7 | Jun 19 – Jun 25 | Dynamic K + query routing | Calibrated threshold; lexical vs. conceptual routing | varies |
| 8 | Jun 26 – Jul 16 | Final polish + ablation | Full ablation table + pipeline diagram | target: 0.30+ |

BM25 baseline (English queries, no translation): ~0.02

- [ ] **Step 2: Add May 21 presentation outline as subsection**

1. The problem: English queries vs German/French/Italian corpus (1 slide)
2. Why BM25 fails cross-lingually (1 slide)
3. Our approach: BGE-M3 hybrid retrieval (2 slides)
4. Results: BM25 vs BGE-M3 comparison table (1-2 slides)
5. Example: one query end-to-end walkthrough (1 slide)
6. Roadmap: reranker → HyDE → Qwen3 → calibration (1 slide)

---

## Task 10: Section 8 — Team & Task Division

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="team">`

- [ ] **Step 1: Write Week 1-2 task division table**

| Person | Task | Depends On | Output | Est. Time |
|--------|------|------------|--------|-----------|
| A | Embed `laws_de.csv` + build FAISS laws index | BGE-M3 downloaded | `laws_dense.npy`, `laws_faiss.index`, `laws_metadata.jsonl` | ~20 min |
| B | Embed `court_considerations.csv` on SLURM | BGE-M3 + SLURM account | `courts_dense.npy`, `courts_faiss.index`, `courts_metadata.jsonl` | 4-8 hrs |
| C | Build query pipeline + evaluation harness | Person A's laws index | `dense_retriever.py`, `run_evaluation.py`, first val F1 result | 1-2 days |

Person A finishes first → helps Person C test the laws-only pipeline while Person B's cluster job runs.

---

## Task 11: Section 9 — Rules & Conventions

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="rules">`

- [ ] **Step 1: Write rules in categorized subsections**

**Package Management**
- `uv pip install <pkg>` — never plain `pip install`. Reproducible lockfiles, faster.
- `uv pip install -e .` to install the local `omnilex` package in dev mode.

**Running Code**
- Activate venv before anything: `source .venv/bin/activate`
- Run tests before pushing: `pytest`
- Lint before pushing: `ruff check . && ruff format .`

**Evaluation Discipline**
- `val.csv` has only 10 queries — do NOT tune hyperparameters against it repeatedly (high overfitting risk)
- Report `val.csv` scores as "calibration check", not strong evidence
- Use train-derived diagnostics for iteration; label them as German-query diagnostics (different from test language)

**Data Rules**
- Never commit raw data (all data paths are gitignored)
- `data/raw/` is read-only — never modify source files
- Processed files (embeddings, indices) go in `data/processed/` (also gitignored, ~15GB)

**Artifact Safety**
- Do NOT use pickle for shared or uploaded artifacts — use `.npy`, FAISS native format, or JSON
- The existing `laws_index.pkl` / `courts_index.pkl` are acceptable for local BM25 prototype only
- Pickle is unsafe for untrusted artifacts and slow at 2.4M documents

**Citation Normalization**
- Always pass raw citation strings through `CitationNormalizer` before comparing or submitting
- Never compare raw strings: `"Art. 1 Abs. 1 ZGB"` normalizes to the same form as `"Art. 1 ZGB"`
- `CitationNormalizer` is the single canonical parser — do not write alternative regex

**Before Large Indexing Jobs (CRITICAL)**
- Run acceptance tests on a 100-row sample FIRST
- Verify: CSV row id → FAISS id alignment, metadata lookup by id, citation normalization, RRF determinism, submission validates
- Only then launch the 4-8 hour court embedding job on SLURM — a bad index wastes GPU hours

**Do Not Reintroduce**
- Raw weighted score fusion (use RRF instead — scores are not cross-scale comparable)
- Hard-coded BGE-M3 paths (use `EmbeddingModel` protocol so Qwen3 is a swap-in)

---

## Task 12: Section 10 — Glossary

**Files:**
- Modify: `docs/onboarding.html` — fill `<section id="glossary">`

- [ ] **Step 1: Write alphabetical quick-reference table**

| Term | Definition |
|------|-----------|
| Art. | "Artikel" — prefix for Swiss federal law citations (e.g., `Art. 221 Abs. 1 StPO`) |
| BGE | "Bundesgerichtsentscheid" — Swiss Federal Court leading decision (e.g., `BGE 116 Ia 56`) |
| BM25 | TF-IDF-based keyword ranking; fails cross-lingually |
| CitationNormalizer | Canonical citation parser at `src/omnilex/citations/normalizer.py` |
| ColBERT | Multi-vector retrieval: one vector per token per passage; precise but expensive |
| Dense retrieval | Embedding-based semantic search via FAISS |
| Dynamic K | Predicting how many citations to return per query (vs. fixed top-k) |
| FAISS | Facebook AI Similarity Search — fast approximate nearest-neighbor |
| FlagEmbedding | Python library for BGE-M3 (dense + sparse + ColBERT) |
| HyDE | Hypothetical Document Embeddings — generate a synthetic passage, embed that |
| IndexHNSWFlat | FAISS index type: Hierarchical Navigable Small World, ~98% recall |
| Macro F1 | Competition metric: average F1 per query (all queries equal weight) |
| RRF | Reciprocal Rank Fusion — rank-based fusion, no score-scale issues |
| SLURM | HPC job scheduler on GWDG/KISSKI cluster for long GPU jobs |
| Sparse retrieval | BGE-M3 neural sparse weights — cross-lingual lexical matching |
| uv | Fast Python package manager used instead of pip |

---

## Task 13: JavaScript Interactivity

**Files:**
- Modify: `docs/onboarding.html` — add inline `<script>` before `</body>`

- [ ] **Step 1: Copy-to-clipboard for code blocks**

```javascript
document.querySelectorAll('.code-block').forEach(block => {
  const btn = block.querySelector('.copy-btn');
  const pre = block.querySelector('pre');
  btn.addEventListener('click', () => {
    navigator.clipboard.writeText(pre.textContent.trim());
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
  });
});
```

- [ ] **Step 2: Active nav highlight on scroll (IntersectionObserver)**

```javascript
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('#main-nav a');
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(a => a.classList.remove('active'));
      const active = document.querySelector(
        `#main-nav a[href="#${entry.target.id}"]`
      );
      if (active) active.classList.add('active');
    }
  });
}, { rootMargin: '-20% 0px -70% 0px' });
sections.forEach(s => observer.observe(s));
```

- [ ] **Step 3: Verify in browser — scroll updates nav; copy buttons work**

---

## Task 14: Section 11 — References

**Files:**
- Modify: `docs/onboarding.html` — add `<section id="references">` and nav link

- [ ] **Step 1: Write references section with annotated links**

A curated reading list with a one-sentence description of what each resource teaches and why it's relevant to this project:

| Resource | What it teaches | Why relevant |
|----------|----------------|--------------|
| [Dense vs. Sparse vs. Hybrid + RRF — Which RAG Technique Actually Works?](https://medium.com/@robertdennyson/dense-vs-sparse-vs-hybrid-rrf-which-rag-technique-actually-works-1228c0ae3f69) | Practical benchmark comparison of dense, sparse, and hybrid retrieval modes with RRF fusion | Directly maps to our retrieval architecture choices — read this to understand *why* we use hybrid + RRF instead of dense-only |
| [BGE-M3 Paper (BAAI)](https://arxiv.org/abs/2309.07597) | BGE-M3's multi-functionality: dense + sparse + ColBERT from one model | Explains the three retrieval channels we use in Week 1-2 |
| [FAISS Documentation](https://faiss.ai/) | FAISS index types, memory tradeoffs, approximate vs. exact search | Reference for choosing IndexHNSWFlat vs. IVFPQ when memory budget is tight |
| [Kaggle Competition](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval) | Competition rules, evaluation metric, leaderboard | The source of truth for submission format and Macro F1 scoring |
| Implementation Strategy Doc (`docs/superpowers/specs/2026-05-17-implementation-strategy-design.md`) | Full week-by-week architecture spec | Go here for the detailed rationale behind every component decision |
| Architecture Review Doc (`docs/superpowers/specs/2026-05-17-architecture-review-findings-for-claude.md`) | Critical review of v1 — what was wrong and why | Go here to understand why RRF replaced weighted scores and why reranking moved earlier |

- [ ] **Step 2: Verify links open correctly in browser**

---

## Task 15: Review & Polish

- [ ] **Step 1: Read through as a new teammate would — check each section answers its purpose**
  - Section 1: Is the First Day Checklist the first thing a newcomer sees and acts on?
  - Section 1: Does the Quick Links card include all four key resources?
  - Section 2: Does it clearly explain WHY BM25 fails?
  - Section 3: Is RRF explained without assuming prior knowledge?
  - Section 3: Does the further reading link (Medium article) appear after the concept cards?
  - Section 4: Does the v1→v2 evolution callout prevent the weighted-score mistake?
  - Section 4: Are the Top 3 Gotchas visually prominent (orange border callout)?
  - Section 5: Can someone go from zero to running `pytest` following only this page?
  - Section 7: Does the horizontal timeline render and show the active week?
  - Section 9: Is the "no pickle for shared artifacts" rule prominently stated?
  - Section 11: Do all reference links open correctly?

- [ ] **Step 2: Verify all code commands match CLAUDE.md (`uv`, `pytest`, `ruff`)**

- [ ] **Step 3: Open in browser, open DevTools console — zero errors**

- [ ] **Step 4: Confirm fully self-contained** — no `<link href="https://...">`, no `<script src="https://...">` (external links in Section 11 open via `target="_blank"` — intentional)

---

## Acceptance Criteria

- [ ] `docs/onboarding.html` opens directly in browser with no external dependencies
- [ ] All 11 nav links (including References) scroll to correct sections
- [ ] Active nav link updates on scroll
- [ ] All code blocks have working copy buttons
- [ ] All concept cards expand/collapse
- [ ] First Day Checklist is visible before scrolling on the Overview section
- [ ] Quick Links card in hero links to Kaggle, strategy doc, review doc, and Medium article
- [ ] Top 3 Gotchas callout (orange border) is present in the Architecture section
- [ ] Horizontal CSS timeline renders above the roadmap table
- [ ] Medium article link appears after concept cards in Section 3
- [ ] References section (Section 11) lists all 6 resources with descriptions
- [ ] Section 5 (Setup) is sufficient to go from clone to `pytest` passing
- [ ] Pipeline diagram matches current implementation strategy
- [ ] No console errors
