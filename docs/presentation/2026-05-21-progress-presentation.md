---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 28px;
  }
  h1 { color: #1a3a5c; font-size: 42px; }
  h2 { color: #1a3a5c; font-size: 34px; }
  table { font-size: 22px; width: 100%; }
  th { background: #1a3a5c; color: white; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
---

<!-- Slide 1: Title -->
# Cross-Lingual Legal Citation Retrieval

### Progress Presentation — Week 3

**Team:** Mohammad Shughri et al. · LUH Agentic AI Course
**Competition:** Omnilex — LLM Agentic Legal Information Retrieval
**Date:** May 22, 2026

---

<!-- Slide 2: The Problem -->
## The Problem

**Task:** Given an English legal query, retrieve the correct Swiss law citations

- Two corpora to search:
  - **Federal laws** — 175,933 passages, ~99% German, short (`Art. X BOOK` format)
  - **Court decisions** — 2.4M passages, 61% DE / 32% FR / 6% IT (`BGE` format)
- **10 validation queries** (English), **train queries are German**
- Gold citations: average 4.1 per query (median 2, max 44)
- Metric: **Macro F1** — every query weighted equally

> **The challenge:** English query tokens don't appear in German/French/Italian documents

---

<!-- Slide 3: Why BM25 Fails -->
## Why BM25 Fails Cross-Lingually

BM25 is a keyword-overlap algorithm — it needs **shared tokens** between query and document.

| English query term | German corpus term | BM25 match? |
|--------------------|--------------------|-------------|
| "detention" | "Haft" | no overlap |
| "contract formation" | "Vertragsabschluss" | no overlap |
| "Art. 221 StPO" | "Art. 221 StPO" | exact match only |

**Result:** BM25 only retrieves citations **literally spelled out** in the query.
Conceptual queries → near-zero F1.

**BM25 Macro F1 on val set (measured): 0.00** — zero for all k values (2, 3, 5, 10)
*(Index built on body text only. Even queries containing explicit article references score zero because article numbers appear in citation metadata, not in the German text body.)*

---

<!-- Slide 4: Our Approach — BGE-M3 Model -->
## Our Approach: BGE-M3 Hybrid Retrieval (1/2)

**Model:** `BAAI/bge-m3` — 0.6B parameters, trained on 100+ languages

Three complementary retrieval signals from a **single encoder**:

| Signal | How it works | Cross-lingual? |
|--------|--------------|----------------|
| **Dense** (1024-dim) | Semantic vector similarity | Yes — "detention" ≈ "Haft" |
| **Sparse** (neural weights) | Learned token importance | Partial |
| **ColBERT** (per-token vectors) | Token-level late interaction | Yes |

**Why BGE-M3:**
- Trained on diverse multilingual text (100+ languages)
- 0.6B params fits any GPU (T4 on Kaggle free tier)
- Embeds 175K laws in ~5 min, 2.4M court decisions in ~4–8 hours

---

<!-- Slide 5: Our Approach — Pipeline -->
## Our Approach: BGE-M3 Hybrid Retrieval (2/2)

```
English Legal Query
    │
    ├─► Citation-anchor extractor  ──► "Art. 221 StPO" in query → direct candidate
    │
    ▼
BGE-M3 Encoder  (3 independent channels per index)
    ├─► Laws FAISS index (175K)    ──► dense top-100 + sparse top-100 + ColBERT top-100
    └─► Courts FAISS index (2.4M)  ──► dense top-100 + sparse top-100
    │
    ▼
Reciprocal Rank Fusion per corpus  score = Σ 1/(60 + rank)
    │
    ▼
Deduplicate → top-k → CitationNormalizer → submission.csv → Macro F1
```

**Code:** fully implemented · 34 tests pass · deployed to Kaggle · evaluation in progress

---

<!-- Slide 6: Results -->
## Results

**Baseline confirmed:** BM25 Macro F1 = 0.00 — the cross-lingual token gap is real and measured

| System | Macro F1 | Status |
|--------|----------|--------|
| BM25 (body text, all k) | **0.00** | Measured on val set |
| BGE-M3 dense — laws only | *evaluation in progress* | Kaggle GPU run scheduled |
| BGE-M3 full hybrid (laws + courts) | *evaluation in progress* | Courts embedding running |

**Expected range (BGE-M3 multilingual benchmarks):** 0.10–0.25 for laws-only

> n=10 val queries — directionally meaningful, not statistically robust.
> One outlier query (44 gold citations) can shift Macro F1 by ±0.05+.
> BM25 note: body-text-only index; citation-aware BM25 would score >0 on queries with explicit article numbers.

---

<!-- Slide 7: Query Walkthrough -->
## Example: Query End-to-End (val_001)

**Query snippet:** *"...three-month extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion)..."*

**Step 1 — Citation anchor extraction:**
`Art. 221 StPO` found in query text → injected as priority candidate

**Step 2 — Expected BGE-M3 dense retrieval (laws, top-3) — evaluation pending:**
Art. 221 StPO, Art. 227 StPO, Art. 212 StPO
(model: "pre-trial detention" ≈ "Untersuchungshaft" in embedding space)

**Step 3 — Gold citations include:**
Art. 221 Abs. 1 StPO, Art. 227 Abs. 1 StPO, Art. 212 Abs. 3 StPO, BGE 137 IV 122 ...

**Takeaway:** Dense retrieval finds conceptually matching laws without token overlap.
BM25 would only return Art. 221 (exact string in query).

---

<!-- Slide 8: Roadmap -->
## Roadmap

| Week | Dates | Addition | Est. Macro F1 |
|------|-------|----------|---------------|
| **1–3** (done) | Apr 30 – May 21 | BGE-M3 hybrid retrieval | 0.15–0.25 (estimated) |
| **3** | May 22–28 | `bge-reranker-v2-m3` reranker | +0.03–0.05 |
| **4** | May 29 – Jun 4 | HyDE: synthetic German passage | +0.02–0.05 |
| **5–6** | Jun 5–18 | Qwen3-Embedding-8B on SLURM | 0.20–0.30 |
| **7** | Jun 19–25 | Citation count calibration | +0.01–0.03 |
| **8** | Jun 26 – Jul 16 | Ablation, final submission | — |

**Next action:** Submit BGE-M3 hybrid to Kaggle leaderboard this week
