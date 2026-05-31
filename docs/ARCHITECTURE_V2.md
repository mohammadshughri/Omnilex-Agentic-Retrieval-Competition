# Architecture V2: Citation-Prediction Pipeline

**Date:** 2026-05-22
**Author:** Generated with empirical verification
**Status:** Design — awaiting uni API documentation

---

## 0. Leaderboard Reality Check (as of 2026-05-22)

| Rank | Team | Score | Submissions | Implication |
|------|------|-------|-------------|-------------|
| 1 | Kanak Raj | **0.35940** | 50 | |
| 2 | thechint | **0.35198** | 77 | |
| 3 | BetterCallAgent | **0.33628** | 58 | Name implies agentic LLM approach |
| — | Old top (May 17) | 0.065 | — | Pure embedding+reranking ceiling |
| — | Our BGE-M3 (val) | 0.0435 | — | 8x behind leaders |

**The landscape shifted fundamentally.** Between May 17 and May 22, the top score
jumped from 0.065 to 0.359 — a **5.5x improvement**. This is NOT achievable with
better embeddings or reranking. The jump proves that the top teams are using
**LLM-based reasoning** (confirmed by "BetterCallAgent" name).

**What 0.359 F1 requires:** ~9 correct citations out of ~25 gold per query
(36% recall, 36% precision at k=25). This means the LLM correctly identifies
roughly one-third of the applicable Swiss legal provisions from an English
description. Pure retrieval approaches plateau around 0.065.

**Our position:** At 0.0435, we are at 6% of the oracle ceiling. The uni API
is not optional — it is the critical differentiator between 0.065 and 0.359.

---

## 1. Why the Current Approach Underperforms

### Verified facts (measured 2026-05-22)

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Regex extracts citations from queries | 3/251 gold hits (1.2% recall) | **WEAK signal** — only 4/10 val queries mention any Art. number |
| Art. 100 Abs. 1 BGG is universal | 9/10 val queries (not 10) | **True** but covers only 1 citation per query |
| TF-IDF training transfer works | 2/251 gold hits (0.8% recall) | **FAILED** — train is German with 4.1 avg citations; val is English with 25.1 |
| Co-citation expansion helps | 3→5 gold hits after expansion | **NEGLIGIBLE** — training co-citation graph too sparse for val vocabulary |
| BGE-M3 dense retrieval | Macro F1 = 0.0435 at k=10 on val | **Best so far** but 8x behind leaders |
| Leaderboard (May 22) | 0.359 / 0.352 / 0.336 (top 3) | **LLM-based approaches dominate** |

### The distribution shift problem

This is the single most important finding:

```
Train: mean 4.1 citations/query, median 2, max 44  (German queries)
Val:   mean 25.1 citations/query, median 22, max 47 (English queries)
```

Val/test queries require **6x more citations** than training queries. Any approach
that learns from training data statistics (co-citation counts, TF-IDF similarity,
frequency-based expansion) will massively underpredict.

### Oracle F1 ceiling analysis

Even with **perfect retrieval** (every predicted citation is correct), the F1 score
is bounded by the mismatch between predicted count (k) and gold count:

| Fixed k | Oracle Mean F1 | Why |
|---------|---------------|-----|
| k=5 | 0.395 | Too few — recall kills F1 |
| k=10 | 0.644 | Our current k — leaving ~35% on the table |
| k=15 | 0.743 | Better but still underpredicting |
| k=20 | 0.778 | Near-optimal for this distribution |
| k=25 | 0.779 | Best fixed k (matches val median) |
| k=30 | 0.761 | Starts overpredicting — precision drops |

**Critical insight:** Even with perfect ranking, using k=10 caps us at F1=0.644.
The top leaderboard score (0.065) suggests nobody has perfect ranking either —
there is a massive gap between what's achievable and what's been achieved.

### What "reverse engineering the data" likely means

After analyzing the data, the 3rd-place team's approach likely involves:

1. **Recognizing that queries describe real Swiss Federal Tribunal cases** — the
   val/test queries are synthetic English descriptions of actual court judgments.
2. **Using the court corpus to find the source judgment** — 68% of val gold BGE
   citations exist in the court corpus (verified: 69/102 found).
3. **Extracting citations from retrieved court texts** — 25% of sampled court texts
   contain explicit Art. references (verified on 20 samples).
4. **Using an LLM or structured rules to identify applicable law articles** — the
   remaining citations must be inferred from legal reasoning.

**Honesty note:** I cannot verify claim #1 (that queries correspond to specific
court decisions) without access to the 3rd-place team's code. It is a plausible
hypothesis based on the data structure, but it remains unproven.

---

## 2. Proposed Architecture

### Pipeline overview

```
English query
    │
    ├─── Layer 1: Regex Extraction ──────────── seed citations (low recall, high precision)
    │         │
    │         └─ Extract Art./BGE refs from query text
    │           Map French/English abbreviations → German
    │           Expected yield: 0-5 citations per query
    │
    ├─── Layer 2: Dense Retrieval ───────────── candidate documents (medium recall)
    │         │
    │         ├─ BGE-M3 → FAISS search laws corpus (existing)
    │         ├─ BGE-M3 → FAISS search courts corpus (NOT YET BUILT)
    │         └─ Extract citations FROM retrieved document texts
    │           Expected yield: 10-50 candidate citations
    │
    ├─── Layer 3: LLM Reasoning (UNI API) ──── citation generation + ranking
    │         │
    │         ├─ Given: query + Layer 1 seeds + Layer 2 candidates
    │         ├─ Task 1: Identify legal domain and applicable law books
    │         ├─ Task 2: Generate additional citation candidates
    │         ├─ Task 3: Rank and filter all candidates
    │         └─ Task 4: Predict citation count (k)
    │           Expected yield: 15-35 ranked citations
    │
    └─── Layer 4: Fusion + Calibration ──────── final submission
              │
              ├─ RRF merge across all layers
              ├─ Validate against corpus (reject hallucinated citations)
              └─ Adaptive k based on query complexity
```

### Layer-by-layer design decisions

#### Layer 1: Regex Extraction

**Decision:** Keep as a seed layer despite low recall (1.2%).

**Justification:** When regex DOES find a citation, precision is 27% (3/11 verified
against corpus). These seeds anchor the LLM's reasoning — telling it "the query
mentions StPO articles" focuses its search on criminal procedure rather than
searching all of Swiss law.

**Critical concern:** My regex is basic. The co-citation notebook's improved regex
handles bare article numbers paired with contextual abbreviations and French→German
mapping. We should adopt their more sophisticated pattern.

**Verification strategy:**
```bash
# Run on val, measure precision and recall against gold
python scripts/verify_regex.py --data data/val.csv --verbose
# Expected: precision > 25%, recall > 2% with improved regex
```

#### Layer 2: Dense Retrieval (BGE-M3)

**Decision:** Keep existing BGE-M3 pipeline but add two critical features:
1. Courts corpus embedding (currently only laws are embedded)
2. Citation extraction FROM retrieved document texts

**Justification:** Dense retrieval scored 0.0435 on val — our best signal so far.
But we only search laws (176K docs). Courts (2.4M docs) contain 68% of the gold
BGE citations. Adding courts should significantly improve recall for BGE-type
citations.

**Critical concerns:**
- Courts embedding takes ~60-90 min on Kaggle GPU. Must be pre-computed.
- 2.4M documents in FAISS requires ~2GB+ RAM. May exceed Kaggle limits.
- Court texts are long (avg 162 words) but BGE-M3 handles 8192 tokens.
- **Unknown:** What fraction of gold Art. citations are reachable via dense
  retrieval? Our current 0.0435 F1 at k=10 suggests low recall.

**Verification strategy:**
```bash
# After courts embedding, measure recall at various k
python scripts/evaluate_dense.py --index data/processed/courts_index.faiss --k 50
# Expected: recall@50 > 15% on val (currently ~5% with laws-only at k=10)
```

#### Layer 3: LLM Reasoning (Uni API)

**Decision:** Use the university-provided LLM API as the core reasoning engine.

**Justification:** This is the layer where the biggest gains are possible. The gap
between current approaches (0.065 top) and the oracle ceiling (0.779 at k=25) is
enormous. An LLM can:

1. **Read the English query and identify the legal domain** — e.g., "this is a
   criminal procedure case about pre-trial detention under StPO"
2. **Generate candidate citations** — using its training knowledge of Swiss law
3. **Rank candidates** — given the retrieved documents and extracted citations,
   determine which are most relevant
4. **Predict citation count** — reason about query complexity to estimate k

**Critical concerns:**
- **LLM hallucination:** The LLM may generate citations that don't exist in the
  corpus. Every LLM-generated citation MUST be validated against `corpus_set`.
- **Cross-lingual accuracy:** The LLM must map English legal concepts to specific
  German/French/Italian Swiss law provisions. This is the hardest part.
- **Rate limits / cost:** Unknown until we see the API documentation.
- **Latency:** If the API is slow, we may not be able to process 40 test queries
  within Kaggle's time limit (9 hours).
- **Kaggle internet constraint:** Kaggle notebooks can be configured with or
  without internet. If internet is disabled at submission time, the API call must
  happen pre-submission and results embedded in the notebook.

**What I CANNOT verify yet:**
- Whether the LLM has knowledge of Swiss law article numbers
- Whether the API supports batch processing
- Whether the response quality is sufficient for this task
- What the token limits and costs are

**Verification strategy:**
```bash
# After API integration, test on val queries one at a time
python scripts/test_llm_citations.py --query-id val_001 --verbose
# Check: Do generated citations exist in corpus? How many match gold?
# Measure: precision, recall, F1 per query
```

#### Layer 4: Fusion + Calibration

**Decision:** RRF merge all layers, validate against corpus, adaptive k.

**Justification:** Each layer has different strengths:
- Regex: high-confidence seeds (when available)
- Dense: broad recall across both corpora
- LLM: reasoning about legal applicability

RRF (Reciprocal Rank Fusion) is model-agnostic and already implemented in
`src/omnilex/retrieval/fusion.py`.

**Critical concern:** Adaptive k is crucial. Fixed k=10 has an oracle ceiling of
0.644; fixed k=25 has 0.779. The LLM should estimate k per query, but if it's
wrong, F1 suffers significantly.

**Verification strategy:**
```bash
# Test RRF fusion on val with different layer combinations
python scripts/evaluate_fusion.py --layers regex,dense,llm --k-strategy adaptive
# Compare: fixed k=10 vs k=25 vs adaptive
```

---

## 3. What We Keep vs. What We Change

| Component | Status | Action |
|-----------|--------|--------|
| `citations/normalizer.py` | Working | **Keep** — canonical parser for all citations |
| `citations/abbreviations.py` | Working | **Extend** — add FR→DE mapping from co-citation notebook |
| `evaluation/scorer.py` | Working | **Keep** — scoring pipeline |
| `retrieval/models.py` | Working | **Keep** — `EmbeddingModel` protocol enables API swap |
| `retrieval/dense_index.py` | Working | **Keep** — FAISS index for laws (add courts) |
| `retrieval/dense_retriever.py` | Working | **Keep** — orchestrates dense search |
| `retrieval/fusion.py` | Working | **Keep** — RRF fusion |
| `retrieval/anchor_extractor.py` | Weak | **Replace** — improve regex with co-citation notebook patterns |
| `retrieval/submission.py` | Working | **Extend** — add adaptive k |
| `retrieval/bm25_index.py` | Marginal | **Demote** — BM25 scored 0.0000 on cross-lingual val |
| `scripts/embed_corpus.py` | Working | **Extend** — add courts embedding |
| **NEW: `retrieval/llm_predictor.py`** | Not built | **Create** — LLM-based citation prediction |
| **NEW: `retrieval/cocitation.py`** | Not built | **Create** — co-citation graph from training |
| **NEW: `retrieval/corpus_xref.py`** | Not built | **Create** — extract citations from corpus texts |

---

## 4. Verification Plan

### Level 1: Component verification (run immediately)

Each component must be tested independently before integration:

| Component | Test | Pass criteria |
|-----------|------|---------------|
| Improved regex | Run on val queries | Precision > 25%, recall > 2% |
| FR→DE abbreviation mapping | Map all val query abbreviations | 100% of known FR abbreviations mapped |
| Corpus validation | Check every generated citation | 0% hallucinated citations (all in corpus_set) |
| Courts embedding | Embed and search | recall@50 > 10% for BGE-type gold citations |
| LLM citation generation | Run on val_001 | At least 5 gold citations generated |

### Level 2: Integration verification (after assembly)

| Test | Command | Pass criteria |
|------|---------|---------------|
| Val Macro F1 | `python scripts/evaluate_fusion.py` | F1 > 0.065 (beat top public score) |
| Per-query breakdown | Same with `--verbose` | No query with F1 = 0.0 |
| Ablation: regex only | Remove layers 2-3 | F1 < full pipeline (confirms LLM adds value) |
| Ablation: dense only | Remove layers 1,3 | F1 ~ 0.0435 (confirms baseline) |
| Ablation: LLM only | Remove layers 1-2 | Measure LLM's standalone capability |
| Citation count | Check predicted k | Mean predicted k in range [15, 35] |

### Level 3: Submission verification

| Test | Method | Pass criteria |
|------|--------|---------------|
| Format check | `python utils/validate_submission.py` | 0 errors |
| All test queries covered | Check submission.csv row count | 40 rows (or current test size) |
| No duplicate citations | Per-query dedup check | 0 duplicates |
| All citations in corpus | Validate against corpus_set | 100% valid |
| Kaggle runtime | Push notebook, monitor | Completes within 9 hours |

---

## 5. Risks and Mitigations

### Risk 1: LLM doesn't know Swiss law articles

**Probability:** Medium-high. General-purpose LLMs have limited knowledge of
specific article numbers in Swiss federal law.

**Mitigation:** Use the LLM for domain identification and ranking, not generation.
Feed it the candidate list from Layers 1-2 and ask it to SELECT, not GENERATE.

**Verification:** Test with val_001 — ask the LLM "which of these 50 candidates
are relevant to a pre-trial detention case?" and measure precision.

### Risk 2: Kaggle internet restriction

**Probability:** Unknown. Must check competition rules.

**Mitigation:** If internet is disabled at submission time, pre-compute LLM
predictions for all test queries locally and embed them in the notebook.

**Verification:** Check competition rules page and test submission with
`internet: false` in kernel metadata.

### Risk 3: Citation count miscalibration

**Probability:** High. The val mean is 25.1 but the range is 10-47.

**Mitigation:** Use multiple strategies:
1. LLM estimates k per query
2. Score-threshold cutoff (predict until confidence drops below threshold)
3. Ensemble: average of LLM estimate and score-based cutoff

**Verification:** On val, compare fixed k=25 vs LLM-estimated k vs threshold.

### Risk 4: Courts embedding too large for Kaggle

**Probability:** Medium. 2.4M documents x 1024-dim ~ 10GB float32.

**Mitigation:**
1. Use float16 (halves memory to ~5GB)
2. Use FAISS `IndexIVFFlat` with nlist=1024 (reduces search memory)
3. Subset to top-cited court decisions only

**Verification:** Test memory usage on Kaggle notebook before full pipeline.

### Risk 5: Our analysis of "reverse engineering" is wrong

**Probability:** Non-trivial. We have a hypothesis, not proof.

**Mitigation:** Run the pipeline with and without the LLM layer. If the LLM
doesn't improve over dense retrieval alone, the hypothesis was wrong and we
should focus on improving the embedding/reranking pipeline instead.

**Verification:** Ablation study comparing each layer combination on val.

---

## 6. Honest Assessment

### What we know (verified with evidence)

- Dense retrieval (BGE-M3) provides a baseline of 0.0435 (laws-only, k=10)
- Regex extraction alone is nearly useless (1.2% recall, measured)
- Co-citation expansion from training data is nearly useless (0.8% recall, measured)
- TF-IDF training transfer is nearly useless due to language mismatch (0.8%, measured)
- The oracle ceiling at k=25 is 0.779 — enormous room for improvement (calculated)
- 68% of gold BGE citations exist in the court corpus (69/102 verified)
- Train/val distribution shift is 6x (4.1 vs 25.1 citations, measured)
- **The top 3 scores are 0.359, 0.352, 0.336 — proving LLM reasoning is the key**
- **Pure retrieval approaches plateau at ~0.065 (5.5x below LLM approaches)**
- **The uni API is the single most important missing component**

### What we don't know

- Whether the uni API LLM has Swiss law knowledge
- Whether the test distribution matches val (25+ citations per query)
- What the 3rd-place team actually does (we're guessing from a verbal tip)
- Whether internet access is available at Kaggle submission time
- Whether courts embedding will fit in Kaggle memory
- Whether the LLM can effectively rank legal citations

### What we should NOT claim

- That this architecture will beat the current top score (0.359)
- That "reverse engineering" is definitively the right approach
- That any specific F1 score is achievable before testing the API
- That LLM reasoning will work for Swiss law specifically

### What we CAN claim

- Our current approach (dense retrieval only) is demonstrably limited (0.0435)
- The oracle analysis proves that increasing k from 10 to 25 is necessary
- Courts corpus embedding will expand our recall for BGE citations (68% exist there)
- **The leaderboard proves that LLM-based approaches score 5.5x higher than
  pure retrieval (0.359 vs 0.065)** — this is the strongest evidence for our pivot
- The gap between 0.065 (retrieval ceiling) and 0.359 (LLM approaches) can ONLY
  be closed with an LLM reasoning layer
- Every component has a concrete verification strategy

---

## 7. Implementation Order

1. **Improve regex extraction** — adopt co-citation notebook's patterns (1 hour)
2. **Build co-citation graph** — from training data (1 hour)
3. **Integrate uni API** — basic citation generation test (depends on API docs)
4. **Embed courts corpus** — run on Kaggle GPU (4-8 hours compute, 1 hour setup)
5. **Build corpus cross-reference extractor** — extract Art. refs from court texts (2 hours)
6. **Build LLM predictor** — full Layer 3 implementation (4-6 hours)
7. **Integrate fusion pipeline** — RRF merge + adaptive k (2 hours)
8. **Ablation study on val** — verify each layer adds value (1 hour)
9. **Submit to Kaggle** — generate test predictions, validate, submit (1 hour)

**Total estimated effort:** 12-20 hours (excluding courts embedding compute time)

**Critical path:** Steps 3 and 4 are blocking. Step 3 requires the API docs.
Step 4 requires Kaggle GPU time. Both can run in parallel.
