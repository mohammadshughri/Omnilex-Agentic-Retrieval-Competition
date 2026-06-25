# BGE-M3 measured update design

## Overview
Update `docs/presentation/presentation.html` to reflect measured BGE-M3 laws-only results and remove pending/expected language across the deck. Replace slide 6 with a full-width metrics table, clarify the measured pipeline, and align roadmap and example slides with the new measured baseline.

## Goals
- Replace pending/expected language with measured results and "early uplift" framing (use planned/future for items not yet run).
- Keep BM25 as 0.0000 with wording that emphasizes irrelevant matches, not "returns nothing."
- Introduce a clean, full-width results table on slide 6 with Macro F1, precision, and recall for multiple top-k values.
- Update roadmap and walkthrough to anchor on the measured 0.0435 Macro F1 uplift and explain next steps.
- Maintain slide consistency: typography, colors, and visual balance.

## Non-goals
- No new data computation or metric recalculation.
- No theme changes or palette adjustments.
- No JS behavior changes.

## Proposed changes

### Slide 3 — BM25
- Update takeaway text to: "BM25 returns irrelevant matches → Macro F1 = 0.0000."
- Ensure wording avoids "returns nothing."

### Slide 4 — BGE-M3
- Replace "evaluation pending" phrasing with: "Dense retrieval implemented and evaluated on laws corpus."
- Remove "Expected 0.10–0.25" if present on this slide.
- Keep Sparse and ColBERT marked as planned.
- Mark the courts embedding time (e.g., "2.4M court passages ~60–90 min") as planned/estimate to avoid implying it is already run.

### Slide 5 — Pipeline
- Add a clear label for the measured system:
  - "Measured run: English query → anchor extraction + BGE-M3 dense → laws FAISS index → RRF → top-k citations (k=2/3/5/10 reported)."
- Replace technical labels in the SVG to match current elements:
  - Encoder subtitle `meaning-vector` stays as the user-facing text (no CLS-token wording).
  - Box titles `Law index` / `Court index` → `Law vector index (FAISS)` / `Court vector index (FAISS)` (short enough to fit).
  - RRF box title → `combine ranked lists (RRF)`.
- Mark courts, sparse, ColBERT as planned/future within the diagram notes using dim text plus an explicit "planned" label. Do not use dashed styling for planned items; dashed lines remain reserved for the anchor branch only.
- Clarify that RRF merges the anchor list with the dense list (two ranked lists, even without courts).
- Update pipeline stats to remove "3 RRF channels (anchors + laws + courts)" and reflect laws-only measured setup (e.g., "2 ranked lists: anchors + laws" with courts planned).

### Slide 6 — Results
- Update the slide header/lead to state results are measured (remove queued/expected language) and include a small caveat that results are from n=10 val queries.
- Replace the current panel layout (including the "Target Hybrid System" panel) with a single full-width table.
- Table columns: System, Top-k, Macro F1, Precision, Recall.
- Use 8 rows total (4 per system):
  - BM25 body-text baseline (val set, body-text only):
    - k=2: 0.0000 / 0.0000 / 0.0000
    - k=3: 0.0000 / 0.0000 / 0.0000
    - k=5: 0.0000 / 0.0000 / 0.0000
    - k=10: 0.0000 / 0.0000 / 0.0000
  - BGE-M3 dense + anchors, laws-only:
    - k=2: Macro F1 0.0258, Precision 0.1500, Recall 0.0142
    - k=3: Macro F1 0.0342, Precision 0.1333, Recall 0.0197
    - k=5: Macro F1 0.0398, Precision 0.1000, Recall 0.0250
    - k=10: **Macro F1 0.0435**, Precision 0.0733, Recall 0.0314
- Remove any mention of the original target range on this slide to avoid mixed messaging.
- Keep the results takeaway compact to avoid overflow: one tk-text block that includes both sentences, with the longer callout bolded within the same block (no separate callout block).

### Slide 7 — Roadmap
- Update baseline row: "Dense BGE-M3 laws-only: measured 0.0435 Macro F1."
- Adjust next-step descriptions:
  - Add courts index (gold citations include court decisions).
  - Add reranker (improve ranking quality).
  - Add sparse/ColBERT (recover exact legal references and fine-grained matches).
  - Add HyDE (generate German legal-style text from English query).
  - Calibrate top-k (top-10 best, recall needs help).
- Place "Add courts index" as a short sub-line in the Week 3 row (the reranker row) so it appears in the Gantt.
- Update the "Next" callout to reflect measured baseline and the immediate next step (e.g., "Next: add courts index, then rerank top candidates.").
- Replace the "All F1 values are estimates" footer to clarify only future phases are estimates.

### Slide 7 (07/07) — Example
- Update the header label from "BGE-M3 Path (Expected)" to "BGE-M3 Path (Measured, laws-only)".
- Replace "Expected: F1 > 0" with:
  - "Measured system improves over BM25 overall: best Macro F1 = 0.0435 (aggregate over val set)."
- Update BGE-M3 step text:
  - "Anchor extraction surfaces Art. 221 Abs. 1 StPO as a priority candidate."
  - "Dense retrieval aims to recover related detention articles, but per-query results still need inspection."
- Replace any remaining "evaluation in progress/target" copy in the walkthrough with measured wording and 4-decimal formatting.
- Use only citations already present in the example query; do not introduce new citations.

## Layout and component updates
- Add CSS classes for a compact, readable results table:
  - `.results-table`, `.results-table th`, `.results-table td`, `.row-highlight`.
  - Use existing font stack and colors for consistency with the deck.
- Replace the slide 6 `results-wrap` grid with a single table container that fits within the existing `zone-main` bounds.
- Use a subtle row highlight for the BGE-M3 k=10 row and bold emphasis only for the best Macro F1 value (0.0435).
- Use consistent 4-decimal formatting across results (e.g., 0.0000 vs 0.00).
  - Update slide 3 BM25 Macro F1 display to 0.0000 for consistency.
  - Update slide 7 F1 mentions to 0.0000 formatting where applicable.

## Validation
- Open the HTML and visually verify:
  - Slide 6 table fits on screen without overflow.
  - All pending/expected/target language is removed or demoted as specified across the full deck (including slide headers and labels).
  - Pipeline and roadmap text remain readable within their boxes.
  - No inconsistent labels remain (FAISS/RRF/CLS-token terms handled).

## Traceability checklist
- Slide 3 BM25: keep 0.0000 and replace takeaway with "BM25 returns irrelevant matches → Macro F1 = 0.0000."
- Slide 4 BGE-M3: status changed to measured; remove expected range text; courts embed time marked planned/estimate.
- Slide 5 Pipeline: measured run label added; FAISS/RRF labels made user-facing; courts/sparse/ColBERT marked planned.
- Slide 6 Results: full-width metrics table with provided values; compact takeaway with measured uplift wording.
- Slide 7 Roadmap: baseline updated to measured 0.0435; next steps aligned to weaknesses (courts, reranker, sparse/ColBERT, HyDE, top-k).
- Slide 7 Example: measured wording replaces expected/target; best Macro F1 cited as aggregate; caution about per-query inspection.
