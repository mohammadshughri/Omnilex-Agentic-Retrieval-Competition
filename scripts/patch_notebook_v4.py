"""Patch the Kaggle notebook for LLM v4 + BGE/Docket plan.

What this script does:
1. Inserts a config + val-data cell after the imports cell (so val_queries is
   always defined even when RUN_DENSE=False).
2. Guards every cell that uses the BGE-M3 model or faiss_index with
   `if RUN_DENSE:` so those 20-min cells are skipped.
3. After the anchor-extraction cell, adds anchors_by_qid dict.
4. Appends new cells for Task 2 (predict_articles_v4), Task 3 (truncation sweep),
   and Task 4 (val_007 diagnostic).

Usage:
    python scripts/patch_notebook_v4.py

Input:  notebooks/bge_m3_push/03_dense_retrieval_bge_m3.ipynb  (patched in-place)
Output: same file, modified
"""
import json
import copy
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks" / "bge_m3_push" / "03_dense_retrieval_bge_m3.ipynb"
)


def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def code_cell(source, cell_id=None):
    c = {
        "cell_type": "code",
        "execution_count": None,
        "outputs": [],
        "source": source,
        "metadata": {},
    }
    if cell_id:
        c["id"] = cell_id
    return c


def md_cell(source, cell_id=None):
    c = {"cell_type": "markdown", "source": source, "metadata": {}}
    if cell_id:
        c["id"] = cell_id
    return c


def guard(cell):
    s = src(cell).rstrip()
    if not s:
        return cell
    indented = "\n".join(
        ("    " + line) if line.strip() else "" for line in s.split("\n")
    )
    new_cell = copy.deepcopy(cell)
    new_cell["source"] = (
        f"if RUN_DENSE:\n{indented}\n"
        "else:\n    print('RUN_DENSE=False — skipping dense cell')"
    )
    return new_cell


def contains(cell, *fragments):
    s = src(cell)
    return all(f in s for f in fragments)


# ── load notebook ─────────────────────────────────────────────────────────────

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Step 1: insert config + val-data cell after imports ───────────────────────

sanity_idx = next(
    (
        i
        for i, c in enumerate(cells)
        if "normalizer sanity check" in src(c).lower()
        or "canonicalize('Art." in src(c)
    ),
    None,
)
if sanity_idx is None:
    sanity_idx = next(i for i, c in enumerate(cells) if "citation_f1" in src(c))

print(f"Inserting config+val-data cell after index {sanity_idx}")

CONFIG_SRC = '''\
# ── Run configuration ─────────────────────────────────────────────────────────
RUN_DENSE = False   # dense retrieval confirmed as noise (Session 9); skip embedding

# Always load val data — needed by both dense and LLM pipelines
import csv as _csv_cfg
with open(str(VAL_CSV), encoding="utf-8") as _f:
    val_queries = list(_csv_cfg.DictReader(_f))

gold_sets = [
    set(normalizer.canonicalize_list(
        [c.strip() for c in q["gold_citations"].split(";") if c.strip()]
    ))
    for q in val_queries
]
gold_sets_by_qid = {q["query_id"]: gs for q, gs in zip(val_queries, gold_sets)}
total_gold = sum(len(g) for g in gold_sets)
print(f"RUN_DENSE = {RUN_DENSE}")
print(f"Val queries: {len(val_queries)}, total gold citations: {total_gold}")
'''

cells.insert(sanity_idx + 1, code_cell(CONFIG_SRC, "config-val-data"))

# ── Step 2: guard dense cells ─────────────────────────────────────────────────

DENSE_SIGS = [
    ("BAAI/bge-m3",),
    ("SentenceTransformer(",),
    ("faiss.write_index",),
    ("faiss.read_index",),
    ("faiss_index.search",),
    ("per_query_results",),
    ("results_by_k",),
    ("DENSE_FETCH", "rrf_fuse(anchor_lists"),
    ("courts_index",),
    ("rrf_fuse_3way",),
    ("faiss_index.reconstruct",),
    ("get_bridge_courts",),
    ("generate_hyde_doc",),
    ("hyde_vecs", "model.encode"),
    ("diag_qi",),
    ("sub_index", "faiss.IndexFlatIP"),
]

guarded = 0
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    if src(cell).startswith("if RUN_DENSE:"):
        continue
    for sig in DENSE_SIGS:
        if all(f in src(cell) for f in sig):
            cells[i] = guard(cell)
            guarded += 1
            break

print(f"Guarded {guarded} dense cells")

# ── Step 3: add anchors_by_qid after anchor extraction ───────────────────────

anchor_idx = next(
    (
        i
        for i, c in enumerate(cells)
        if "extract_citation_anchors" in src(c) and "anchor_lists" in src(c)
    ),
    None,
)
if anchor_idx is not None:
    cells.insert(
        anchor_idx + 1,
        code_cell(
            "# Map query_id -> anchor list (used by Task 2+ evaluation)\n"
            "anchors_by_qid = {q['query_id']: al "
            "for q, al in zip(val_queries, anchor_lists)}",
            "anchors-by-qid",
        ),
    )
    print(f"Inserted anchors_by_qid cell after index {anchor_idx}")
else:
    print("WARNING: anchor extraction cell not found")

# ── Step 4: append Tasks 2, 3, 4 cells ───────────────────────────────────────

V4_DEF_SRC = '''\
import re as _re

LLM_MODEL = "qwen3-30b-a3b-instruct-2507"   # Session 9: outperformed 122b on this task
MAX_ARTICLES_REQUEST = 25                    # request many, truncate later (Task 3 calibrates)

V4_SYSTEM = (
    "You are a Swiss legal expert. Given a case description, determine which "
    "Swiss federal law articles a court would cite when deciding this case.\\n\\n"
    "Work in two steps:\\n\\n"
    "ISSUES:\\n"
    "List the distinct legal issues in the case, one line each.\\n\\n"
    "CITATIONS:\\n"
    "List the specific applicable articles, ONE PER LINE, format:\\n"
    "Art. <number> [Abs. <number>] <GERMAN abbreviation>\\n"
    "Example: Art. 221 Abs. 1 StPO\\n\\n"
    "Rules:\\n"
    "- Use GERMAN law abbreviations only (ZGB, OR, StGB, StPO, ZPO, SchKG, BGG, BV, "
    "IVG, AVIG, UVG, KVG, ATSG, ...). Never French (CC, CO, CP, CPP, LAI, LACI) or Italian.\\n"
    "- Order citations from MOST confident to LEAST confident.\\n"
    "- List at most {n} citations. Quality over quantity.\\n"
    "- After the citation list, output nothing else."
)


def predict_articles_v4(query: str, n: int = MAX_ARTICLES_REQUEST) -> list:
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
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "CITATIONS" in text:
        text = text.split("CITATIONS", 1)[1]

    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip(" -*\\u2022:\\t")
        if "Art." not in line:
            continue
        line = line[line.index("Art."):]
        for fr, de in FR_TO_DE.items():
            line = _re.sub(rf"\\b{_re.escape(fr)}\\b", de, line)
        canon = normalizer.canonicalize(line)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def expand_parents(citations: list) -> list:
    """Add Abs.-stripped parent form for every prediction that has Abs. (Session 9 fix)."""
    out, seen = [], set()
    for c in citations:
        for form in (c, _re.sub(r"\\s+Abs\\.\\s+\\d+", "", c)):
            if form not in seen:
                seen.add(form)
                out.append(form)
    return out


print("predict_articles_v4 and expand_parents defined")
'''

EVAL_HELPER_SRC = '''\
def macro_f1_for(preds_by_qid: dict):
    """preds_by_qid: query_id -> iterable of canonical citation strings."""
    rows, f1s = [], []
    for q in val_queries:
        qid = q["query_id"]
        gold = gold_sets_by_qid[qid]
        pred = set(preds_by_qid.get(qid, []))
        s = citation_f1(list(pred), list(gold))
        f1s.append(s["f1"])
        rows.append({
            "query_id": qid, "gold": len(gold), "pred": len(pred),
            "tp": len(pred & gold), "f1": round(s["f1"], 4),
        })
    return sum(f1s) / len(f1s), rows

print("macro_f1_for defined")
'''

DETERMINISM_SRC = '''\
# Determinism check: 3 runs on val_004 (stochastic in Session 9 at temp=0.1)
val_004 = next(q for q in val_queries if q["query_id"] == "val_004")
runs = [predict_articles_v4(val_004["query"]) for _ in range(3)]
print("Run lengths:", [len(r) for r in runs])
print("Identical across runs:", runs[0] == runs[1] == runs[2])
gold4 = gold_sets_by_qid["val_004"]
for i, r in enumerate(runs):
    tp = len(set(expand_parents(r)) & gold4)
    print(f"  run {i}: {len(r)} articles, {tp} TP")
'''

FULL_EVAL_SRC = '''\
from tqdm.auto import tqdm as _tqdm_v4

v4_preds_ordered = {}
for q in _tqdm_v4(val_queries, desc="v4 predict"):
    v4_preds_ordered[q["query_id"]] = predict_articles_v4(q["query"])

v4_full = {
    qid: set(expand_parents(lst)) | set(anchors_by_qid.get(qid, []))
    for qid, lst in v4_preds_ordered.items()
}
macro, rows = macro_f1_for(v4_full)
print(f"v4 (untruncated, n=25) + anchor: Macro F1 = {macro:.4f}   [v2 baseline: 0.0926]")
for r in rows:
    print(r)
'''

SWEEP_SRC = '''\
print(f"{'N':>4}  {'Macro F1':>9}  {'mean pred':>9}")
sweep = {}
for N in [5, 8, 10, 12, 15, 20, 25]:
    preds = {
        qid: set(expand_parents(lst[:N])) | set(anchors_by_qid.get(qid, []))
        for qid, lst in v4_preds_ordered.items()
    }
    macro, rows = macro_f1_for(preds)
    sweep[N] = macro
    mean_pred = sum(r["pred"] for r in rows) / len(rows)
    print(f"{N:>4}  {macro:>9.4f}  {mean_pred:>9.1f}")

BEST_N = max(sweep, key=sweep.get)
print(f"\\nBest N = {BEST_N}, Macro F1 = {sweep[BEST_N]:.4f}  [previous best: 0.0926]")

v4_best = {
    qid: set(expand_parents(lst[:BEST_N])) | set(anchors_by_qid.get(qid, []))
    for qid, lst in v4_preds_ordered.items()
}
'''

DIAG_SRC = '''\
import re as _re_diag

q7 = next(q for q in val_queries if q["query_id"] == "val_007")
gold7 = sorted(gold_sets_by_qid["val_007"])
pred7 = v4_preds_ordered.get("val_007", [])

print("QUERY:\\n", q7["query"][:600], "\\n")
print(f"GOLD ({len(gold7)}):")
for g in gold7:
    print("  ", g)
print(f"\\nLLM v4 PREDICTED ({len(pred7)}):")
for p in pred7:
    print("  ", p)


def _book(c):
    m = _re_diag.search(r"([A-Z\\u00c4\\u00d6\\u00dc][A-Za-z\\u00c4\\u00d6\\u00dc\\u00e4\\u00f6\\u00fc]+)$", c)
    return m.group(1) if m else "?"


def _artno(c):
    m = _re_diag.search(r"Art\\.\\s+(\\d+)", c)
    return int(m.group(1)) if m else None


gold_books = {_book(g) for g in gold7 if g.startswith("Art.")}
pred_books = {_book(p) for p in pred7}
print(f"\\nGold books     : {sorted(gold_books)}")
print(f"Predicted books: {sorted(pred_books)}")
print(f"Book overlap   : {sorted(gold_books & pred_books)}")
near = [
    (p, g)
    for p in pred7
    for g in gold7
    if g.startswith("Art.")
    and _book(p) == _book(g)
    and _artno(p) and _artno(g)
    and abs(_artno(p) - _artno(g)) <= 3
]
print(f"Near misses (same book, \\u00b13 articles): {near}")
'''

new_cells = [
    md_cell("# Task 2 — predict_articles_v4: deterministic, confidence-ordered", "task2-header"),
    code_cell(V4_DEF_SRC, "task2-v4-def"),
    code_cell(EVAL_HELPER_SRC, "task2-eval-helper"),
    code_cell(DETERMINISM_SRC, "task2-determinism"),
    code_cell(FULL_EVAL_SRC, "task2-full-eval"),
    md_cell("# Task 3 — Citation-count calibration: truncation sweep", "task3-header"),
    code_cell(SWEEP_SRC, "task3-sweep"),
    md_cell("# Task 4 — val_007 diagnostic", "task4-header"),
    code_cell(DIAG_SRC, "task4-val007-diag"),
]

cells.extend(new_cells)
print(f"Appended {len(new_cells)} new cells for Tasks 2-4")

# ── write back ────────────────────────────────────────────────────────────────

nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone. Notebook has {len(cells)} cells total.")
print(f"Written to: {NB_PATH}")
