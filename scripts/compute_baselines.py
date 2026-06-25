"""Recompute baselines after normalizer fix.

Run: python scripts/compute_baselines.py
Depends on: Task 1 (normalizer fix) being complete.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.metrics import citation_f1
from omnilex.retrieval.anchor_extractor import extract_citation_anchors


def oracle_f1_at_k(gold_counts: list[int], k: int) -> float:
    """If you got the top-k predictions perfectly right, what F1 would you get?"""
    f1s = []
    for n_gold in gold_counts:
        if n_gold == 0:
            f1s.append(1.0 if k == 0 else 0.0)
            continue
        tp = min(k, n_gold)
        precision = tp / k if k > 0 else 0
        recall = tp / n_gold
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    normalizer = CitationNormalizer()

    with open(data_dir / "val.csv", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    # 1. Full citation breakdown
    per_query = []
    all_gold_counts = []
    total_art, total_bge, total_docket = 0, 0, 0

    for q in queries:
        raw_cites = [c.strip() for c in q["gold_citations"].split(";") if c.strip()]
        canonical = normalizer.canonicalize_list(raw_cites)
        all_gold_counts.append(len(canonical))

        n_art = sum(1 for c in canonical if c.startswith("Art."))
        n_bge = sum(1 for c in canonical if c.startswith("BGE"))
        n_docket = sum(1 for c in canonical
                       if re.match(r"\d+[A-Z]_\d+/\d+", c))

        total_art += n_art
        total_bge += n_bge
        total_docket += n_docket

        per_query.append({
            "query_id": q["query_id"],
            "raw_count": len(raw_cites),
            "canonical_count": len(canonical),
            "art": n_art, "bge": n_bge, "docket": n_docket,
        })

    # 2. Anchor-only evaluation
    anchor_f1s = []
    for q in queries:
        raw_cites = [c.strip() for c in q["gold_citations"].split(";") if c.strip()]
        gold = normalizer.canonicalize_list(raw_cites)
        anchors = extract_citation_anchors(q["query"])
        scores = citation_f1(anchors, gold)
        anchor_f1s.append(scores["f1"])
    anchor_macro_f1 = sum(anchor_f1s) / len(anchor_f1s)

    # 3. Oracle F1 at various k
    oracle = {}
    for k in [5, 10, 15, 20, 25, 30]:
        oracle[str(k)] = round(oracle_f1_at_k(all_gold_counts, k), 4)

    results = {
        "normalizer_version": "with_docket_support",
        "val_queries": len(queries),
        "val_total_citations": sum(all_gold_counts),
        "val_mean_citations": round(sum(all_gold_counts) / len(all_gold_counts), 1),
        "citation_type_totals": {
            "Art": total_art, "BGE": total_bge, "docket": total_docket,
        },
        "anchor_only_macro_f1": round(anchor_macro_f1, 4),
        "oracle_f1_at_k": oracle,
        "per_query": per_query,
    }

    print(f"Val: {results['val_queries']} queries, {results['val_total_citations']} total citations")
    print(f"  Mean citations/query: {results['val_mean_citations']}")
    print(f"  Types: {results['citation_type_totals']}")
    print(f"  Anchor-only Macro F1: {results['anchor_only_macro_f1']}")
    print(f"  Oracle F1 at k: {results['oracle_f1_at_k']}")
    print(f"\n  Per-query:")
    for pq in per_query:
        print(f"    {pq['query_id']}: {pq['canonical_count']} citations "
              f"({pq['art']} Art + {pq['bge']} BGE + {pq['docket']} docket)")

    out = results_dir / "baselines_corrected.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
