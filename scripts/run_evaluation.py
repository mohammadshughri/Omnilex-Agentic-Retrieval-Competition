"""Evaluate retrieval pipeline on val.csv.

Usage:
    python scripts/run_evaluation.py \
        --laws-index data/processed/bge_m3/laws \
        --val-csv data/raw/lexam/val.csv \
        --top-k 2 3 5 10 \
        --output results/bgem3_laws_only.json
"""

import argparse
import csv
import json
from pathlib import Path

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.metrics import citation_f1, macro_f1
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.dense_retriever import DenseRetriever
from omnilex.retrieval.models import BgeM3Embedder


def load_val_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on val.csv")
    parser.add_argument("--laws-index", type=Path, default=None)
    parser.add_argument("--courts-index", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    normalizer = CitationNormalizer()

    print(f"Loading model {args.model}...")
    embedder = BgeM3Embedder(model_name=args.model)

    laws_index = DenseIndex.load(args.laws_index) if args.laws_index else None
    courts_index = DenseIndex.load(args.courts_index) if args.courts_index else None

    retriever = DenseRetriever(
        embedder=embedder,
        laws_index=laws_index,
        courts_index=courts_index,
    )

    queries = load_val_queries(args.val_csv)
    print(f"Evaluating on {len(queries)} queries")

    all_gold = []
    all_retrieved: dict[int, list] = {}

    for k in args.top_k:
        all_retrieved[k] = []

    for q in queries:
        gold_raw = q.get("gold_citations", "")
        gold_canonical = normalizer.canonicalize_list(
            [c.strip() for c in gold_raw.split(";") if c.strip()]
        )
        all_gold.append(gold_canonical)

        candidates = retriever.retrieve(q["query"], top_k=max(args.top_k), faiss_top_k=100)

        for k in args.top_k:
            top = candidates[:k]
            pred_canonical = normalizer.canonicalize_list([c.citation_raw for c in top])
            all_retrieved[k].append(pred_canonical)

    results = {"val_queries": len(queries)}

    for k in args.top_k:
        scores = macro_f1(all_retrieved[k], all_gold)
        results[f"top_{k}"] = scores
        print(f"  top-{k}: Macro F1={scores['macro_f1']:.4f}  "
              f"P={scores['macro_precision']:.4f}  R={scores['macro_recall']:.4f}")

    best_k = max(args.top_k, key=lambda k: results[f"top_{k}"]["macro_f1"])
    results["best_k"] = best_k
    results["per_query"] = []

    for i, q in enumerate(queries):
        pq = citation_f1(all_retrieved[best_k][i], all_gold[i])
        pq["query_id"] = q["query_id"]
        pq["query_preview"] = q["query"][:100]
        pq["num_gold"] = len(all_gold[i])
        pq["num_predicted"] = len(all_retrieved[best_k][i])
        results["per_query"].append(pq)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print(f"\nBest k={best_k} -> Macro F1={results[f'top_{best_k}']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
