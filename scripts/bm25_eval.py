#!/usr/bin/env python3
"""BM25 baseline evaluation on val.csv.

Builds BM25 index from laws_de.csv and court_considerations.csv,
retrieves top-k citations for each val query, and scores with Macro F1.

Usage:
    python scripts/bm25_eval.py
    python scripts/bm25_eval.py --laws-only --top-k 5 10
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.evaluation.scorer import Scorer
from omnilex.retrieval.bm25_index import BM25Index


def load_laws(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, usecols=["citation", "text"], dtype=str)
    df = df.dropna(subset=["citation", "text"])
    return df.to_dict("records")


def load_courts(csv_path: Path, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(csv_path, usecols=["citation", "text"], dtype=str, nrows=limit)
    df = df.dropna(subset=["citation", "text"])
    return df.to_dict("records")


def retrieve(
    query: str,
    law_index: BM25Index,
    court_index: BM25Index | None,
    top_k: int,
) -> list[str]:
    results = law_index.search(query, top_k=top_k)
    if court_index:
        results += court_index.search(query, top_k=top_k)
    seen, citations = set(), []
    for r in results:
        c = r.get("citation", "").strip()
        if c and c not in seen:
            seen.add(c)
            citations.append(c)
    return citations[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laws-csv", type=Path, default=ROOT / "data" / "laws_de.csv")
    parser.add_argument("--courts-csv", type=Path, default=ROOT / "data" / "court_considerations.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT / "data" / "val.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "bm25_baseline.json")
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--laws-only", action="store_true", help="Skip court decisions index (faster)")
    args = parser.parse_args()

    print(f"Loading laws from {args.laws_csv} ...")
    laws_docs = load_laws(args.laws_csv)
    print(f"  {len(laws_docs):,} law passages loaded")

    print("Building laws BM25 index ...")
    law_index = BM25Index(documents=laws_docs, text_field="text", citation_field="citation")

    court_index = None
    if not args.laws_only and args.courts_csv.exists():
        print(f"Loading courts from {args.courts_csv} (this may take a few minutes) ...")
        courts_docs = load_courts(args.courts_csv)
        print(f"  {len(courts_docs):,} court passages loaded")
        print("Building courts BM25 index ...")
        court_index = BM25Index(documents=courts_docs, text_field="text", citation_field="citation")
    else:
        print("Skipping courts index (--laws-only or file not found)")

    val_df = pd.read_csv(args.val_csv)
    scorer = Scorer()
    results_by_k: dict[int, dict] = {}

    for k in args.top_k:
        rows = []
        for _, row in val_df.iterrows():
            citations = retrieve(row["query"], law_index, court_index, top_k=k)
            rows.append({"query_id": row["query_id"], "predicted_citations": ";".join(citations)})

        submission_df = pd.DataFrame(rows)
        gold_df = val_df[["query_id", "gold_citations"]].copy()

        sub_path = Path(tempfile.mktemp(suffix=".csv"))
        submission_df.to_csv(sub_path, index=False, encoding="utf-8")

        gold_path = Path(tempfile.mktemp(suffix=".csv"))
        gold_df.to_csv(gold_path, index=False, encoding="utf-8")

        scores = scorer.score(sub_path, gold_path)
        sub_path.unlink()
        gold_path.unlink()

        results_by_k[k] = scores
        print(
            f"k={k:2d}  Macro F1={scores['macro_f1']:.4f}"
            f"  Precision={scores['macro_precision']:.4f}"
            f"  Recall={scores['macro_recall']:.4f}"
        )

    output = {
        "system": "bm25_baseline",
        "laws_only": args.laws_only or court_index is None,
        "results": {str(k): v for k, v in results_by_k.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
