"""Build dense FAISS indices from raw corpus CSVs.

Usage:
    python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws
    python scripts/embed_corpus.py --corpus courts --output data/processed/bge_m3/courts
    python scripts/embed_corpus.py --corpus laws --output data/processed/bge_m3/laws --limit 100
"""

import argparse
import csv
import sys
from pathlib import Path


def load_csv_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            records.append(dict(row))
    return records


def main():
    parser = argparse.ArgumentParser(description="Build dense FAISS index from corpus CSV")
    parser.add_argument("--corpus", choices=["laws", "courts"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()

    data_dir = Path("data")

    if args.corpus == "laws":
        csv_path = data_dir / "laws_de.csv"
        title_field = "title"
    else:
        csv_path = data_dir / "court_considerations.csv"
        title_field = None

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run from Omnilex-Agentic-Retrieval-Competition/")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    records = load_csv_records(csv_path, limit=args.limit)
    print(f"Loaded {len(records)} records")

    from omnilex.retrieval.models import BgeM3Embedder
    from omnilex.retrieval.dense_index import DenseIndexBuilder

    print(f"Loading model {args.model}...")
    embedder = BgeM3Embedder(model_name=args.model)

    builder = DenseIndexBuilder(embedder)
    print(f"Building index -> {args.output}")
    builder.build_from_records(
        records=records,
        citation_field="citation",
        text_field="text",
        title_field=title_field,
        output_dir=args.output,
        batch_size=args.batch_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
