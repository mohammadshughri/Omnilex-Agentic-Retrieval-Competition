"""Dense retrieval baseline using GWDG API embeddings (e5-mistral-7b-instruct).

Usage:
  python scripts/eval_dense_baseline.py                     # full corpus
  python scripts/eval_dense_baseline.py --sample 5000       # quick test on first 5000 rows
  python scripts/eval_dense_baseline.py --sample 5000 --no-cache  # ignore cached index
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.metrics import citation_f1

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LAWS_CSV = DATA_DIR / "laws_de.csv"
VAL_CSV = DATA_DIR / "val.csv"

EMBED_MODEL = "e5-mistral-7b-instruct"
EMBED_DIM = 4096
BATCH_SIZE = 100

api_key = os.environ.get("GWDG_API_KEY") or os.environ.get("KISSKI_API_KEY")
base_url = (
    os.environ.get("GWDG_BASE_URL")
    or os.environ.get("KISSKI_BASE_URL")
    or "https://chat-ai.academiccloud.de/v1"
)
client = OpenAI(api_key=api_key, base_url=base_url)
normalizer = CitationNormalizer()


def embed_texts(texts: list[str], retries: int = 3) -> np.ndarray:
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise


def load_laws(max_docs: int | None = None) -> tuple[list[dict], list[str]]:
    records, texts = [], []
    with open(LAWS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({"citation_raw": row["citation"]})
            parts = [p for p in [row.get("citation", ""), row.get("title", ""), row.get("text", "")] if p]
            texts.append(" ".join(parts)[:512])
            if max_docs and len(records) >= max_docs:
                break
    return records, texts


def build_index(records: list[dict], texts: list[str], index_path: Path, meta_path: Path) -> faiss.Index:
    n = len(texts)
    print(f"Embedding {n} laws in batches of {BATCH_SIZE} via {EMBED_MODEL}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    start = time.time()
    chunk_vecs: list[np.ndarray] = []
    chunk_size = 0

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        vecs = embed_texts(batch)
        chunk_vecs.append(vecs)
        chunk_size += len(vecs)

        if chunk_size >= 2000 or (i + BATCH_SIZE) >= n:
            block = np.vstack(chunk_vecs).astype(np.float32)
            faiss.normalize_L2(block)
            index.add(block)
            chunk_vecs = []
            chunk_size = 0

        if (i // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - start
            pct = (i + len(batch)) / n * 100
            eta = elapsed / max(i + len(batch), 1) * (n - i - len(batch))
            print(f"  {i+len(batch):>6}/{n} ({pct:.1f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    faiss.write_index(index, str(index_path))
    print(f"Done in {time.time()-start:.0f}s — index saved to {index_path}")
    return index


def load_or_build_index(max_docs: int | None, no_cache: bool) -> tuple[faiss.Index, list[dict]]:
    tag = f"_sample{max_docs}" if max_docs else "_full"
    index_path = DATA_DIR / f"laws_e5mistral{tag}.faiss"
    meta_path = DATA_DIR / f"laws_e5mistral{tag}_metadata.json"

    if not no_cache and index_path.exists() and meta_path.exists():
        print(f"Loading cached index: {index_path}")
        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            records = json.load(f)
        return index, records

    records, texts = load_laws(max_docs)
    return build_index(records, texts, index_path, meta_path), records


def evaluate(index: faiss.Index, records: list[dict], k_values: list[int]) -> dict:
    with open(VAL_CSV, encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    print(f"\nEmbedding {len(queries)} val queries...")
    query_vecs = embed_texts([q["query"] for q in queries])
    faiss.normalize_L2(query_vecs)

    results_by_k = {}
    for k in k_values:
        _, idx_mat = index.search(query_vecs, min(k, index.ntotal))
        f1s = []
        for qi, q in enumerate(queries):
            gold = set(normalizer.canonicalize_list(
                [c.strip() for c in q["gold_citations"].split(";") if c.strip()]
            ))
            predicted = set(normalizer.canonicalize_list(
                [records[idx]["citation_raw"] for idx in idx_mat[qi] if idx >= 0]
            ))
            f1s.append(citation_f1(list(predicted), list(gold))["f1"])

        results_by_k[str(k)] = round(sum(f1s) / len(f1s), 4)
        print(f"  k={k:2d}  macro_f1={results_by_k[str(k)]:.4f}")
    return results_by_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only first N rows of laws_de.csv (omit for full corpus)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Rebuild index even if cached version exists")
    args = parser.parse_args()

    index, records = load_or_build_index(args.sample, args.no_cache)

    print("\nEvaluating at k=10 and k=25...")
    k_results = evaluate(index, records, k_values=[10, 25])

    suffix = f"_sample{args.sample}" if args.sample else "_full"
    baseline_path = RESULTS_DIR / "baselines_corrected.json"
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    baseline[f"dense_e5mistral_laws{suffix}"] = {
        "macro_f1_by_k": k_results,
        "model": EMBED_MODEL,
        "corpus_size": len(records),
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nSaved to {baseline_path}")
    print(f"  k=10  macro_f1={k_results.get('10', 'N/A')}")
    print(f"  k=25  macro_f1={k_results.get('25', 'N/A')}")
