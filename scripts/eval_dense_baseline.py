"""Dense retrieval baseline — supports local multilingual-e5-large or GWDG API embeddings.

Usage:
  # local multilingual-e5-large (recommended — cross-lingual EN→DE, free, 1024-dim)
  python scripts/eval_dense_baseline.py --embedder local --sample 1000

  # GWDG API e5-mistral-7b-instruct (4096-dim, EN-primary)
  python scripts/eval_dense_baseline.py --embedder api --sample 1000

  # full corpus run (local is faster with no rate limits)
  python scripts/eval_dense_baseline.py --embedder local

  # ignore cached index and rebuild
  python scripts/eval_dense_baseline.py --embedder local --no-cache
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

normalizer = CitationNormalizer()


# ── Embedders ─────────────────────────────────────────────────────────────────

class LocalEmbedder:
    """multilingual-e5-large via sentence-transformers (CPU, 1024-dim).

    Best for cross-lingual retrieval: English queries against German laws.
    Prefix convention: queries get 'query: ', docs get 'passage: '.
    """
    MODEL_NAME = "intfloat/multilingual-e5-large"
    DIM = 1024
    BATCH_SIZE = 64

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"Loading {self.MODEL_NAME} (downloads on first run ~560MB)...")
        self._model = SentenceTransformer(self.MODEL_NAME)
        print("Model loaded.")

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(prefixed, batch_size=self.BATCH_SIZE,
                                  normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        vecs = self._model.encode(prefixed, batch_size=self.BATCH_SIZE,
                                  normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)


class ApiEmbedder:
    """e5-mistral-7b-instruct via GWDG API (4096-dim, EN-primary)."""
    MODEL_NAME = "e5-mistral-7b-instruct"
    DIM = 4096
    BATCH_SIZE = 100

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get("GWDG_API_KEY") or os.environ.get("KISSKI_API_KEY")
        base_url = (os.environ.get("GWDG_BASE_URL")
                    or os.environ.get("KISSKI_BASE_URL")
                    or "https://chat-ai.academiccloud.de/v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _embed(self, texts: list[str], retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            try:
                resp = self._client.embeddings.create(model=self.MODEL_NAME, input=texts)
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"    retry {attempt+1}: {e}", flush=True)
                    time.sleep(2 ** attempt)
                else:
                    raise

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


# ── Index build / load ─────────────────────────────────────────────────────────

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


def build_index(embedder, records: list[dict], texts: list[str],
                index_path: Path, meta_path: Path) -> faiss.Index:
    n = len(texts)
    bs = embedder.BATCH_SIZE
    print(f"Embedding {n} laws with {embedder.MODEL_NAME} (batch={bs})...", flush=True)
    index = faiss.IndexFlatIP(embedder.DIM)
    start = time.time()
    chunk_vecs: list[np.ndarray] = []
    chunk_size = 0

    for i in range(0, n, bs):
        batch = texts[i: i + bs]
        vecs = embedder.embed_docs(batch)
        chunk_vecs.append(vecs)
        chunk_size += len(vecs)

        if chunk_size >= 2000 or (i + bs) >= n:
            block = np.vstack(chunk_vecs).astype(np.float32)
            faiss.normalize_L2(block)
            index.add(block)
            chunk_vecs = []
            chunk_size = 0

        if (i // bs) % 10 == 0:
            elapsed = time.time() - start
            pct = (i + len(batch)) / n * 100
            eta = elapsed / max(i + len(batch), 1) * (n - i - len(batch))
            print(f"  {i+len(batch):>6}/{n} ({pct:.1f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    faiss.write_index(index, str(index_path))
    print(f"Done in {time.time()-start:.0f}s — saved to {index_path}", flush=True)
    return index


def load_or_build_index(embedder, max_docs: int | None, no_cache: bool):
    tag = f"_{embedder.MODEL_NAME.replace('/', '_').replace('-', '_')}"
    if max_docs:
        tag += f"_sample{max_docs}"
    index_path = DATA_DIR / f"laws{tag}.faiss"
    meta_path = DATA_DIR / f"laws{tag}_metadata.json"

    if not no_cache and index_path.exists() and meta_path.exists():
        print(f"Loading cached index: {index_path}", flush=True)
        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            records = json.load(f)
        return index, records

    records, texts = load_laws(max_docs)
    return build_index(embedder, records, texts, index_path, meta_path), records


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(embedder, index: faiss.Index, records: list[dict], k_values: list[int]) -> dict:
    with open(VAL_CSV, encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    print(f"\nEmbedding {len(queries)} val queries...", flush=True)
    query_vecs = embedder.embed_queries([q["query"] for q in queries])
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
        print(f"  k={k:2d}  macro_f1={results_by_k[str(k)]:.4f}", flush=True)
    return results_by_k


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["local", "api"], default="local",
                        help="local=multilingual-e5-large (recommended), api=e5-mistral via GWDG")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only first N rows of laws_de.csv")
    parser.add_argument("--no-cache", action="store_true",
                        help="Rebuild index even if cached version exists")
    args = parser.parse_args()

    embedder = LocalEmbedder() if args.embedder == "local" else ApiEmbedder()
    index, records = load_or_build_index(embedder, args.sample, args.no_cache)

    print("\nEvaluating at k=10 and k=25...", flush=True)
    k_results = evaluate(embedder, index, records, k_values=[10, 25])

    baseline_path = RESULTS_DIR / "baselines_corrected.json"
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    key = f"dense_{embedder.MODEL_NAME.split('/')[-1].replace('-', '_')}"
    if args.sample:
        key += f"_sample{args.sample}"
    baseline[key] = {
        "macro_f1_by_k": k_results,
        "model": embedder.MODEL_NAME,
        "corpus_size": len(records),
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nSaved to {baseline_path}")
    print(f"  k=10  macro_f1={k_results.get('10', 'N/A')}")
    print(f"  k=25  macro_f1={k_results.get('25', 'N/A')}")
