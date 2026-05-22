"""Dense FAISS index builder and loader."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from tqdm.auto import tqdm

from omnilex.retrieval.models import EmbeddingModel


class DenseIndexBuilder:
    """Builds a FAISS index + metadata from document records."""

    def __init__(self, embedder: EmbeddingModel):
        self._embedder = embedder

    def _compose_text(
        self,
        record: dict,
        text_field: str,
        citation_field: str | None,
        title_field: str | None,
    ) -> str:
        parts = []
        if citation_field and citation_field in record:
            parts.append(record[citation_field])
        if title_field and title_field in record:
            parts.append(record[title_field])
        parts.append(record.get(text_field, ""))
        return " ".join(parts)

    def build_from_records(
        self,
        records: list[dict],
        citation_field: str,
        text_field: str,
        output_dir: Path | str,
        title_field: str | None = None,
        batch_size: int = 256,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        texts = [
            self._compose_text(r, text_field, citation_field, title_field)
            for r in records
        ]

        all_vecs = []
        with tqdm(total=len(texts), desc="Embedding", unit=" docs") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self._embedder.encode_documents(batch, batch_size=batch_size)
                all_vecs.append(vecs)
                pbar.update(len(batch))

        embeddings = np.vstack(all_vecs).astype(np.float32)
        np.save(output_dir / "dense.npy", embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        normed = embeddings.copy()
        faiss.normalize_L2(normed)
        index.add(normed)
        faiss.write_index(index, str(output_dir / "faiss.index"))

        with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for idx, rec in enumerate(records):
                meta = {
                    "idx": idx,
                    "citation_raw": rec.get(citation_field, ""),
                    "text_preview": rec.get(text_field, "")[:200],
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")


class DenseIndex:
    """Loaded dense FAISS index with metadata."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict],
        embeddings: np.ndarray | None = None,
    ):
        self.index = index
        self.metadata = metadata
        self.embeddings = embeddings

    @classmethod
    def load(cls, index_dir: Path | str) -> DenseIndex:
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "faiss.index"))

        metadata = []
        with open(index_dir / "metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))

        embeddings = None
        npy_path = index_dir / "dense.npy"
        if npy_path.exists():
            embeddings = np.load(npy_path)

        return cls(index=index, metadata=metadata, embeddings=embeddings)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
    ) -> list[dict]:
        qv = query_vectors.astype(np.float32).copy()
        faiss.normalize_L2(qv)
        scores, indices = self.index.search(qv, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results
