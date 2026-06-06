"""Model protocols for retrieval — embedding and reranking abstractions."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Protocol for dense embedding models (BGE-M3, Qwen3, etc.)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @abstractmethod
    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


class RerankerModel(ABC):
    """Protocol for reranker models (cross-encoder, causal LM, etc.)."""

    @abstractmethod
    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray: ...


class BgeM3Embedder(EmbeddingModel):
    """BGE-M3 embedding model via sentence-transformers (dense-only)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from sentence_transformers import SentenceTransformer
        import torch
        self._model = SentenceTransformer(model_name)
        if use_fp16 and torch.cuda.is_available():
            self._model.half()
        self.last_sparse_weights: list[dict] | None = None

    @property
    def embedding_dim(self) -> int:
        return 1024

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_sparse: bool = False,
    ) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)
