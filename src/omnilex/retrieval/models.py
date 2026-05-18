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
    """BGE-M3 embedding model via FlagEmbedding.

    Produces dense vectors. Sparse weights are stored in
    self.last_sparse_weights after each encode_documents call
    with return_sparse=True.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
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
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        if return_sparse:
            self.last_sparse_weights = output["lexical_weights"]
        return np.array(output["dense_vecs"], dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        output = self._model.encode(
            queries,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(output["dense_vecs"], dtype=np.float32)
