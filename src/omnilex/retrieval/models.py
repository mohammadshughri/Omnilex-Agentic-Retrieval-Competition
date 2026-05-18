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
