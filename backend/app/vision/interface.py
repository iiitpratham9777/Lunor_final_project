"""
Modular Vision Interface.

Models can be swapped without changing the agent.
Supports:
- Global embeddings
- Patch-level embeddings
- Multi-scale features
- Image similarity
- Region-level analysis
- Cross-view comparison
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    path: str
    view_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    global_embedding: List[float]
    patch_embeddings: Optional[List[List[float]]] = None
    model_name: str
    dim: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnomalyScoreResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # higher = more anomalous
    spatial_map: Optional[List[List[float]]] = None
    method: str
    details: Dict[str, Any] = Field(default_factory=dict)


class SimilarityResult(BaseModel):
    scores: List[float]
    indices: List[int]
    method: str


class CrossViewResult(BaseModel):
    consistency_score: float = Field(ge=0.0, le=1.0)
    pairwise_distances: List[float]
    outlier_views: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class VisionBackend(ABC):
    """Abstract vision backend — implement this for new models."""

    name: str

    @abstractmethod
    async def embed(self, image: ImageInput) -> EmbeddingResult:
        ...

    @abstractmethod
    async def embed_batch(self, images: List[ImageInput]) -> List[EmbeddingResult]:
        ...

    @abstractmethod
    async def compute_anomaly_score(
        self,
        query_embedding: List[float],
        reference_embeddings: List[List[float]],
        method: str = "mahalanobis",
    ) -> AnomalyScoreResult:
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        gallery_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> SimilarityResult:
        ...

    @abstractmethod
    async def cross_view_consistency(
        self,
        embeddings: List[EmbeddingResult],
        view_ids: List[str],
    ) -> CrossViewResult:
        ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine_distances(embeddings: List[List[float]]) -> List[float]:
    mats = [np.array(e, dtype=np.float64) for e in embeddings]
    dists = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            sim = cosine_similarity(mats[i], mats[j])
            dists.append(1.0 - sim)
    return dists
