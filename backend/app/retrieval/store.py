"""
Retrieval layer: QUERY → EMBEDDING → RETRIEVAL → RERANKING → CONTEXT

Uses FAISS when available, otherwise pure-numpy fallback.
Stores:
- Image embeddings
- Text embeddings
- Investigation history snippets
- Metadata
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievedItem(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = "text"  # text | image | investigation


class RetrievalResult(BaseModel):
    items: List[RetrievedItem]
    query: str
    top_k: int
    backend: str


class VectorStore:
    """Simple FAISS / numpy vector store with metadata."""

    def __init__(self, dim: int = 384, index_path: Optional[str] = None):
        self.dim = dim
        self.settings = get_settings()
        self.index_path = Path(index_path or self.settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._embeddings: List[np.ndarray] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._faiss_index = None
        self._use_faiss = False

        self._try_load_faiss()
        self._load_from_disk()

    def _try_load_faiss(self) -> None:
        try:
            import faiss  # noqa: F401
            self._use_faiss = True
            logger.info("FAISS available")
        except ImportError:
            logger.info("FAISS not installed — using numpy cosine search")
            self._use_faiss = False

    def _load_from_disk(self) -> None:
        meta_file = self.index_path / "metadata.pkl"
        emb_file = self.index_path / "embeddings.npy"
        if meta_file.exists() and emb_file.exists():
            try:
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)
                self._ids = data["ids"]
                self._metadatas = data["metadatas"]
                self._embeddings = list(np.load(emb_file))
                if self._use_faiss and self._embeddings:
                    self._rebuild_faiss()
                logger.info(f"Loaded {len(self._ids)} vectors from disk")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")

    def _save_to_disk(self) -> None:
        try:
            with open(self.index_path / "metadata.pkl", "wb") as f:
                pickle.dump({"ids": self._ids, "metadatas": self._metadatas}, f)
            if self._embeddings:
                np.save(self.index_path / "embeddings.npy", np.stack(self._embeddings))
        except Exception as e:
            logger.warning(f"Could not save index: {e}")

    def _rebuild_faiss(self) -> None:
        if not self._use_faiss or not self._embeddings:
            return
        import faiss
        mat = np.stack(self._embeddings).astype(np.float32)
        faiss.normalize_L2(mat)
        self._faiss_index = faiss.IndexFlatIP(self.dim)
        self._faiss_index.add(mat)

    def add(
        self,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        eid = item_id or str(uuid4())
        emb = np.array(embedding, dtype=np.float32)
        if emb.shape[0] != self.dim:
            # simple pad/truncate
            if emb.shape[0] < self.dim:
                emb = np.pad(emb, (0, self.dim - emb.shape[0]))
            else:
                emb = emb[: self.dim]
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        self._embeddings.append(emb)
        self._metadatas.append({**(metadata or {}), "content": content})
        self._ids.append(eid)

        if self._use_faiss:
            self._rebuild_faiss()
        self._save_to_disk()
        return eid

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[RetrievedItem]:
        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        if q.shape[0] != self.dim:
            if q.shape[0] < self.dim:
                q = np.pad(q, (0, self.dim - q.shape[0]))
            else:
                q = q[: self.dim]
        q = q / (np.linalg.norm(q) + 1e-9)

        if self._use_faiss and self._faiss_index is not None:
            import faiss
            q2 = q.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(q2)
            scores, indices = self._faiss_index.search(q2, min(top_k * 2, len(self._ids)))
            candidates = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            sims = [float(np.dot(q, e)) for e in self._embeddings]
            candidates = sorted(enumerate(sims), key=lambda x: -x[1])[: top_k * 2]

        results: List[RetrievedItem] = []
        for idx, score in candidates:
            meta = self._metadatas[idx]
            if filter_type and meta.get("source_type") != filter_type:
                continue
            results.append(
                RetrievedItem(
                    id=self._ids[idx],
                    content=meta.get("content", ""),
                    score=score,
                    metadata={k: v for k, v in meta.items() if k != "content"},
                    source_type=meta.get("source_type", "text"),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def count(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Text embedding (lightweight)
# ---------------------------------------------------------------------------

class TextEmbedder:
    """Sentence-transformers when available, otherwise hash-based fallback."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self._model = None
        self.dim = 384
        self._try_load()

    def _try_load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformers: {self.model_name}")
        except Exception as e:
            logger.warning(f"sentence-transformers unavailable ({e}) — using hash embedder")
            self._model = None

    def embed(self, text: str) -> List[float]:
        if self._model is not None:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        # Deterministic fallback
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        emb = rng.randn(self.dim).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        return emb.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# Singleton-ish
_store: Optional[VectorStore] = None
_embedder: Optional[TextEmbedder] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        emb = get_text_embedder()
        _store = VectorStore(dim=emb.dim)
    return _store


def get_text_embedder() -> TextEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    return _embedder
