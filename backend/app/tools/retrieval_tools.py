"""Retrieval and knowledge tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.app.retrieval.store import get_text_embedder, get_vector_store, RetrievedItem
from backend.app.tools.base import BaseTool


class KnowledgeRetrieveInput(BaseModel):
    query: str
    top_k: int = 5
    filter_type: Optional[str] = None


class KnowledgeRetrieveOutput(BaseModel):
    items: List[Dict[str, Any]]
    query: str
    count: int


class KnowledgeRetriever(BaseTool[KnowledgeRetrieveInput, KnowledgeRetrieveOutput]):
    name = "knowledge_retriever"
    description = "Retrieve contextual knowledge / historical investigations via vector search."
    input_schema = KnowledgeRetrieveInput
    output_schema = KnowledgeRetrieveOutput

    async def _execute(self, validated_input: KnowledgeRetrieveInput) -> KnowledgeRetrieveOutput:
        embedder = get_text_embedder()
        store = get_vector_store()
        q_emb = embedder.embed(validated_input.query)
        items = store.search(q_emb, top_k=validated_input.top_k, filter_type=validated_input.filter_type)
        return KnowledgeRetrieveOutput(
            items=[i.model_dump() for i in items],
            query=validated_input.query,
            count=len(items),
        )


class SimilaritySearchInput(BaseModel):
    query_embedding: List[float]
    top_k: int = 5


class SimilaritySearchOutput(BaseModel):
    items: List[Dict[str, Any]]
    count: int


class SimilaritySearchTool(BaseTool[SimilaritySearchInput, SimilaritySearchOutput]):
    name = "similarity_search"
    description = "Search stored image embeddings for similar examples."
    input_schema = SimilaritySearchInput
    output_schema = SimilaritySearchOutput

    async def _execute(self, validated_input: SimilaritySearchInput) -> SimilaritySearchOutput:
        store = get_vector_store()
        items = store.search(validated_input.query_embedding, top_k=validated_input.top_k, filter_type="image")
        return SimilaritySearchOutput(items=[i.model_dump() for i in items], count=len(items))
