"""Seed demo knowledge into the vector store."""

from backend.app.core.logging import get_logger
from backend.app.retrieval.store import get_text_embedder, get_vector_store

logger = get_logger(__name__)

DEMO_KNOWLEDGE = [
    {
        "content": "Surface scratches on metal parts are common manufacturing defects and often appear as linear discontinuities in texture.",
        "metadata": {"source_type": "text", "topic": "surface_defect", "label": "anomaly"},
    },
    {
        "content": "Normal variation in lighting and slight perspective changes should not be classified as anomalies if cross-view consistency remains high.",
        "metadata": {"source_type": "text", "topic": "normal_variation", "label": "normal"},
    },
    {
        "content": "Crack-like patterns that appear consistently across multiple camera views have higher likelihood of being true structural defects.",
        "metadata": {"source_type": "text", "topic": "crack", "label": "anomaly"},
    },
    {
        "content": "Low-quality or heavily compressed images reduce confidence of visual anomaly detectors; prefer requesting additional views.",
        "metadata": {"source_type": "text", "topic": "image_quality", "label": "uncertain"},
    },
    {
        "content": "When retrieved examples are irrelevant to the object category, semantic confidence should be down-weighted.",
        "metadata": {"source_type": "text", "topic": "retrieval", "label": "uncertain"},
    },
    {
        "content": "Discoloration that is uniform across the object is more often a material property than a localized defect.",
        "metadata": {"source_type": "text", "topic": "discoloration", "label": "normal"},
    },
    {
        "content": "Historical investigations with anomaly scores above 0.7 and supporting cross-view evidence were confirmed as defects in 82% of reviewed cases (synthetic demo statistic).",
        "metadata": {"source_type": "investigation", "topic": "prior", "label": "anomaly"},
    },
]


def seed_demo_knowledge() -> int:
    store = get_vector_store()
    if store.count() >= len(DEMO_KNOWLEDGE):
        logger.info(f"Vector store already has {store.count()} items — skip seed")
        return 0
    embedder = get_text_embedder()
    n = 0
    for item in DEMO_KNOWLEDGE:
        emb = embedder.embed(item["content"])
        store.add(emb, item["content"], metadata=item["metadata"])
        n += 1
    logger.info(f"Seeded {n} knowledge items")
    return n
