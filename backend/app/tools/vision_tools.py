"""Vision-related tools with typed I/O."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.app.tools.base import BaseTool
from backend.app.vision.backends import combined_anomaly_score, get_vision_backend
from backend.app.vision.interface import ImageInput


class FeatureExtractInput(BaseModel):
    image_paths: List[str]
    view_ids: Optional[List[str]] = None


class FeatureExtractOutput(BaseModel):
    embeddings: List[List[float]]
    dims: List[int]
    model_name: str
    view_ids: List[str]


class ImageFeatureExtractor(BaseTool[FeatureExtractInput, FeatureExtractOutput]):
    name = "image_feature_extractor"
    description = "Extract global (and optional patch) embeddings from one or more images."
    input_schema = FeatureExtractInput
    output_schema = FeatureExtractOutput

    async def _execute(self, validated_input: FeatureExtractInput) -> FeatureExtractOutput:
        backend = get_vision_backend()
        view_ids = validated_input.view_ids or [f"view_{i}" for i in range(len(validated_input.image_paths))]
        images = [
            ImageInput(path=p, view_id=vid)
            for p, vid in zip(validated_input.image_paths, view_ids)
        ]
        results = await backend.embed_batch(images)
        return FeatureExtractOutput(
            embeddings=[r.global_embedding for r in results],
            dims=[r.dim for r in results],
            model_name=results[0].model_name if results else "none",
            view_ids=view_ids,
        )


class AnomalyScoreInput(BaseModel):
    query_embedding: List[float]
    reference_embeddings: List[List[float]] = Field(default_factory=list)
    method: str = "distance_to_mean"
    cross_view_inconsistency: float = 0.0


class AnomalyScoreOutput(BaseModel):
    score: float
    method: str
    details: Dict[str, Any]
    combined_score: Optional[float] = None


class AnomalyScorer(BaseTool[AnomalyScoreInput, AnomalyScoreOutput]):
    name = "anomaly_scorer"
    description = "Compute feature-space anomaly score given query and reference embeddings."
    input_schema = AnomalyScoreInput
    output_schema = AnomalyScoreOutput

    async def _execute(self, validated_input: AnomalyScoreInput) -> AnomalyScoreOutput:
        backend = get_vision_backend()
        result = await backend.compute_anomaly_score(
            validated_input.query_embedding,
            validated_input.reference_embeddings,
            method=validated_input.method,
        )
        # Also produce combined score using documented formula
        visual_dev = result.score
        ref_dist = result.score  # same source for prototype
        combined = combined_anomaly_score(
            visual_deviation=visual_dev,
            reference_distance=ref_dist,
            cross_view_inconsistency=validated_input.cross_view_inconsistency,
        )
        return AnomalyScoreOutput(
            score=result.score,
            method=result.method,
            details=result.details,
            combined_score=combined,
        )


class CrossViewInput(BaseModel):
    embeddings: List[List[float]]
    view_ids: List[str]


class CrossViewOutput(BaseModel):
    consistency_score: float
    pairwise_distances: List[float]
    outlier_views: List[str]
    details: Dict[str, Any]


class CrossViewComparator(BaseTool[CrossViewInput, CrossViewOutput]):
    name = "cross_view_comparator"
    description = "Measure consistency across multiple views of the same object."
    input_schema = CrossViewInput
    output_schema = CrossViewOutput

    async def _execute(self, validated_input: CrossViewInput) -> CrossViewOutput:
        from backend.app.vision.interface import EmbeddingResult

        backend = get_vision_backend()
        emb_objs = [
            EmbeddingResult(global_embedding=e, model_name="n/a", dim=len(e))
            for e in validated_input.embeddings
        ]
        result = await backend.cross_view_consistency(emb_objs, validated_input.view_ids)
        return CrossViewOutput(
            consistency_score=result.consistency_score,
            pairwise_distances=result.pairwise_distances,
            outlier_views=result.outlier_views,
            details=result.details,
        )
