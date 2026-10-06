"""
Orchestrator — explicit state-machine / LangGraph-style loop.

INPUT → UNDERSTAND → PLAN → INVESTIGATE → RETRIEVE → REASON → VERIFY → DECIDE → EXPLAIN → LEARN
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.app.agents.planner import Planner, PlanStep
from backend.app.agents.verifier import Verifier
from backend.app.core.config import get_settings
from backend.app.core.logging import InvestigationLogger, get_logger
from backend.app.models.state import (
    ConfidenceBreakdown,
    DecisionType,
    Evidence,
    InvestigationState,
    Observation,
    ObservationSource,
    VisualFeatures,
)
from backend.app.tools.base import ToolRegistry
from backend.app.tools.retrieval_tools import KnowledgeRetriever, SimilaritySearchTool
from backend.app.tools.vision_tools import AnomalyScorer, CrossViewComparator, ImageFeatureExtractor

logger = get_logger(__name__)


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ImageFeatureExtractor())
    reg.register(AnomalyScorer())
    reg.register(CrossViewComparator())
    reg.register(KnowledgeRetriever())
    reg.register(SimilaritySearchTool())
    return reg


class Orchestrator:
    """
    Stateful Planner–Executor–Verifier loop.
    Hard iteration limit, timeouts, no infinite loops.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.settings = get_settings()
        self.planner = Planner()
        self.verifier = Verifier()
        self.tools = tool_registry or build_tool_registry()
        self._embeddings_cache: Dict[str, List[float]] = {}

    async def run(self, state: InvestigationState) -> InvestigationState:
        ilog = InvestigationLogger(state.investigation_id)
        ilog.info(f"Starting investigation (images={len(state.input_images)})")
        state.status = "planning"
        state.max_iterations = self.settings.MAX_AGENT_ITERATIONS
        t0 = time.perf_counter()

        try:
            while state.can_continue():
                state.iteration += 1
                ilog.info(f"Iteration {state.iteration}")

                # ---- PLAN ----
                state.status = "planning"
                plan = await self.planner.plan(state)

                if plan.should_stop:
                    ilog.info(f"Planner requested stop: {plan.stop_reason}")
                    break

                if not plan.next_steps:
                    break

                # ---- EXECUTE steps ----
                state.status = "investigating"
                for step in plan.next_steps:
                    if not state.can_continue():
                        break
                    await self._execute_step(state, step, ilog)

                # ---- VERIFY ----
                state.status = "verifying"
                vresult = await self.verifier.verify(state)

                if vresult.passed and not vresult.requires_more_evidence:
                    break
                if state.iteration >= state.max_iterations:
                    break

            # ---- DECIDE ----
            state.status = "decided"
            decision = self.verifier.decide(state)
            state.final_decision = decision

            # ---- EXPLAIN ----
            state.explanation = self._build_explanation(state)

            # ---- LEARN (store embeddings / summary for future retrieval) ----
            await self._learn(state)

        except Exception as e:
            logger.exception(f"Orchestrator failure: {e}")
            state.errors.append(str(e))
            state.status = "failed"
            state.final_decision = DecisionType.UNCERTAIN
            state.uncertainty_reason = f"Internal error: {e}"

        state.latency["total_ms"] = (time.perf_counter() - t0) * 1000
        state.updated_at = state.updated_at  # touch
        ilog.info(
            f"Finished: decision={state.final_decision}, "
            f"confidence={state.confidence.overall:.2f}, "
            f"iterations={state.iteration}"
        )
        return state

    async def _execute_step(
        self,
        state: InvestigationState,
        step: PlanStep,
        ilog: InvestigationLogger,
    ) -> None:
        if step.action == "tool" and step.tool_name:
            await self._run_tool(state, step, ilog)
        elif step.action == "reason":
            await self._reason(state, ilog)
        elif step.action == "verify":
            # already handled in main loop
            pass
        else:
            state.add_trace("executor", "unknown_action", {"action": step.action})

    async def _run_tool(
        self,
        state: InvestigationState,
        step: PlanStep,
        ilog: InvestigationLogger,
    ) -> None:
        tool_name = step.tool_name
        tool_input = dict(step.tool_input)

        # Fill inputs from state when empty
        if tool_name == "anomaly_scorer" and not tool_input.get("query_embedding"):
            if state.visual_features and state.visual_features.global_embedding:
                tool_input["query_embedding"] = state.visual_features.global_embedding
                # Reference: use other views or empty
                tool_input["reference_embeddings"] = []
                if state.visual_features.cross_view_consistency is not None:
                    tool_input["cross_view_inconsistency"] = 1.0 - state.visual_features.cross_view_consistency

        if tool_name == "cross_view_comparator" and not tool_input.get("embeddings"):
            if state.visual_features and state.visual_features.global_embedding:
                # For multi-view we need the list — stored in cache
                embs = list(self._embeddings_cache.values()) or [state.visual_features.global_embedding]
                tool_input["embeddings"] = embs
                tool_input["view_ids"] = list(self._embeddings_cache.keys()) or ["view_0"]

        try:
            result, record = await self.tools.call(tool_name, tool_input)
            state.add_tool_call(record)
            ilog.tool_call(tool_name, record.latency_ms, True)
            self._ingest_tool_result(state, tool_name, result)
        except Exception as e:
            state.errors.append(f"{tool_name}: {e}")
            state.add_trace("executor", "tool_error", {"tool": tool_name, "error": str(e)})
            ilog.tool_call(tool_name, 0, False)

    def _ingest_tool_result(self, state: InvestigationState, tool_name: str, result: Any) -> None:
        if tool_name == "image_feature_extractor":
            embs = result.embeddings
            view_ids = result.view_ids
            for vid, emb in zip(view_ids, embs):
                self._embeddings_cache[vid] = emb
            # Store primary embedding
            primary = embs[0] if embs else None
            if primary:
                if state.visual_features is None:
                    state.visual_features = VisualFeatures()
                state.visual_features.global_embedding = primary
                state.add_observation(
                    Observation(
                        source=ObservationSource.VISION,
                        content=f"Extracted {len(embs)} embedding(s) with model={result.model_name}, dim={result.dims[0] if result.dims else '?'}",
                        is_fact=True,
                        confidence=0.95,
                        raw_data={"n_views": len(embs), "model": result.model_name},
                    )
                )

        elif tool_name == "anomaly_scorer":
            if state.visual_features is None:
                state.visual_features = VisualFeatures()
            score = result.combined_score if result.combined_score is not None else result.score
            state.visual_features.anomaly_score = score
            state.confidence.visual = 1.0 - abs(score - 0.5) * 0.5  # crude
            state.add_observation(
                Observation(
                    source=ObservationSource.VISION,
                    content=f"Anomaly score={score:.3f} (method={result.method})",
                    is_fact=True,
                    confidence=0.9,
                    raw_data=result.details,
                )
            )
            state.add_evidence(
                Evidence(
                    description=f"Feature-space anomaly score {score:.3f}",
                    supports_hypothesis=score > 0.55,
                    strength=min(1.0, abs(score - 0.5) * 2),
                    source="anomaly_scorer",
                )
            )

        elif tool_name == "cross_view_comparator":
            if state.visual_features is None:
                state.visual_features = VisualFeatures()
            state.visual_features.cross_view_consistency = result.consistency_score
            state.confidence.cross_view = result.consistency_score
            state.add_observation(
                Observation(
                    source=ObservationSource.CROSS_VIEW,
                    content=f"Cross-view consistency={result.consistency_score:.3f}; outliers={result.outlier_views}",
                    is_fact=True,
                    confidence=0.9,
                )
            )
            state.add_evidence(
                Evidence(
                    description=f"Cross-view consistency {result.consistency_score:.3f}",
                    supports_hypothesis=result.consistency_score < 0.5,  # low consistency supports anomaly hyp
                    strength=1.0 - result.consistency_score,
                    source="cross_view_comparator",
                )
            )

        elif tool_name == "knowledge_retriever":
            items = result.items
            state.retrieved_context = items
            avg_score = sum(i.get("score", 0) for i in items) / max(len(items), 1)
            state.confidence.semantic = min(1.0, avg_score + 0.2)
            state.confidence.retrieval_relevance = avg_score
            for it in items[:3]:
                state.add_observation(
                    Observation(
                        source=ObservationSource.RETRIEVAL,
                        content=f"Retrieved: {it.get('content', '')[:120]}",
                        is_fact=True,
                        confidence=float(it.get("score", 0.5)),
                        raw_data=it,
                    )
                )
            if items:
                state.add_evidence(
                    Evidence(
                        description=f"Retrieved {len(items)} contextual items (avg relevance={avg_score:.2f})",
                        supports_hypothesis=None,
                        strength=avg_score,
                        source="knowledge_retriever",
                    )
                )

    async def _reason(self, state: InvestigationState, ilog: InvestigationLogger) -> None:
        """
        Structured reasoning over evidence.
        Never invent visual observations — only reason over what is already in state.
        """
        state.add_trace("reasoning", "start")
        conf = state.confidence
        conf.recompute()

        summary_parts = []
        if state.visual_features and state.visual_features.anomaly_score is not None:
            summary_parts.append(f"Visual anomaly score: {state.visual_features.anomaly_score:.3f}")
        if state.visual_features and state.visual_features.cross_view_consistency is not None:
            summary_parts.append(f"Cross-view consistency: {state.visual_features.cross_view_consistency:.3f}")
        summary_parts.append(f"Evidence count: {len(state.evidence)}")
        summary_parts.append(f"Retrieved items: {len(state.retrieved_context)}")

        # Simple rule-based inference (LLM can be plugged in when API key present)
        inference = " | ".join(summary_parts)
        state.add_observation(
            Observation(
                source=ObservationSource.TOOL,
                content=f"Reasoning summary: {inference}",
                is_fact=False,  # this is inference
                confidence=conf.overall,
            )
        )
        state.add_trace("reasoning", "done", {"summary": inference})
        ilog.info(f"Reasoned: {inference[:100]}")

    def _build_explanation(self, state: InvestigationState) -> str:
        lines = ["## Why did the AI decide this?", ""]
        lines.append("### Observed Facts")
        for o in state.observations:
            if o.is_fact:
                lines.append(f"- [{o.source.value}] {o.content}")
        lines.append("")
        lines.append("### Evidence")
        for e in state.evidence:
            support = {True: "supports", False: "opposes", None: "neutral"}[e.supports_hypothesis]
            lines.append(f"- ({support}, strength={e.strength:.2f}) {e.description}")
        lines.append("")
        if state.current_hypothesis:
            lines.append(f"### Hypothesis\n{state.current_hypothesis.statement}")
        lines.append("")
        if state.verification_results:
            v = state.verification_results[-1]
            lines.append("### Verification")
            lines.append(f"- Passed: {v.passed}")
            for r in v.reasons:
                lines.append(f"- {r}")
        lines.append("")
        lines.append("### Confidence")
        c = state.confidence
        lines.append(f"- Visual: {c.visual:.2f}")
        lines.append(f"- Semantic: {c.semantic:.2f}")
        lines.append(f"- Cross-view: {c.cross_view:.2f}")
        lines.append(f"- Retrieval relevance: {c.retrieval_relevance:.2f}")
        lines.append(f"- Overall: {c.overall:.2f} ({c.level.value})")
        lines.append("")
        lines.append(f"### Decision\n{state.final_decision.value if state.final_decision else 'pending'}")
        if state.uncertainty_reason:
            lines.append(f"\nUncertainty reason: {state.uncertainty_reason}")
        return "\n".join(lines)

    async def _learn(self, state: InvestigationState) -> None:
        """Persist investigation summary into long-term memory (vector store)."""
        try:
            from backend.app.retrieval.store import get_text_embedder, get_vector_store

            store = get_vector_store()
            embedder = get_text_embedder()
            summary = (
                f"Investigation {state.investigation_id}: "
                f"decision={state.final_decision}, "
                f"confidence={state.confidence.overall:.2f}, "
                f"hypothesis={state.current_hypothesis.statement if state.current_hypothesis else 'n/a'}"
            )
            emb = embedder.embed(summary)
            store.add(
                emb,
                summary,
                metadata={
                    "source_type": "investigation",
                    "investigation_id": state.investigation_id,
                    "decision": str(state.final_decision),
                },
            )
            # Also store image embeddings if present
            if state.visual_features and state.visual_features.global_embedding:
                store.add(
                    state.visual_features.global_embedding,
                    f"Image embedding from {state.investigation_id}",
                    metadata={
                        "source_type": "image",
                        "investigation_id": state.investigation_id,
                        "anomaly_score": state.visual_features.anomaly_score,
                    },
                )
        except Exception as e:
            logger.warning(f"Learn step failed (non-fatal): {e}")
