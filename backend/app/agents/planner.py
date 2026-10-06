"""
Planner Node.

1. Understand the investigation goal
2. Identify possible hypotheses
3. Determine what evidence is missing
4. Select appropriate tools
5. Decide the next investigation step
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.models.state import (
    Hypothesis,
    InvestigationState,
    Observation,
    ObservationSource,
)

logger = get_logger(__name__)


class PlanStep(BaseModel):
    action: str  # tool name or "decide" | "reason" | "verify"
    tool_name: Optional[str] = None
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    priority: float = 0.5


class PlannerOutput(BaseModel):
    hypotheses: List[Hypothesis]
    next_steps: List[PlanStep]
    missing_evidence: List[str]
    should_stop: bool = False
    stop_reason: Optional[str] = None


class Planner:
    """
    Deterministic + LLM-assisted planner.
    Prefer structured rules; use LLM only for hypothesis generation when configured.
    """

    def __init__(self):
        self.settings = get_settings()

    async def plan(self, state: InvestigationState) -> PlannerOutput:
        state.add_trace("planner", "start", {"iteration": state.iteration})

        missing: List[str] = []
        next_steps: List[PlanStep] = []
        hypotheses = list(state.hypotheses)

        # --- Bootstrap hypothesis if none ---
        if not hypotheses:
            hyp = self._initial_hypotheses(state)
            hypotheses.extend(hyp)
            state.hypotheses = hypotheses
            if hyp:
                state.current_hypothesis = hyp[0]

        # --- Determine missing evidence ---
        has_visual = any(o.source == ObservationSource.VISION for o in state.observations)
        has_retrieval = len(state.retrieved_context) > 0
        has_cross_view = (
            state.visual_features is not None
            and state.visual_features.cross_view_consistency is not None
        )
        has_anomaly_score = (
            state.visual_features is not None
            and state.visual_features.anomaly_score is not None
        )
        n_images = len(state.input_images)

        if not has_visual and n_images > 0:
            missing.append("visual_features")
            next_steps.append(
                PlanStep(
                    action="tool",
                    tool_name="image_feature_extractor",
                    tool_input={
                        "image_paths": state.input_images,
                        "view_ids": [f"view_{i}" for i in range(n_images)],
                    },
                    rationale="No visual features extracted yet",
                    priority=1.0,
                )
            )

        if has_visual and not has_anomaly_score:
            missing.append("anomaly_score")
            # Will be filled by executor after embeddings exist
            next_steps.append(
                PlanStep(
                    action="tool",
                    tool_name="anomaly_scorer",
                    tool_input={},  # filled by executor from state
                    rationale="Need anomaly score from feature space",
                    priority=0.9,
                )
            )

        if n_images >= 2 and not has_cross_view:
            missing.append("cross_view_consistency")
            next_steps.append(
                PlanStep(
                    action="tool",
                    tool_name="cross_view_comparator",
                    tool_input={},
                    rationale="Multiple views available — check consistency",
                    priority=0.85,
                )
            )

        if not has_retrieval:
            missing.append("retrieved_context")
            next_steps.append(
                PlanStep(
                    action="tool",
                    tool_name="knowledge_retriever",
                    tool_input={
                        "query": state.textual_description
                        or (state.current_hypothesis.statement if state.current_hypothesis else "surface anomaly defect"),
                        "top_k": 5,
                    },
                    rationale="No contextual knowledge retrieved yet",
                    priority=0.7,
                )
            )

        # If we have enough evidence, go to reasoning / verification
        if has_visual and has_anomaly_score and (has_retrieval or state.iteration >= 2):
            next_steps.append(
                PlanStep(
                    action="reason",
                    rationale="Sufficient primary evidence collected — reason over structured evidence",
                    priority=0.6,
                )
            )
            next_steps.append(
                PlanStep(
                    action="verify",
                    rationale="Run independent verification",
                    priority=0.5,
                )
            )

        # Sort by priority
        next_steps.sort(key=lambda s: -s.priority)

        # Stop conditions
        should_stop = False
        stop_reason = None
        if state.iteration >= state.max_iterations:
            should_stop = True
            stop_reason = "max_iterations_reached"
        elif state.confidence.level.value == "high" and state.verification_results:
            last_v = state.verification_results[-1]
            if last_v.passed and not last_v.requires_more_evidence:
                should_stop = True
                stop_reason = "high_confidence_verified"
        # After tools have run and at least one verification, stop collecting
        elif (
            has_visual
            and has_anomaly_score
            and state.verification_results
            and state.iteration >= 2
        ):
            should_stop = True
            stop_reason = "evidence_collected_and_verified"

        if not next_steps and not should_stop:
            next_steps.append(
                PlanStep(action="verify", rationale="No more tools needed — verify", priority=0.5)
            )

        out = PlannerOutput(
            hypotheses=hypotheses,
            next_steps=next_steps[:4],  # limit
            missing_evidence=missing,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )
        state.add_trace("planner", "done", {
            "next_actions": [s.action for s in out.next_steps],
            "missing": missing,
            "should_stop": should_stop,
        })
        return out

    def _initial_hypotheses(self, state: InvestigationState) -> List[Hypothesis]:
        desc = (state.textual_description or "").lower()
        hyps = []
        if any(k in desc for k in ("defect", "anomaly", "scratch", "crack", "damage")):
            hyps.append(
                Hypothesis(
                    statement="The object may contain a surface defect or visual anomaly.",
                    priority=0.8,
                    required_evidence=["local visual features", "reference similarity", "cross-view consistency"],
                )
            )
        hyps.append(
            Hypothesis(
                statement="The object is within normal visual variation (no significant anomaly).",
                priority=0.5,
                required_evidence=["visual features", "reference distribution"],
            )
        )
        if not hyps:
            hyps.append(
                Hypothesis(
                    statement="Unknown — need visual and contextual evidence to form a hypothesis.",
                    priority=0.3,
                    required_evidence=["visual features", "retrieved context"],
                )
            )
        return hyps
