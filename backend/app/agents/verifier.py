"""
Independent Verification Stage.

Checks:
- Does the evidence support the hypothesis?
- Are visual and semantic signals consistent?
- Is retrieved context relevant?
- Is confidence justified?
- Did the agent hallucinate an observation?
- Is more evidence required?
"""

from __future__ import annotations

from typing import List

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.models.state import (
    ConfidenceBreakdown,
    DecisionType,
    InvestigationState,
    ObservationSource,
    VerificationResult,
)

logger = get_logger(__name__)


class Verifier:
    def __init__(self):
        self.settings = get_settings()

    async def verify(self, state: InvestigationState) -> VerificationResult:
        state.add_trace("verifier", "start")
        reasons: List[str] = []
        hallucination_flags: List[str] = []
        requires_more = False
        consistency = 0.5

        # 1. Hallucination check: observations must come from tools / vision
        for obs in state.observations:
            if not obs.is_fact and obs.source == ObservationSource.VISION:
                # Inferences labeled as vision are suspicious
                hallucination_flags.append(
                    f"Observation marked as inference but source=vision: {obs.content[:80]}"
                )

        # 2. Visual evidence present?
        visual_obs = [o for o in state.observations if o.source == ObservationSource.VISION]
        if not visual_obs and state.input_images:
            reasons.append("No visual observations recorded despite input images")
            requires_more = True

        # 3. Anomaly score consistency
        vf = state.visual_features
        anomaly_score = vf.anomaly_score if vf else None
        cross_view = vf.cross_view_consistency if vf else None

        if anomaly_score is not None:
            if anomaly_score > 0.7:
                reasons.append(f"High visual anomaly score ({anomaly_score:.2f})")
            elif anomaly_score < 0.3:
                reasons.append(f"Low visual anomaly score ({anomaly_score:.2f}) suggests normal")
            else:
                reasons.append(f"Ambiguous visual anomaly score ({anomaly_score:.2f})")
                requires_more = True

        # 4. Cross-view consistency
        if cross_view is not None:
            if cross_view < 0.4:
                reasons.append(f"Low cross-view consistency ({cross_view:.2f}) — views disagree")
                requires_more = True
            else:
                reasons.append(f"Cross-view consistency acceptable ({cross_view:.2f})")

        # 5. Retrieval relevance
        if state.retrieved_context:
            avg_rel = sum(c.get("score", 0.0) for c in state.retrieved_context) / len(state.retrieved_context)
            if avg_rel < 0.3:
                reasons.append(f"Retrieved context has low relevance (avg={avg_rel:.2f})")
                requires_more = True
            else:
                reasons.append(f"Retrieved context relevance OK (avg={avg_rel:.2f})")
        else:
            reasons.append("No retrieved context available")
            if state.iteration < 3:
                requires_more = True

        # 6. Confidence calibration
        conf = state.confidence
        conf.recompute()
        if conf.overall > 0.85 and requires_more:
            reasons.append("High confidence but missing evidence — down-weighting")
            conf.overall = min(conf.overall, 0.75)
            conf.recompute()

        # 7. Evidence vs hypothesis support
        support_count = sum(1 for e in state.evidence if e.supports_hypothesis is True)
        oppose_count = sum(1 for e in state.evidence if e.supports_hypothesis is False)
        if support_count + oppose_count > 0:
            consistency = support_count / (support_count + oppose_count)
            reasons.append(f"Evidence support ratio: {support_count}/{support_count + oppose_count}")

        # 8. Conflict detection
        visual_says_anomaly = anomaly_score is not None and anomaly_score > 0.6
        visual_says_normal = anomaly_score is not None and anomaly_score < 0.35
        if visual_says_anomaly and oppose_count > support_count:
            reasons.append("CONFLICT: visual anomaly signal vs opposing evidence")
            requires_more = True
            consistency = min(consistency, 0.4)
        if visual_says_normal and support_count > oppose_count and anomaly_score is not None:
            # opposing would mean "supports anomaly hypothesis"
            pass

        passed = (
            not hallucination_flags
            and not requires_more
            and consistency >= 0.5
            and conf.overall >= self.settings.CONFIDENCE_MODERATE_THRESHOLD
        )

        result = VerificationResult(
            hypothesis_id=state.current_hypothesis.id if state.current_hypothesis else None,
            passed=passed,
            reasons=reasons,
            consistency_score=float(consistency),
            hallucination_flags=hallucination_flags,
            requires_more_evidence=requires_more,
            suggested_next_actions=["collect_more_views", "retrieve_better_context"] if requires_more else [],
        )
        state.verification_results.append(result)
        state.add_trace("verifier", "done", {
            "passed": passed,
            "requires_more": requires_more,
            "consistency": consistency,
        })
        return result

    def decide(self, state: InvestigationState) -> DecisionType:
        """Final decision mapping after verification."""
        conf = state.confidence
        conf.recompute()
        anomaly_score = state.visual_features.anomaly_score if state.visual_features else None

        last_v = state.verification_results[-1] if state.verification_results else None

        if last_v and last_v.requires_more_evidence and state.iteration < state.max_iterations:
            return DecisionType.UNCERTAIN

        if conf.level.value == "low" or conf.overall < self.settings.CONFIDENCE_MODERATE_THRESHOLD:
            state.uncertainty_reason = "Overall confidence below threshold or conflicting evidence"
            return DecisionType.INSUFFICIENT_EVIDENCE

        if anomaly_score is not None:
            if anomaly_score >= 0.6 and conf.overall >= 0.6:
                return DecisionType.ANOMALY
            if anomaly_score <= 0.35 and conf.overall >= 0.6:
                return DecisionType.NORMAL

        # Fallback
        if last_v and not last_v.passed:
            state.uncertainty_reason = "; ".join(last_v.reasons[:3])
            return DecisionType.UNCERTAIN

        return DecisionType.UNCERTAIN
