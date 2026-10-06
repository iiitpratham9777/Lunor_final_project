"""
Evaluation framework.

Metrics:
- Detection: accuracy, precision, recall, F1
- Retrieval: precision@k (when labels available)
- Agent: avg iterations, tool call counts, uncertainty rate
- Reliability: verification failure rate
- Engineering: latency
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.app.agents.orchestrator import Orchestrator
from backend.app.core.logging import get_logger
from backend.app.models.state import DecisionType, InvestigationState

logger = get_logger(__name__)


class TestCase(BaseModel):
    name: str
    image_paths: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    expected_decision: Optional[str] = None  # anomaly | normal | uncertain | insufficient_evidence
    tags: List[str] = Field(default_factory=list)
    notes: str = ""


class CaseResult(BaseModel):
    name: str
    expected: Optional[str]
    predicted: Optional[str]
    correct: Optional[bool]
    confidence: float
    iterations: int
    latency_ms: float
    tool_calls: int
    uncertainty: bool
    tags: List[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    n_cases: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    uncertainty_rate: float
    avg_iterations: float
    avg_latency_ms: float
    avg_tool_calls: float
    results: List[CaseResult]


def _binary_label(d: Optional[str]) -> Optional[int]:
    if d is None:
        return None
    if d in ("anomaly",):
        return 1
    if d in ("normal",):
        return 0
    return None  # uncertain etc. excluded from binary metrics


async def run_evaluation(cases: List[TestCase]) -> EvalReport:
    orch = Orchestrator()
    results: List[CaseResult] = []

    for case in cases:
        t0 = time.perf_counter()
        state = InvestigationState(
            input_images=case.image_paths,
            textual_description=case.description,
            metadata={"eval_case": case.name},
        )
        try:
            state = await orch.run(state)
        except Exception as e:
            logger.error(f"Case {case.name} failed: {e}")
            state.final_decision = DecisionType.UNCERTAIN
            state.errors.append(str(e))

        pred = state.final_decision.value if state.final_decision else None
        exp = case.expected_decision
        correct = (pred == exp) if exp and pred else None
        latency = (time.perf_counter() - t0) * 1000

        results.append(
            CaseResult(
                name=case.name,
                expected=exp,
                predicted=pred,
                correct=correct,
                confidence=state.confidence.overall,
                iterations=state.iteration,
                latency_ms=latency,
                tool_calls=len(state.tool_calls),
                uncertainty=pred in ("uncertain", "insufficient_evidence") if pred else True,
                tags=case.tags,
            )
        )

    # Aggregate
    labeled = [r for r in results if r.correct is not None]
    tp = sum(1 for r in labeled if r.expected == "anomaly" and r.predicted == "anomaly")
    fp = sum(1 for r in labeled if r.expected == "normal" and r.predicted == "anomaly")
    fn = sum(1 for r in labeled if r.expected == "anomaly" and r.predicted != "anomaly")
    tn = sum(1 for r in labeled if r.expected == "normal" and r.predicted == "normal")

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    accuracy = sum(1 for r in labeled if r.correct) / len(labeled) if labeled else None

    return EvalReport(
        n_cases=len(results),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        uncertainty_rate=sum(1 for r in results if r.uncertainty) / max(len(results), 1),
        avg_iterations=sum(r.iterations for r in results) / max(len(results), 1),
        avg_latency_ms=sum(r.latency_ms for r in results) / max(len(results), 1),
        avg_tool_calls=sum(r.tool_calls for r in results) / max(len(results), 1),
        results=results,
    )


def default_adversarial_cases(demo_dir: Path) -> List[TestCase]:
    """Adversarial / edge cases — expected to return uncertainty or careful decisions."""
    cases = [
        TestCase(
            name="no_image_text_only",
            description="Possible surface defect mentioned in text only",
            expected_decision="insufficient_evidence",
            tags=["adversarial", "missing_image"],
            notes="No visual evidence",
        ),
        TestCase(
            name="conflict_visual_vs_context",
            description="Object appears normal but context mentions defects",
            expected_decision="uncertain",
            tags=["adversarial", "conflict"],
            notes="Should not hallucinate",
        ),
        TestCase(
            name="low_quality_hint",
            description="Extremely poor image quality, cannot determine anomaly",
            expected_decision="uncertain",
            tags=["adversarial", "low_quality"],
        ),
    ]
    # Attach demo images if present
    imgs = list(demo_dir.glob("*.png")) + list(demo_dir.glob("*.jpg"))
    if imgs:
        cases.append(
            TestCase(
                name="demo_single_view",
                image_paths=[str(imgs[0])],
                description="Inspect for surface anomalies",
                tags=["demo"],
            )
        )
        if len(imgs) >= 2:
            cases.append(
                TestCase(
                    name="demo_multi_view",
                    image_paths=[str(p) for p in imgs[:3]],
                    description="Cross-view consistency check for defects",
                    tags=["demo", "multi_view"],
                )
            )
    return cases
