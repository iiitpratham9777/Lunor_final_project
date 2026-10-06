"""
Typed Investigation State — the single source of truth for the agent.
Never pass the entire conversation history to every model call.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DecisionType(str, Enum):
    ANOMALY = "anomaly"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConfidenceLevel(str, Enum):
    HIGH = "high"          # > 0.85
    MODERATE = "moderate"  # 0.60 – 0.85
    LOW = "low"            # < 0.60


class ObservationSource(str, Enum):
    VISION = "vision"
    RETRIEVAL = "retrieval"
    METADATA = "metadata"
    CROSS_VIEW = "cross_view"
    USER = "user"
    TOOL = "tool"


class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: ObservationSource
    content: str
    raw_data: Optional[Dict[str, Any]] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_fact: bool = True  # True = observed fact, False = inference


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    observation_ids: List[str] = Field(default_factory=list)
    description: str
    supports_hypothesis: Optional[bool] = None
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    required_evidence: List[str] = Field(default_factory=list)
    status: str = "open"  # open | supported | rejected | revised
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToolCallRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Optional[Dict[str, Any]] = None
    success: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VerificationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    hypothesis_id: Optional[str] = None
    passed: bool
    reasons: List[str] = Field(default_factory=list)
    consistency_score: float = Field(ge=0.0, le=1.0, default=0.0)
    hallucination_flags: List[str] = Field(default_factory=list)
    requires_more_evidence: bool = False
    suggested_next_actions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VisualFeatures(BaseModel):
    global_embedding: Optional[List[float]] = None
    patch_embeddings: Optional[List[List[float]]] = None
    anomaly_score: Optional[float] = None
    anomaly_map: Optional[List[List[float]]] = None  # spatial heatmap
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    cross_view_consistency: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceBreakdown(BaseModel):
    visual: float = Field(ge=0.0, le=1.0, default=0.0)
    semantic: float = Field(ge=0.0, le=1.0, default=0.0)
    cross_view: float = Field(ge=0.0, le=1.0, default=0.0)
    retrieval_relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    overall: float = Field(ge=0.0, le=1.0, default=0.0)
    level: ConfidenceLevel = ConfidenceLevel.LOW

    def recompute(self) -> None:
        # Weighted combination — documented formula
        self.overall = (
            0.40 * self.visual
            + 0.25 * self.semantic
            + 0.20 * self.cross_view
            + 0.15 * self.retrieval_relevance
        )
        if self.overall > 0.85:
            self.level = ConfidenceLevel.HIGH
        elif self.overall >= 0.60:
            self.level = ConfidenceLevel.MODERATE
        else:
            self.level = ConfidenceLevel.LOW


class InvestigationState(BaseModel):
    """Full investigation state. This is the LangGraph / state-machine state."""

    investigation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Input
    input_images: List[str] = Field(default_factory=list)  # paths or URLs
    textual_description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Core investigation
    current_hypothesis: Optional[Hypothesis] = None
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    visual_features: Optional[VisualFeatures] = None
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)

    # Agent control
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 8
    status: str = "initialized"  # initialized | planning | investigating | verifying | decided | failed

    # Decision
    final_decision: Optional[DecisionType] = None
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    uncertainty_reason: Optional[str] = None
    explanation: Optional[str] = None

    # Observability
    latency: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)
        self.updated_at = datetime.utcnow()

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        self.updated_at = datetime.utcnow()

    def add_tool_call(self, tc: ToolCallRecord) -> None:
        self.tool_calls.append(tc)
        self.updated_at = datetime.utcnow()

    def add_trace(self, node: str, action: str, details: Optional[Dict] = None) -> None:
        self.trace.append({
            "iteration": self.iteration,
            "node": node,
            "action": action,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        })

    def can_continue(self) -> bool:
        return (
            self.iteration < self.max_iterations
            and self.status not in ("decided", "failed")
            and len(self.errors) < 5
        )

    def summary_for_llm(self) -> Dict[str, Any]:
        """Structured, compact view for LLM — never the full history dump."""
        return {
            "investigation_id": self.investigation_id,
            "iteration": self.iteration,
            "status": self.status,
            "current_hypothesis": self.current_hypothesis.statement if self.current_hypothesis else None,
            "observations": [
                {"source": o.source, "content": o.content, "is_fact": o.is_fact, "confidence": o.confidence}
                for o in self.observations[-12:]  # recent only
            ],
            "evidence_count": len(self.evidence),
            "evidence_summary": [
                {"description": e.description, "supports": e.supports_hypothesis, "strength": e.strength}
                for e in self.evidence[-8:]
            ],
            "visual_anomaly_score": self.visual_features.anomaly_score if self.visual_features else None,
            "cross_view_consistency": self.visual_features.cross_view_consistency if self.visual_features else None,
            "retrieved_context_count": len(self.retrieved_context),
            "confidence": self.confidence.model_dump(),
            "last_verification": self.verification_results[-1].model_dump() if self.verification_results else None,
            "errors": self.errors[-3:],
        }


class InvestigationCreate(BaseModel):
    textual_description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvestigationResponse(BaseModel):
    investigation_id: str
    status: str
    final_decision: Optional[DecisionType] = None
    confidence: Optional[ConfidenceBreakdown] = None
    uncertainty_reason: Optional[str] = None
    explanation: Optional[str] = None
    iteration: int
    trace_length: int
    created_at: datetime
    updated_at: datetime
