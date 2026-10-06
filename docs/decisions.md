# Architecture Decision Records (selected)

## ADR-001: Explicit state machine over single mega-prompt
We use a Planner–Executor–Verifier loop with a typed `InvestigationState`.
Rationale: observability, testability, prevention of infinite loops, separation of facts vs inferences.

## ADR-002: OpenCV / mock vision as default
Heavy models (OpenCLIP, DINOv2) are optional. Default backend always runs offline.
Interface is swappable via `VisionBackend` ABC.

## ADR-003: Uncertainty is a first-class outcome
Decision enum includes `uncertain` and `insufficient_evidence`.
Verifier can force more evidence or stop with low confidence.

## ADR-004: Typed tools only
No arbitrary code execution. Every tool has Pydantic I/O, timeout, retries, logging.

## ADR-005: Retrieval over conversation dump
`summary_for_llm()` exposes a compact structured view. Long-term memory lives in the vector store.
