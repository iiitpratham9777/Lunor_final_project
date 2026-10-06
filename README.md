# Multimodal AI Investigation & Anomaly Intelligence Agent

**Stateful Planner–Executor–Verifier agent** for multimodal anomaly investigation.

This is a portfolio-grade system demonstrating:
AI Agents · Multimodal AI · Computer Vision · RAG / Vector Search · Structured LLM Reasoning · Explicit State Management · Evaluation · FastAPI · Production Reliability

> Not a chatbot wrapper. Not a single-prompt classifier.  
> An investigator: **OBSERVE → HYPOTHESIZE → COLLECT EVIDENCE → VERIFY → REVISE → DECIDE**.

---

## Problem

Classical anomaly detection returns a score. Real investigation requires:

1. Whether an anomaly is present  
2. Where it is  
3. What type it may be  
4. Supporting visual evidence  
5. Agreement with contextual / semantic evidence  
6. Whether more investigation is needed  
7. Calibrated confidence  
8. Explicit uncertainty when evidence conflicts or is insufficient  

The system must **not** hallucinate visual observations or force binary certainty.

---

## Architecture

```
USER → ORCHESTRATOR (state machine)
         ├─ PLANNER        (hypotheses, missing evidence, next tools)
         ├─ EXECUTOR       (typed tools: vision, retrieval, cross-view)
         ├─ REASONER       (structured evidence only — no invented facts)
         └─ VERIFIER       (consistency, hallucination flags, confidence)
                ├─ confident → DECISION
                └─ uncertain / conflict → back to PLANNER (bounded)
```

**State** is a typed Pydantic object (`InvestigationState`).  
The full conversation is **never** dumped into every model call.

### Documented anomaly score

```
AnomalyScore = α · VisualDeviation + β · ReferenceDistance + γ · CrossViewInconsistency
```

Default weights: α=0.45, β=0.35, γ=0.20.  
Feature-space methods: distance-to-mean, max-distance, kNN-distance (cosine).

---

## Quick Start (Demo)

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

export PYTHONPATH=.
python scripts/run_demo.py
```

Expected behaviour on the synthetic anomaly images:
- Extracts multi-view embeddings (OpenCV backend)
- Computes cross-view consistency + feature-space anomaly score
- Retrieves demo knowledge
- Verifies consistency
- Returns **UNCERTAIN** or **INSUFFICIENT_EVIDENCE** when views disagree strongly or confidence is low
- Emits a full investigation trace and structured explanation

### API server

```bash
export PYTHONPATH=.
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

Example flow:

```bash
# Create investigation
curl -X POST http://localhost:8000/api/v1/investigations \
  -H 'Content-Type: application/json' \
  -d '{"textual_description":"Inspect for surface defects"}'

# Upload images
curl -X POST http://localhost:8000/api/v1/investigations/{id}/images \
  -F 'files=@tests/demo_data/anomaly_view1.png' \
  -F 'files=@tests/demo_data/anomaly_view2.png'

# Run agent
curl -X POST http://localhost:8000/api/v1/investigations/{id}/run

# Trace / evidence / report
curl http://localhost:8000/api/v1/investigations/{id}/trace
curl http://localhost:8000/api/v1/investigations/{id}/evidence
curl http://localhost:8000/api/v1/investigations/{id}/report
```

---

## Project Layout

```
backend/
  app/
    agents/          # planner, verifier, orchestrator
    vision/          # interface + OpenCV / mock backends
    retrieval/       # FAISS / numpy vector store + text embedder
    tools/           # typed tools (feature extractor, anomaly scorer, …)
    models/          # Pydantic state + SQLAlchemy ORM
    api/routes/      # FastAPI investigations API
    evaluation/      # metrics + adversarial cases
    services/        # seed knowledge
    core/            # config, logging
frontend/            # Next.js dashboard (scaffold)
tests/demo_data/     # synthetic images
docs/architecture.md
scripts/run_demo.py
docker/
```

---

## Key Design Decisions

| Principle | Implementation |
|-----------|----------------|
| Explicit state | `InvestigationState` Pydantic model |
| No infinite loops | `max_iterations`, stop after evidence+verify |
| Observations ≠ inferences | `Observation.is_fact` flag |
| Independent verification | Separate `Verifier` node |
| Uncertainty first | Decision types include `uncertain` / `insufficient_evidence` |
| Swappable vision | `VisionBackend` ABC + factory |
| Typed tools | Input/output schemas, timeout, retries, logging |
| Retrieval over prompt stuffing | Vector store + compact `summary_for_llm()` |
| Observable | Full `trace`, tool latency, confidence history |

---

## Evaluation

```bash
PYTHONPATH=. python -c "
import asyncio
from pathlib import Path
from backend.app.evaluation.runner import default_adversarial_cases, run_evaluation
async def main():
    report = await run_evaluation(default_adversarial_cases(Path('tests/demo_data')))
    print(report.model_dump())
asyncio.run(main())
"
```

Metrics reported: accuracy / precision / recall / F1 (when labels exist), uncertainty rate, avg iterations, avg latency, avg tool calls.

Adversarial cases deliberately include missing images, conflicting signals, and low-quality hints — the agent is expected to return uncertainty rather than hallucinate.

---

## Limitations (honest)

- Demo uses OpenCV hand-crafted features + hash text embeddings when heavy models are not installed.
- OpenCLIP / DINOv2 / sentence-transformers / FAISS are optional upgrades (see `requirements.txt` comments).
- LLM reasoning is rule-based by default (`LLM_PROVIDER=mock`); plug in OpenAI for richer hypothesis language.
- Frontend is scaffolded; primary demo is API + CLI.
- No claim of SOTA anomaly detection accuracy — the focus is agent architecture, verification, and uncertainty handling.

---

## Resume Bullets

**AI Engineer**  
Designed and implemented a stateful Planner–Executor–Verifier multimodal agent for visual anomaly investigation with explicit confidence calibration, independent verification, and bounded tool-calling loops.

**Software Engineer**  
Built production-style FastAPI services with typed tool contracts, Pydantic state machines, vector retrieval, evaluation harness, and full investigation traces for observability.

**ML Engineer**  
Implemented modular vision backends (feature-space anomaly scoring + cross-view consistency), RAG over investigation history, and an evaluation suite measuring detection metrics, agent efficiency, and uncertainty behaviour.

---

## Interview Talk Tracks

**30 seconds**  
“I built an investigation agent, not a classifier. It plans, calls vision and retrieval tools, verifies consistency between visual and semantic evidence, and returns calibrated confidence or explicit uncertainty instead of forcing a binary label.”

**2 minutes**  
Cover: PEV loop, typed state, observation vs inference, anomaly score formula, verifier checks, stop conditions, demo results on conflicting multi-view case.

**5 minutes**  
Architecture diagram, tool registry, memory (working + long-term vector store), evaluation dimensions, failure modes (missing image, conflict, low quality), what would change for production (real CLIP, GPU batching, Postgres, auth).

---

## License

MIT — portfolio use encouraged.
