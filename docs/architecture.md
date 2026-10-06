# Multimodal AI Investigation & Anomaly Intelligence Agent

## Architecture Overview

**Stateful Planner–Executor–Verifier (PEV) Architecture**

```
USER INPUT (images + optional text/metadata)
        │
        ▼
┌───────────────────┐
│   ORCHESTRATOR    │  ← LangGraph / explicit state machine
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌──────────┐
│ PLANNER│  │  MEMORY  │
└───┬────┘  └────┬─────┘
    │            │
    ▼            ▼
┌─────────────────────────────────────┐
│           EXECUTOR                  │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ Vision  │ │Retrieval│ │ Tools  │ │
│  └─────────┘ └─────────┘ └────────┘ │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌───────────────────┐
│     VERIFIER      │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 CONFIDENT   UNCERTAIN / CONFLICT
    │           │
    ▼           ▼
 DECISION   → back to PLANNER
```

### Core Loop
OBSERVE → FORM HYPOTHESIS → COLLECT EVIDENCE → VERIFY → REVISE → DECIDE

### Design Principles
1. Explicit state (Pydantic) — never dump full conversation into every LLM call
2. Observations ≠ Inferences ≠ Uncertainties
3. Independent Verifier stage
4. Confidence-calibrated decisions + explicit uncertainty
5. Modular vision & retrieval interfaces (swappable models)
6. Full observability (trace, latency, tool calls, confidence history)
7. Hard limits on iterations / retries / timeouts
8. Deterministic tools preferred over LLM where possible

### Technology Choices (Prototype)
- Agent orchestration: LangGraph (or pure Python state machine fallback)
- Vision: OpenCLIP / DINOv2-style embeddings + OpenCV + feature-space scoring
- Retrieval: FAISS + sentence-transformers
- LLM reasoning: OpenAI-compatible or local (configurable)
- Backend: FastAPI + Pydantic + SQLAlchemy
- DB: SQLite (dev) / PostgreSQL (prod)
- Frontend: Next.js 14 + TypeScript + Tailwind + Recharts
- Vector store: FAISS (local) with optional Chroma/Qdrant adapters
