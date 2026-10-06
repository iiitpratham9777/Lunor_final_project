"""Investigation API routes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.agents.orchestrator import Orchestrator
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.models.state import (
    DecisionType,
    InvestigationCreate,
    InvestigationResponse,
    InvestigationState,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/investigations", tags=["investigations"])

# In-memory store for demo (DB optional)
_INVESTIGATIONS: Dict[str, InvestigationState] = {}
_orchestrator = Orchestrator()


class RunRequest(BaseModel):
    force: bool = False


@router.post("", response_model=InvestigationResponse)
async def create_investigation(body: InvestigationCreate):
    state = InvestigationState(
        textual_description=body.textual_description,
        metadata=body.metadata or {},
    )
    _INVESTIGATIONS[state.investigation_id] = state
    return _to_response(state)


@router.post("/{investigation_id}/images")
async def upload_images(
    investigation_id: str,
    files: List[UploadFile] = File(...),
):
    state = _get(investigation_id)
    settings = get_settings()
    upload_dir = Path(settings.IMAGE_UPLOAD_DIR) / investigation_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        # Validate
        if f.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"Unsupported type: {f.content_type}")
        content = await f.read()
        if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, "File too large")
        ext = Path(f.filename or "img.jpg").suffix or ".jpg"
        path = upload_dir / f"{uuid.uuid4().hex}{ext}"
        path.write_bytes(content)
        state.input_images.append(str(path))
        saved.append({"path": str(path), "filename": f.filename})

    _INVESTIGATIONS[investigation_id] = state
    return {"uploaded": saved, "total_images": len(state.input_images)}


@router.post("/{investigation_id}/run", response_model=InvestigationResponse)
async def run_investigation(investigation_id: str, body: RunRequest = RunRequest()):
    state = _get(investigation_id)
    if not state.input_images and not state.textual_description:
        raise HTTPException(400, "Provide at least one image or textual description")
    if state.status == "decided" and not body.force:
        return _to_response(state)

    state = await _orchestrator.run(state)
    _INVESTIGATIONS[investigation_id] = state
    return _to_response(state)


@router.get("/{investigation_id}", response_model=InvestigationResponse)
async def get_investigation(investigation_id: str):
    return _to_response(_get(investigation_id))


@router.get("/{investigation_id}/trace")
async def get_trace(investigation_id: str):
    state = _get(investigation_id)
    return {
        "investigation_id": investigation_id,
        "iteration": state.iteration,
        "status": state.status,
        "trace": state.trace,
        "tool_calls": [t.model_dump() for t in state.tool_calls],
        "errors": state.errors,
        "latency": state.latency,
    }


@router.get("/{investigation_id}/evidence")
async def get_evidence(investigation_id: str):
    state = _get(investigation_id)
    return {
        "observations": [o.model_dump() for o in state.observations],
        "evidence": [e.model_dump() for e in state.evidence],
        "retrieved_context": state.retrieved_context,
        "visual_features": state.visual_features.model_dump() if state.visual_features else None,
        "confidence": state.confidence.model_dump(),
        "hypotheses": [h.model_dump() for h in state.hypotheses],
        "verification_results": [v.model_dump() for v in state.verification_results],
    }


@router.get("/{investigation_id}/report")
async def get_report(investigation_id: str):
    state = _get(investigation_id)
    return {
        "investigation_id": investigation_id,
        "decision": state.final_decision,
        "confidence": state.confidence.model_dump(),
        "uncertainty_reason": state.uncertainty_reason,
        "explanation": state.explanation,
        "iterations": state.iteration,
        "latency_ms": state.latency.get("total_ms"),
        "trace_summary": [
            {"node": t["node"], "action": t["action"], "iteration": t["iteration"]}
            for t in state.trace
        ],
    }


@router.get("")
async def list_investigations():
    return [
        {
            "investigation_id": s.investigation_id,
            "status": s.status,
            "decision": s.final_decision,
            "confidence": s.confidence.overall,
            "images": len(s.input_images),
            "created_at": s.created_at.isoformat(),
        }
        for s in _INVESTIGATIONS.values()
    ]


def _get(investigation_id: str) -> InvestigationState:
    if investigation_id not in _INVESTIGATIONS:
        raise HTTPException(404, "Investigation not found")
    return _INVESTIGATIONS[investigation_id]


def _to_response(state: InvestigationState) -> InvestigationResponse:
    return InvestigationResponse(
        investigation_id=state.investigation_id,
        status=state.status,
        final_decision=state.final_decision,
        confidence=state.confidence,
        uncertainty_reason=state.uncertainty_reason,
        explanation=state.explanation,
        iteration=state.iteration,
        trace_length=len(state.trace),
        created_at=state.created_at,
        updated_at=state.updated_at,
    )
