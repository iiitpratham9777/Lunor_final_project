"""
Multimodal AI Investigation & Anomaly Intelligence Agent — FastAPI entrypoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import investigations
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.models.db import init_db

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.IMAGE_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"DB init skipped/failed: {e}")
    try:
        from backend.app.services.seed import seed_demo_knowledge
        seed_demo_knowledge()
    except Exception as e:
        logger.warning(f"Seed failed: {e}")
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} started")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Stateful Planner–Executor–Verifier agent for multimodal anomaly investigation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"] if settings.DEBUG else settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(investigations.router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "vision_backend": settings.VISION_BACKEND,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.get("/")
async def root():
    return {
        "message": "Multimodal AI Investigation Agent API",
        "docs": "/docs",
        "health": "/health",
        "api": settings.API_PREFIX,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})
