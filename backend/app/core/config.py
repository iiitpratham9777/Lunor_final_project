"""Configuration management — all secrets via environment variables."""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Multimodal AI Investigation Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # API
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:////tmp/maiia_data/investigations.db"

    # LLM
    LLM_PROVIDER: str = "openai"  # openai | mock | local
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: Optional[str] = None
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 2

    # Vision
    VISION_BACKEND: str = "openclip"  # openclip | mock | dinov2
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "openai"
    MAX_IMAGE_SIZE_MB: float = 10.0
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    IMAGE_UPLOAD_DIR: str = "/tmp/maiia_data/uploads"

    # Retrieval / Vector
    VECTOR_BACKEND: str = "faiss"  # faiss | chroma
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "/tmp/maiia_data/faiss_index"
    TOP_K_RETRIEVAL: int = 5

    # Agent limits
    MAX_AGENT_ITERATIONS: int = 8
    MAX_TOOL_RETRIES: int = 2
    TOOL_TIMEOUT_SECONDS: float = 15.0
    CONFIDENCE_HIGH_THRESHOLD: float = 0.85
    CONFIDENCE_MODERATE_THRESHOLD: float = 0.60

    # Security
    API_KEY: Optional[str] = None  # simple header auth if set
    RATE_LIMIT_PER_MINUTE: int = 60

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
