"""Structured logging for the investigation agent."""

import logging
import sys
from typing import Any, Dict

from backend.app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce noise from third-party libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class InvestigationLogger:
    """Attach structured context to investigation logs."""

    def __init__(self, investigation_id: str):
        self.investigation_id = investigation_id
        self.logger = get_logger(f"investigation.{investigation_id[:8]}")

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(f"[{self.investigation_id[:8]}] {msg}", extra=kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(f"[{self.investigation_id[:8]}] {msg}", extra=kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(f"[{self.investigation_id[:8]}] {msg}", extra=kwargs)

    def tool_call(self, tool: str, latency_ms: float, success: bool, **kwargs: Any) -> None:
        status = "OK" if success else "FAIL"
        self.logger.info(
            f"[{self.investigation_id[:8]}] TOOL {tool} [{status}] {latency_ms:.1f}ms",
            extra=kwargs,
        )
