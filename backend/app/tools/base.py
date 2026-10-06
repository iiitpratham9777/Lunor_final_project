"""
Structured Tool System.

Every tool has:
- Typed input / output (Pydantic)
- Validation
- Timeout
- Error handling
- Logging
- No arbitrary execution
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.models.state import ToolCallRecord

logger = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolError(Exception):
    def __init__(self, tool_name: str, message: str, retryable: bool = False):
        self.tool_name = tool_name
        self.message = message
        self.retryable = retryable
        super().__init__(f"[{tool_name}] {message}")


class BaseTool(ABC, Generic[InputT, OutputT]):
    name: str
    description: str
    input_schema: Type[InputT]
    output_schema: Type[OutputT]

    def __init__(self):
        self.settings = get_settings()
        self.timeout = self.settings.TOOL_TIMEOUT_SECONDS
        self.max_retries = self.settings.MAX_TOOL_RETRIES

    async def run(self, raw_input: Dict[str, Any] | InputT) -> OutputT:
        """Execute tool with validation, timeout, retries, and logging."""
        start = time.perf_counter()
        last_error: Optional[Exception] = None

        # Validate input
        try:
            if isinstance(raw_input, dict):
                validated_input = self.input_schema.model_validate(raw_input)
            else:
                validated_input = raw_input
        except ValidationError as e:
            raise ToolError(self.name, f"Invalid input: {e}", retryable=False) from e

        for attempt in range(1, self.max_retries + 2):
            try:
                result = await asyncio.wait_for(
                    self._execute(validated_input),
                    timeout=self.timeout,
                )
                # Validate output
                if not isinstance(result, self.output_schema):
                    result = self.output_schema.model_validate(result)

                latency = (time.perf_counter() - start) * 1000
                logger.info(f"Tool {self.name} succeeded in {latency:.1f}ms (attempt {attempt})")
                return result

            except asyncio.TimeoutError:
                last_error = ToolError(self.name, f"Timeout after {self.timeout}s", retryable=True)
                logger.warning(f"Tool {self.name} timeout (attempt {attempt})")
            except ToolError as e:
                last_error = e
                if not e.retryable:
                    break
                logger.warning(f"Tool {self.name} failed (attempt {attempt}): {e.message}")
            except Exception as e:
                last_error = ToolError(self.name, str(e), retryable=True)
                logger.exception(f"Tool {self.name} unexpected error (attempt {attempt})")

            if attempt <= self.max_retries:
                await asyncio.sleep(0.5 * attempt)

        latency = (time.perf_counter() - start) * 1000
        raise last_error or ToolError(self.name, "Unknown failure")

    def make_record(
        self,
        raw_input: Dict[str, Any],
        output: Optional[OutputT],
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> ToolCallRecord:
        return ToolCallRecord(
            tool_name=self.name,
            input_summary={k: str(v)[:200] for k, v in (raw_input or {}).items()},
            output_summary=output.model_dump() if output else None,
            success=success,
            latency_ms=latency_ms,
            error=error,
        )

    @abstractmethod
    async def _execute(self, validated_input: InputT) -> OutputT:
        """Implement the actual tool logic here."""
        ...


class ToolRegistry:
    """Central registry — only registered tools can be called by the agent."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolError(name, f"Unknown tool: {name}", retryable=False)
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def call(self, name: str, raw_input: Dict[str, Any]) -> tuple[Any, ToolCallRecord]:
        tool = self.get(name)
        start = time.perf_counter()
        try:
            result = await tool.run(raw_input)
            latency = (time.perf_counter() - start) * 1000
            record = tool.make_record(raw_input, result, True, latency)
            return result, record
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            record = tool.make_record(raw_input, None, False, latency, str(e))
            raise
