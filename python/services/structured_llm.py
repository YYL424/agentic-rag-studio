"""Provider-aware Pydantic structured output for OpenAI-compatible chat models."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    """Raised after every compatible structured-output strategy has failed."""


class StructuredLLMAdapter:
    """Select a structured-output strategy by provider and enforce a timeout."""

    def __init__(
        self,
        llm: Any,
        *,
        method: str | None = None,
        timeout_seconds: float | None = None,
        provider_hint: str | None = None,
    ) -> None:
        self.llm = llm
        self.method = method or settings.structured_output_method
        self.timeout_seconds = timeout_seconds or settings.structured_output_timeout_seconds
        self.provider_hint = (
            provider_hint or f"{settings.openai_base_url} {settings.openai_model}"
        ).lower()

    def methods(self) -> tuple[str, ...]:
        if self.method != "auto":
            return (self.method,)
        if "deepseek" in self.provider_hint:
            return ("json_mode",)
        return ("function_calling", "json_mode")

    async def invoke(
        self,
        schema: type[BaseModel],
        messages: list[BaseMessage],
        *,
        purpose: str,
    ) -> BaseModel:
        errors: list[str] = []
        for method in self.methods():
            started = time.perf_counter()
            try:
                structured = self.llm.with_structured_output(schema, method=method)
                request_messages = self._with_json_schema(messages, schema) if method == "json_mode" else messages
                async with asyncio.timeout(self.timeout_seconds):
                    result = await structured.ainvoke(request_messages)
                logger.info(
                    "structured output purpose=%s method=%s elapsed_ms=%.1f",
                    purpose,
                    method,
                    (time.perf_counter() - started) * 1000,
                )
                return result
            except Exception as exc:
                detail = f"{method}: {str(exc)[:160]}"
                errors.append(detail)
                logger.warning("structured output purpose=%s failed (%s)", purpose, detail)
        raise StructuredOutputError(f"{purpose} failed: {'; '.join(errors)}")

    @staticmethod
    def _with_json_schema(
        messages: list[BaseMessage],
        schema: type[BaseModel],
    ) -> list[BaseMessage]:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        suffix = (
            "\n\n请只返回一个符合以下 JSON Schema 的 json object，"
            "不要输出 Markdown、解释或思考过程：\n"
            f"{schema_json}"
        )
        output: list[BaseMessage] = []
        injected = False
        for message in messages:
            if isinstance(message, SystemMessage) and not injected:
                output.append(SystemMessage(content=f"{message.content}{suffix}"))
                injected = True
            else:
                output.append(message)
        if not injected:
            output.insert(0, SystemMessage(content=suffix.lstrip()))
        return output
