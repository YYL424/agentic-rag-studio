"""Provider-aware structured output tests; no real model calls."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from services.structured_llm import StructuredLLMAdapter, StructuredOutputError


class Output(BaseModel):
    value: str


class FakeRunnable:
    def __init__(self, result=None, error: Exception | None = None, delay: float = 0) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class FakeLLM:
    def __init__(self, runnables: dict[str, FakeRunnable]) -> None:
        self.runnables = runnables
        self.methods = []

    def with_structured_output(self, schema, method):
        self.methods.append(method)
        return self.runnables[method]


@pytest.mark.asyncio
async def test_deepseek_uses_json_mode_and_injects_schema():
    runnable = FakeRunnable(result=Output(value="ok"))
    llm = FakeLLM({"json_mode": runnable})
    adapter = StructuredLLMAdapter(llm, provider_hint="https://api.deepseek.com deepseek-reasoner")

    result = await adapter.invoke(
        Output,
        [SystemMessage(content="system"), HumanMessage(content="hello")],
        purpose="test",
    )

    assert result.value == "ok"
    assert llm.methods == ["json_mode"]
    assert '"value"' in runnable.messages[0].content


@pytest.mark.asyncio
async def test_openai_falls_back_from_function_calling_to_json_mode():
    llm = FakeLLM({
        "function_calling": FakeRunnable(error=ValueError("tools unavailable")),
        "json_mode": FakeRunnable(result=Output(value="fallback")),
    })
    adapter = StructuredLLMAdapter(llm, provider_hint="https://api.openai.com gpt-4o")

    result = await adapter.invoke(Output, [HumanMessage(content="hello")], purpose="test")

    assert result.value == "fallback"
    assert llm.methods == ["function_calling", "json_mode"]


@pytest.mark.asyncio
async def test_structured_output_timeout_is_bounded():
    llm = FakeLLM({"json_mode": FakeRunnable(result=Output(value="late"), delay=0.05)})
    adapter = StructuredLLMAdapter(
        llm,
        method="json_mode",
        timeout_seconds=0.001,
        provider_hint="deepseek",
    )

    with pytest.raises(StructuredOutputError, match="test failed"):
        await adapter.invoke(Output, [HumanMessage(content="hello")], purpose="test")
