# -*- coding: utf-8 -*-

import asyncio
import pytest

from app.core.agent.graph.intent_agent import build_intent_graph


class DummyLLM:
    async def ainvoke(self, messages):
        # Very naive classifier for testing purposes
        user = next((m.get("content") for m in messages if m.get("role") == "user"), "").lower()
        if any(k in user for k in ["搜索", "search", "找一下"]):
            content = '{"intent": "search", "confidence": 0.9}'
        elif any(k in user for k in ["执行", "run", "执行任务", "exec"]):
            content = '{"intent": "exec", "confidence": 0.85}'
        elif any(k in user for k in ["写", "改写", "summarize", "写作", "write"]):
            content = '{"intent": "write", "confidence": 0.8}'
        elif any(k in user for k in ["你好", "hello", "hi"]):
            content = '{"intent": "smalltalk", "confidence": 0.7}'
        else:
            content = '{"intent": "qa", "confidence": 0.75}'

        class R:
            def __init__(self, c):
                self.content = c

        return R(content)


@pytest.mark.asyncio
async def test_intent_detection_basic_cases():
    llm = DummyLLM()
    graph = build_intent_graph(llm)

    async def invoke(messages):
        return await graph.ainvoke({"messages": messages, "session_id": "test-session"})

    # Search
    res = await invoke([{ "role": "user", "content": "帮我搜索下最新的AI新闻" }])
    assert res["intent"] == "search"
    assert 0 <= res["intent_confidence"] <= 1

    # Exec
    res = await invoke([{ "role": "user", "content": "帮我执行一个任务：清理日志" }])
    assert res["intent"] == "exec"

    # Write
    res = await invoke([{ "role": "user", "content": "请帮我写一段总结" }])
    assert res["intent"] == "write"

    # Smalltalk
    res = await invoke([{ "role": "user", "content": "你好" }])
    assert res["intent"] == "smalltalk"

    # QA fallback
    res = await invoke([{ "role": "user", "content": "什么是RAG系统？" }])
    assert res["intent"] in ("qa", "other")




