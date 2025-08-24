# -*- coding: utf-8 -*-

import pytest

from app.core.agent.graph.intent_agent import build_intent_graph
from app.core.agent.graph.entity_agent import build_entity_graph
from app.core.agent.graph.context_agent import build_context_graph
from app.core.agent.graph.rewrite_agent import build_rewrite_graph
from app.core.agent.graph import rag_agent as rag_module


class DummyLLM:
    async def ainvoke(self, messages):
        # Route by simple heuristics based on system prompts / user text
        system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
        user = next((m.get("content") for m in messages if m.get("role") == "user"), "")

        # Intent classifier
        if "intent classifier" in system:
            content = '{"intent": "qa", "confidence": 0.9}'
            return type("R", (), {"content": content})

        # Entity extraction
        if "information extraction" in system:
            if "上周收到的账单" in user:
                content = (
                    '{"entities": {"subject": "账单", "filters": {"action": "收到"}}, "time_text": "上周"}'
                )
            elif "上个月的呢" in user or "上个月" in user:
                content = '{"entities": {}, "time_text": "上个月"}'
            else:
                content = '{"entities": {}, "time_text": null}'
            return type("R", (), {"content": content})

        # Query rewrite
        if "查询改写助手" in system:
            return type("R", (), {"content": "请检索: 账单 在给定时间范围内"})

        # Default
        return type("R", (), {"content": ""})


class DummyRetriever:
    def __init__(self, query: str) -> None:
        self.query = query

    def multi_query(self, to_expand_to_n_queries: int = 3, stream: bool | None = False):
        return [self.query]

    def retrieve_top_k(self, k: int, collections: list[str], filter_setting: dict | None = None, generated_queries=list[str]):
        class Hit:
            def __init__(self, content):
                self.payload = {"content": content}

        return [Hit("证据段落A"), Hit("证据段落B"), Hit("证据段落C")]

    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        return [h.payload["content"] for h in hits][:keep_top_k]


@pytest.mark.asyncio
async def test_rag_multi_turn_pipeline(monkeypatch):
    llm = DummyLLM()

    intent_g = build_intent_graph(llm)
    entity_g = build_entity_graph(llm)
    context_g = build_context_graph()
    rewrite_g = build_rewrite_graph(llm)
    monkeypatch.setattr(rag_module, "VectorRetriever", DummyRetriever)
    rag_g = rag_module.build_rag_graph()

    # Turn 1
    state = {"messages": [{"role": "user", "content": "上周收到的账单"}], "session_id": "s1"}
    state |= await intent_g.ainvoke(state)
    assert state.get("intent") in ("qa", "write", "search") or state.get("intent") == "qa"

    state |= await entity_g.ainvoke(state)
    assert state.get("entities", {}).get("subject") == "账单"
    assert state.get("time_text") == "上周"

    state |= await context_g.ainvoke(state)
    assert state.get("context_frame", {}).get("subject") == "账单"
    assert state.get("time_range") and state["time_range"].get("granularity") == "week"

    state |= await rewrite_g.ainvoke(state)
    assert state.get("rewritten_query")

    state |= await rag_g.ainvoke(state)
    assert state.get("context_docs") and len(state["context_docs"]) > 0
    assert any(m.get("role") == "system" for m in state.get("messages", []))

    # Turn 2 (coreference to last month)
    state["messages"].append({"role": "user", "content": "上个月的呢？"})

    state |= await entity_g.ainvoke(state)
    state |= await context_g.ainvoke(state)
    assert state.get("context_frame", {}).get("subject") == "账单"
    assert state.get("time_range") and state["time_range"].get("granularity") == "month"

    state |= await rewrite_g.ainvoke(state)
    state |= await rag_g.ainvoke(state)
    assert state.get("context_docs") and len(state["context_docs"]) > 0




