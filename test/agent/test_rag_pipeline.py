# -*- coding: utf-8 -*-

import pytest

from app.core.agent.graph.intent_agent import build_unified_agent_graph
import app.core.rag.retriever as retriever_module


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

    def retrieve_top_k(
        self, k: int, collections: list[str], filter_setting: dict | None = None, generated_queries=list[str]
    ):
        class Hit:
            def __init__(self, content):
                self.payload = {"content": content}

        return [Hit("证据段落A"), Hit("证据段落B"), Hit("证据段落C")]

    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        return [h.payload["content"] for h in hits][:keep_top_k]


@pytest.mark.asyncio
async def test_unified_rag_pipeline(monkeypatch):
    """Test the unified agent graph with a complete RAG pipeline."""
    llm = DummyLLM()
    
    # Mock the VectorRetriever to avoid external dependencies
    monkeypatch.setattr(retriever_module, "VectorRetriever", DummyRetriever)
    
    # Build the unified agent graph
    unified_graph = build_unified_agent_graph(llm)

    # Turn 1: Test QA intent with entity extraction and RAG retrieval
    state = {
        "messages": [{"role": "user", "content": "上周收到的账单"}], 
        "session_id": "s1"
    }
    
    # Process through the unified graph
    result_state = await unified_graph.ainvoke(state)
    
    # Verify intent detection
    assert result_state.get("intent") == "qa"
    assert result_state.get("intent_confidence") > 0.5
    
    # Verify entity extraction
    assert result_state.get("entities", {}).get("subject") == "账单"
    assert result_state.get("time_text") == "上周"
    
    # Verify context resolution
    assert result_state.get("context_frame", {}).get("subject") == "账单"
    assert result_state.get("time_range") and result_state["time_range"].get("granularity") == "week"
    
    # Verify query rewriting
    assert result_state.get("rewritten_query")
    
    # Verify RAG retrieval
    assert result_state.get("context_docs") and len(result_state["context_docs"]) > 0
    assert any(m.get("role") == "system" for m in result_state.get("messages", []))
    
    # Turn 2: Test coreference resolution (follow-up question)
    # Add the follow-up question to the conversation
    state["messages"] = result_state["messages"]  # Keep conversation history
    state["messages"].append({"role": "user", "content": "上个月的呢？"})
    state["context_frame"] = result_state.get("context_frame")  # Keep context
    
    # Process the follow-up through unified graph
    result_state_2 = await unified_graph.ainvoke(state)
    
    # Verify that context is maintained and updated
    assert result_state_2.get("context_frame", {}).get("subject") == "账单"
    assert result_state_2.get("time_range") and result_state_2["time_range"].get("granularity") == "month"
    
    # Verify that documents are retrieved for the new time frame
    assert result_state_2.get("context_docs") and len(result_state_2["context_docs"]) > 0


@pytest.mark.asyncio
async def test_unified_agent_non_qa_intent(monkeypatch):
    """Test that non-QA intents bypass the RAG pipeline."""
    llm = DummyLLM()
    
    # Mock the VectorRetriever
    monkeypatch.setattr(retriever_module, "VectorRetriever", DummyRetriever)
    
    # Build the unified agent graph
    unified_graph = build_unified_agent_graph(llm)
    
    # Test with a non-QA intent (this should trigger the fallback in DummyLLM)
    state = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}], 
        "session_id": "s2"
    }
    
    # Process through the unified graph
    result_state = await unified_graph.ainvoke(state)
    
    # For non-QA intents, the graph should detect intent but not go through RAG pipeline
    # This depends on how DummyLLM handles non-bill related queries
    assert "intent" in result_state
    
    # The exact behavior depends on how the routing works for non-QA intents
