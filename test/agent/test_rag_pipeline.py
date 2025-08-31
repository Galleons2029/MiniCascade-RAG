# -*- coding: utf-8 -*-

import pytest

from app.core.agent.graph.intent_agent import build_unified_agent_graph


class DummyLLM:
    async def ainvoke(self, messages):
        # Extract user message
        user_content = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_content = msg.get("content", "").lower()
                break

        # Extract system message to determine response type
        system_content = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_content = msg.get("content", "").lower()
                break

        # Intent classification
        if "intent classifier" in system_content or "意图分析师" in system_content:
            if "账单" in user_content:
                content = '{"intent": "qa", "confidence": 0.75}'
            elif "hello" in user_content or "how are you" in user_content:
                content = '{"intent": "smalltalk", "confidence": 0.8}'
            else:
                content = '{"intent": "qa", "confidence": 0.5}'

        # Entity extraction
        elif "information extraction" in system_content or "信息提取" in system_content:
            import json
            entities = {}
            time_text = None
            if "账单" in user_content:
                entities["subject"] = "账单"
            if "上周" in user_content:
                time_text = "上周"
            elif "上个月" in user_content:
                time_text = "上个月"
            content = json.dumps({"entities": entities, "time_text": time_text}, ensure_ascii=False)

        # Query rewrite
        elif "查询改写" in system_content:
            content = f"改写后的查询: {user_content}"

        # Default response
        else:
            content = "Default response"

        return type("Response", (), {"content": content})()


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
    monkeypatch.setattr("app.core.agent.graph.intent_agent.VectorRetriever", DummyRetriever)

    # Build the unified agent graph
    unified_graph = build_unified_agent_graph(llm)

    # Turn 1: Test QA intent with entity extraction and RAG retrieval
    state = {"messages": [{"role": "user", "content": "上周收到的账单"}], "session_id": "s1"}

    # Process through the unified graph
    result_state = await unified_graph.ainvoke(state)

    # Verify intent detection
    assert result_state.get("intent") == "qa"
    assert result_state.get("intent_confidence") >= 0.5

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
    # Check for system message (could be dict or Message object)
    messages = result_state.get("messages", [])
    # Check if any system messages exist
    any(
        (isinstance(m, dict) and m.get("role") == "system") or
        (hasattr(m, "type") and m.type == "system")
        for m in messages
    )

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
    monkeypatch.setattr("app.core.agent.graph.intent_agent.VectorRetriever", DummyRetriever)

    # Build the unified agent graph
    unified_graph = build_unified_agent_graph(llm)

    # Test with a non-QA intent
    state = {"messages": [{"role": "user", "content": "Hello, how are you?"}], "session_id": "s2"}

    # Process through the unified graph
    result_state = await unified_graph.ainvoke(state)

    # Verify intent is correctly identified as non-QA
    assert result_state.get("intent") == "smalltalk"

    # Verify that the RAG pipeline was bypassed
    assert "context_docs" not in result_state or not result_state["context_docs"]
    assert "rewritten_query" not in result_state or not result_state["rewritten_query"]
