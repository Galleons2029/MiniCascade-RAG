# -*- coding: utf-8 -*-
# @Time   : 2025/8/17 10:12
# @Author : Galleons
# @File   : intent_agent.py

"""
Unified Agent Graph for MiniCascade-RAG.

This module combines all sub-agents (intent detection, entity extraction, 
context resolution, query rewrite, and RAG retrieval) into a single,
comprehensive agent graph.
"""

from datetime import datetime, timedelta
from typing import Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.core.rag.retriever import VectorRetriever
from app.pipeline.inference_pipeline.config import settings as rag_settings
from app.models import GraphState


def _normalize_time(time_text: Optional[str]) -> dict | None:
    """Normalize relative time expressions to absolute time ranges."""
    if not time_text:
        return None
    now = datetime.utcnow()
    tt = time_text.strip().lower()
    try:
        if "上周" in tt or "last week" in tt:
            # naive week: last 7 days window
            end = now - timedelta(days=now.weekday() + 1)
            start = end - timedelta(days=6)
            return {"start": start.isoformat(), "end": end.isoformat(), "granularity": "week"}
        if "上个月" in tt or "last month" in tt:
            # naive last 30 days
            end = now.replace(day=1) - timedelta(days=1)
            start = end.replace(day=1)
            return {"start": start.isoformat(), "end": end.isoformat(), "granularity": "month"}
    except Exception:
        pass
    return None


def _get_latest_user_message(messages) -> Optional[str]:
    """Extract the latest user message from the conversation."""
    for m in reversed(messages):
        try:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "user":
                return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        except Exception:
            continue
    return None


def build_unified_agent_graph(llm) -> CompiledStateGraph:
    """Build a unified agent graph that combines all RAG processing steps.

    Args:
        llm: A chat model instance exposing `ainvoke` compatible with LangChain's ChatOpenAI.

    Returns:
        CompiledStateGraph: A compiled LangGraph with all agent functionalities.
    """

    async def _detect_intent(state: GraphState) -> dict:
        """Step 1: Detect user intent from the latest user message."""
        detected_intent: str = "other"
        confidence: float = 0.5

        try:
            latest_user = _get_latest_user_message(state.messages)
            if not latest_user:
                return {"intent": detected_intent, "intent_confidence": confidence}

            system = (
                "You are an intent classifier. Classify the user's latest message into one of: "
                "qa, write, search, exec, smalltalk, other. "
                "Return a JSON object with keys: intent, confidence (0-1)."
            )
            user = f"Message: {latest_user}\nRespond with JSON only."
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])

            # Parse model output
            import json
            content = getattr(resp, "content", "") or ""
            try:
                data = json.loads(content) if isinstance(content, str) else {}
                detected_intent = str(data.get("intent", detected_intent))
                confidence = float(data.get("confidence", confidence))
            except Exception:
                lc = content.lower() if isinstance(content, str) else ""
                for k in ["qa", "write", "search", "exec", "smalltalk"]:
                    if k in lc:
                        detected_intent = k
                        break

            logger.info("intent_detected", intent=detected_intent, confidence=confidence)
        except Exception as e:
            logger.warning("intent_detection_failed", error=str(e))

        return {"intent": detected_intent, "intent_confidence": confidence}

    async def _extract_entities(state: GraphState) -> dict:
        """Step 2: Extract entities and time expressions from user message."""
        latest_user = _get_latest_user_message(state.messages)
        if not latest_user:
            return {}

        system = (
            "You are an information extraction assistant. Extract entities and slots from the user's message. "
            "Return JSON with keys: entities (object), time_text (string)."
        )
        user = f"Message: {latest_user}\nRespond with JSON only."
        
        try:
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            import json
            content = getattr(resp, "content", "") or ""
            data = json.loads(content) if isinstance(content, str) else {}
            entities = data.get("entities") or {}
            time_text = data.get("time_text")
            
            logger.info("entities_extracted", entities=entities, time_text=time_text)
            return {"entities": entities, "time_text": time_text}
        except Exception as e:
            logger.warning("entity_extraction_failed", error=str(e))
            return {}

    async def _resolve_context(state: GraphState) -> dict:
        """Step 3: Resolve context and coreferences using previous context frame."""
        frame = state.context_frame or {}
        entities = state.entities or {}

        # inherit subject/filters if omitted this turn
        subject = entities.get("subject") or frame.get("subject")
        filters = entities.get("filters") or frame.get("filters")

        # normalize time
        time_text = state.time_text
        time_range = _normalize_time(time_text) or frame.get("time_range")

        new_frame = {
            "subject": subject,
            "filters": filters,
            "time_range": time_range,
        }
        
        logger.info("context_resolved", context_frame=new_frame)
        return {"context_frame": new_frame, "time_range": time_range}

    async def _rewrite_query(state: GraphState) -> dict:
        """Step 4: Rewrite user intent + entities + context into a precise query."""
        entities = state.entities or {}
        frame = state.context_frame or {}
        
        # Simple template; can be made domain-aware
        subject = entities.get("subject") or frame.get("subject") or "信息"
        time_part = ""
        if frame.get("time_range"):
            tr = frame["time_range"]
            time_part = f" 时间范围: {tr.get('start')} 到 {tr.get('end')}"
        filters = entities.get("filters") or frame.get("filters")
        filter_part = f" 过滤: {filters}" if filters else ""

        user_latest = _get_latest_user_message(state.messages)
        prompt = (
            f"请基于意图与上下文，将用户问题改写为用于检索的精确查询。\n"
            f"主题: {subject}{time_part}{filter_part}\n"
            f"用户原话: {user_latest}\n"
            f"只返回改写后的查询。"
        )

        try:
            resp = await llm.ainvoke([
                {"role": "system", "content": "你是查询改写助手。"},
                {"role": "user", "content": prompt},
            ])
            rewritten = getattr(resp, "content", "").strip()
            
            logger.info("query_rewritten", original=user_latest, rewritten=rewritten)
            return {"rewritten_query": rewritten}
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e))
            return {}

    async def _retrieve_documents(state: GraphState) -> dict:
        """Step 5: Perform RAG retrieval using the rewritten query."""
        # Choose query: prefer rewritten_query, fallback to latest user
        query = state.rewritten_query
        if not query:
            query = _get_latest_user_message(state.messages)

        if not query:
            return {}

        collections = state.doc_names or ["dir_default"]
        try:
            retriever = VectorRetriever(query=query)
            generated = retriever.multi_query(to_expand_to_n_queries=3)
            hits = retriever.retrieve_top_k(
                k=rag_settings.TOP_K,
                collections=collections,
                generated_queries=generated,
            )
            passages = retriever.rerank(hits=hits, keep_top_k=rag_settings.KEEP_TOP_K)
            
            # Optionally inject as system message
            system_msg = {
                "role": "system",
                "content": "\n\n".join(passages) if passages else "",
            }
            
            logger.info("documents_retrieved", 
                       query=query, 
                       num_passages=len(passages), 
                       collections=collections)
            return {"context_docs": passages, "messages": [system_msg] if passages else []}
        except Exception as e:
            logger.warning("rag_retrieval_failed", error=str(e))
            return {}

    async def _route_by_intent(state: GraphState) -> Literal["rag_pipeline", "direct_response"]:
        """Route based on detected intent."""
        intent = (state.intent or "other").lower()
        if intent in ("qa", "write"):
            return "rag_pipeline"
        else:
            # For search, exec, smalltalk, other - skip RAG pipeline
            return "direct_response"

    # Build the graph
    graph = StateGraph(GraphState)
    
    # Add all nodes
    graph.add_node("detect_intent", _detect_intent)
    graph.add_node("extract_entities", _extract_entities)
    graph.add_node("resolve_context", _resolve_context)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("retrieve_documents", _retrieve_documents)
    
    # Set entry point
    graph.set_entry_point("detect_intent")
    
    # Add conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        _route_by_intent,
        {
            "rag_pipeline": "extract_entities",
            "direct_response": END,
        }
    )
    
    # For RAG pipeline: intent -> entities -> context -> rewrite -> retrieve -> END
    graph.add_edge("extract_entities", "resolve_context")
    graph.add_edge("resolve_context", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", END)
    
    return graph.compile(name="UnifiedAgentGraph")


# Backward compatibility - keep the original function name
def build_intent_graph(llm) -> CompiledStateGraph:
    """Legacy function name for backward compatibility."""
    return build_unified_agent_graph(llm)