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
            # Handle both dict format and LangChain message objects
            if isinstance(m, dict):
                if m.get("role") == "user":
                    return m.get("content", "")
            else:
                # Handle LangChain message objects (HumanMessage, AIMessage, etc.)
                from langchain_core.messages import HumanMessage
                if isinstance(m, HumanMessage):
                    return getattr(m, "content", "")
                # Also check for role attribute
                role = getattr(m, "role", None)
                if role == "user":
                    return getattr(m, "content", "")
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

        latest_user = _get_latest_user_message(state.messages)
        if not latest_user:
            return {"intent": detected_intent, "intent_confidence": confidence}

        system = """# ROLE
你是一个顶级的用户意图分析师（Intent Classifier）。你的职责是精确分析用户的真实意图，并将其分类到最合适的类别中。你需要进行深度理解和批判性思考，确保分类的准确性和一致性。

# CONTEXT
## 意图类别定义 (Intent Categories):
- **qa**: 问答类 - 用户寻求信息、解释、答案或知识
  * 特征：疑问词（什么、如何、为什么）、询问语气、求知欲望
  * 示例：什么是RAG？、如何实现AI？、为什么会这样？

- **write**: 写作类 - 用户要求创作、编写、总结或生成内容
  * 特征：创作动词（写、编写、总结、生成）、内容产出需求
  * 示例：写一份报告、总结这篇文章、生成代码

- **search**: 搜索类 - 用户要求主动搜索、查找、检索信息
  * 特征：搜索动词（搜索、查找、检索）、明确的搜索意图
  * 示例：搜索最新论文、查找相关资料、检索数据

- **exec**: 执行类 - 用户要求执行具体任务、操作或命令
  * 特征：执行动词（执行、运行、操作）、具体任务指令
  * 示例：执行备份、运行脚本、操作数据库

- **smalltalk**: 闲聊类 - 日常对话、问候、情感交流
  * 特征：社交性语言、情感表达、非任务导向
  * 示例：你好、今天天气不错、谢谢你

- **other**: 其他类 - 不属于以上任何明确类别的内容
  * 特征：模糊意图、复合需求、无法明确分类

## 分析约束 (Constraints):
- 必须严格按照下面的"分析流程"进行思考
- 输出格式必须是严格的JSON格式
- 置信度必须基于客观分析，范围0.1-1.0

# EXECUTION FLOW
请严格遵循以下流程：

**1. [理解]**
深度理解用户消息的核心意图：
- 关键词分析：识别动词、疑问词、指示词
- 语境分析：理解完整语义和隐含意图
- 目标识别：用户真正想要达成什么

**2. [分类]**
基于理解结果进行精确分类：
- 主要意图：最符合哪个类别的特征
- 置信度评估：基于特征匹配度和语义清晰度
- 边界情况：如果模糊，选择最可能的类别

**3. [输出]**
严格按照JSON格式输出结果。"""
        user = f"用户消息: {latest_user}"

        try:
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])

            # Parse model output
            import json
            content = getattr(resp, "content", "") or ""
            logger.info("llm_raw_response", content=content, user_message=latest_user)

            try:
                data = json.loads(content) if isinstance(content, str) else {}
                detected_intent = str(data.get("intent", detected_intent))
                confidence = float(data.get("confidence", confidence))
                logger.info("json_parse_success", data=data)
            except Exception as e:
                logger.warning("json_parse_failed", content=content, error=str(e))
                lc = content.lower() if isinstance(content, str) else ""
                for k in ["qa", "write", "search", "exec", "smalltalk"]:
                    if k in lc:
                        detected_intent = k
                        break
                logger.info("fallback_parse_result", detected_intent=detected_intent)

            logger.info("intent_detected", intent=detected_intent, confidence=confidence)
        except Exception as e:
            logger.error("llm_call_failed", error=str(e), user_message=latest_user)

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
            "请基于意图与上下文，将用户问题改写为用于检索的精确查询。\n"
            f"主题: {subject}{time_part}{filter_part}\n"
            f"用户原话: {user_latest}\n"
            "只返回改写后的查询。"
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