# -*- coding: utf-8 -*-
# @Time   : 2025/8/17 10:12
# @Author : ggbond
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


def _evaluate_task_complexity(user_input: str, intent: str, intent_confidence: float) -> float:
    """Evaluate task complexity based on multiple factors.

    Args:
        user_input: User's input text
        intent: Detected intent
        intent_confidence: Confidence of intent detection

    Returns:
        float: Complexity score (0.0 - 1.0)
    """
    complexity_score = 0.0

    # Factor 1: Text length and structure (0.0 - 0.3)
    text_length = len(user_input)
    if text_length > 100:
        complexity_score += 0.2
    elif text_length > 50:
        complexity_score += 0.1

    # Count sentences
    sentence_count = (user_input.count('。') + user_input.count('？') + user_input.count('！') +
                     user_input.count('.') + user_input.count('?') + user_input.count('!'))
    if sentence_count > 2:
        complexity_score += 0.1

    # Factor 2: Complexity keywords (0.0 - 0.4)
    high_complexity_keywords = [
        # 中文复杂关键词
        "分析", "比较", "评估", "研究", "调查", "综合", "详细", "深入", "全面", "系统",
        "优缺点", "利弊", "对比", "差异", "相似", "关联", "影响", "原因", "结果", "趋势",
        "策略", "方案", "建议", "解决方案", "实施", "步骤", "流程", "工作流",
        # 英文复杂关键词
        "analyze", "compare", "evaluate", "research", "investigate", "comprehensive", "detailed",
        "in-depth", "systematic", "pros and cons", "advantages", "disadvantages", "differences",
        "similarities", "correlations", "impact", "causes", "effects", "trends", "strategy",
        "solution", "implementation", "workflow", "process"
    ]

    multi_step_keywords = [
        "步骤", "流程", "工作流", "如何", "怎么", "教程", "指南", "方法", "过程",
        "首先", "然后", "接下来", "最后", "第一", "第二", "第三",
        "step", "process", "workflow", "how to", "tutorial", "guide", "method",
        "first", "then", "next", "finally", "step by step"
    ]

    user_lower = user_input.lower()

    # Check for high complexity keywords
    high_complexity_count = sum(1 for keyword in high_complexity_keywords if keyword in user_lower)
    complexity_score += min(high_complexity_count * 0.1, 0.3)

    # Check for multi-step keywords
    multi_step_count = sum(1 for keyword in multi_step_keywords if keyword in user_lower)
    complexity_score += min(multi_step_count * 0.1, 0.2)

    # Factor 3: Intent-based complexity (0.0 - 0.2)
    if intent == "write":
        complexity_score += 0.1  # Writing tasks are generally more complex
    elif intent == "search":
        complexity_score += 0.05  # Search tasks can be complex

    # Factor 4: Intent confidence (0.0 - 0.1)
    # Lower confidence might indicate more complex/ambiguous requests
    if intent_confidence < 0.7:
        complexity_score += 0.1
    elif intent_confidence < 0.8:
        complexity_score += 0.05

    # Factor 5: Question complexity patterns (0.0 - 0.2)
    complex_patterns = [
        "为什么", "怎么样", "如何实现", "什么原因", "有什么区别", "哪个更好",
        "why", "how does", "what causes", "what's the difference", "which is better"
    ]

    pattern_count = sum(1 for pattern in complex_patterns if pattern in user_lower)
    complexity_score += min(pattern_count * 0.1, 0.2)

    # Normalize to 0.0 - 1.0 range
    complexity_score = min(complexity_score, 1.0)

    return complexity_score


async def _evaluate_complexity_and_route(
    state: GraphState
) -> Literal["direct_response", "rag_pipeline", "supervisor_pipeline"]:
    """Evaluate task complexity and route to appropriate pipeline."""
    intent = (state.intent or "other").lower()
    intent_confidence = state.intent_confidence or 0.5
    latest_user = _get_latest_user_message(state.messages)

    if not latest_user:
        return "direct_response"

    # Step 1: Check if it's a simple intent that doesn't need RAG
    if intent in ("smalltalk", "exec"):
        logger.info("routing_decision",
                   route="direct_response",
                   reason="Simple intent, no RAG needed",
                   intent=intent)
        return "direct_response"

    # Step 2: Evaluate complexity for qa/write/search intents
    complexity_score = _evaluate_task_complexity(latest_user, intent, intent_confidence)

    # Step 3: Route based on complexity
    if complexity_score < 0.3:
        # Simple questions - direct response without RAG
        logger.info("routing_decision",
                   route="direct_response",
                   reason="Low complexity, direct response",
                   complexity_score=complexity_score,
                   intent=intent)
        return "direct_response"
    elif complexity_score < 0.7:
        # Medium complexity - use RAG
        logger.info("routing_decision",
                   route="rag_pipeline",
                   reason="Medium complexity, use RAG",
                   complexity_score=complexity_score,
                   intent=intent)
        return "rag_pipeline"
    else:
        # High complexity - use supervisor for advanced processing
        logger.info("routing_decision",
                   route="supervisor_pipeline",
                   reason="High complexity, use supervisor",
                   complexity_score=complexity_score,
                   intent=intent)
        return "supervisor_pipeline"


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
你是一个顶级的用户意图分析师（Intent Classifier）。你的职责是精确分析用户的真实意图，
并将其分类到最合适的类别中。你需要进行深度理解和批判性思考，确保分类的准确性和一致性。

# CONTEXT
## 意图类别定义 (Intent Categories):
- **qa**: 问答类 - 用户寻求信息、解释、答案或知识
  * 特征：疑问词（什么、如何、为什么、怎么样、显示了什么）、询问语气、求知欲望
  * 关键区别：用户期望得到**直接回答**，而不是搜索过程
  * 示例：什么是RAG？、如何实现AI？、销售数据显示了什么趋势？、
    上个月的情况如何？

- **write**: 写作类 - 用户要求创作、编写、总结或生成内容
  * 特征：创作动词（写、编写、总结、生成）、内容产出需求
  * 示例：写一份报告、总结这篇文章、生成代码

- **search**: 搜索类 - 用户要求主动搜索、查找、检索信息
  * 特征：搜索动词（搜索、查找、检索、找一下）、明确的搜索行为指令
  * 关键区别：用户期望执行**搜索动作**，而不是直接获得答案
  * 示例：搜索最新论文、查找相关资料、检索数据、
    帮我找一下相关信息

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
- **qa vs search 区分**：
  * 如果用户想要**直接答案**（如"什么是..."、"如何..."、"显示了什么"） → qa
  * 如果用户想要**执行搜索**（如"搜索..."、"查找..."、"检索..."） → search
- 置信度评估：基于特征匹配度和语义清晰度
- 边界情况：如果模糊，选择最可能的类别

**3. [输出]**
严格按照JSON格式输出结果。"""
        # 构建用户提示词，包含对话历史
        context_info = ""
        if len(state.messages) > 1:
            # 有对话历史，构建上下文
            context_lines = []
            for msg in state.messages[:-1]:  # 排除最后一条消息
                # 处理 LangChain 消息对象
                if hasattr(msg, "type"):
                    msg_type = msg.type
                    content = getattr(msg, "content", "")
                else:
                    # 处理字典格式的消息
                    msg_type = msg.get("role", "unknown")
                    content = msg.get("content", "")

                if msg_type in ["human", "user"]:
                    context_lines.append(f"用户: {content}")
                elif msg_type in ["ai", "assistant"]:
                    context_lines.append(f"助手: {content}")

            if context_lines:
                context_info = f"""
**对话历史**:
{chr(10).join(context_lines)}

"""

        user = f"""# GOAL
分析以下用户消息的真实意图：
{context_info}**当前用户消息**: "{latest_user}"

请按照执行流程进行分析，最终输出JSON格式：
{{"intent": "分类结果", "confidence": 置信度}}"""

        try:
            resp = await llm.ainvoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )

            # Parse model output
            import json

            content = getattr(resp, "content", "") or ""
            logger.info("llm_raw_response", content=content, user_message=latest_user)

            try:
                # 处理可能包含 markdown 代码块和额外内容的 JSON
                json_content = content
                if isinstance(content, str):
                    # 移除 markdown 代码块标记
                    json_content = content.strip()

                    # 查找 JSON 代码块
                    if "```json" in json_content:
                        # 提取 ```json 和 ``` 之间的内容
                        start_idx = json_content.find("```json") + 7
                        end_idx = json_content.find("```", start_idx)
                        if end_idx != -1:
                            json_content = json_content[start_idx:end_idx].strip()
                    elif json_content.startswith("```"):
                        # 处理普通代码块
                        json_content = json_content[3:]
                        if json_content.endswith("```"):
                            json_content = json_content[:-3]
                        json_content = json_content.strip()

                    # 如果还有额外内容，尝试只提取 JSON 部分
                    if json_content and not json_content.startswith("{"):
                        # 查找第一个 { 和最后一个 }
                        start_brace = json_content.find("{")
                        end_brace = json_content.rfind("}")
                        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                            json_content = json_content[start_brace : end_brace + 1]

                data = json.loads(json_content) if json_content else {}
                detected_intent = str(data.get("intent", detected_intent))
                confidence = float(data.get("confidence", confidence))
                logger.info("json_parse_success", data=data, original_content=content)
            except Exception as e:
                logger.warning("json_parse_failed", content=content, error=str(e))
                # 备用解析：从文本中提取意图
                lc = content.lower() if isinstance(content, str) else ""
                for k in ["qa", "write", "search", "exec", "smalltalk"]:
                    if k in lc:
                        detected_intent = k
                        break
                # 尝试提取置信度
                import re

                confidence_match = re.search(r'"confidence":\s*([0-9.]+)', content)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                    except ValueError:
                        pass
                logger.info("fallback_parse_result", detected_intent=detected_intent, confidence=confidence)

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
            resp = await llm.ainvoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
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
            resp = await llm.ainvoke(
                [
                    {"role": "system", "content": "你是查询改写助手。"},
                    {"role": "user", "content": prompt},
                ]
            )
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
            # Ensure generated is a list (in case it's a generator)
            if hasattr(generated, '__iter__') and not isinstance(generated, (str, bytes)):
                generated = list(generated)

            hits = retriever.retrieve_top_k(
                k=rag_settings.TOP_K,
                collections=collections,
                generated_queries=generated,
            )
            passages = retriever.rerank(hits=hits, keep_top_k=rag_settings.KEEP_TOP_K)
            # Ensure passages is a list (in case it's a generator)
            if hasattr(passages, '__iter__') and not isinstance(passages, (str, bytes)):
                passages = list(passages)

            # Optionally inject as system message
            system_msg = {
                "role": "system",
                "content": "\n\n".join(passages) if passages else "",
            }

            logger.info("documents_retrieved", query=query, num_passages=len(passages), collections=collections)
            return {"context_docs": passages, "messages": [system_msg] if passages else []}
        except Exception as e:
            logger.warning("rag_retrieval_failed", error=str(e))
            return {}

    async def _handle_supervisor_pipeline(state: GraphState) -> dict:
        """Step 6: Handle complex tasks using supervisor agent."""
        try:
            # Import supervisor graph builder
            from app.core.agent.graph.supervisor_agent import build_supervisor_graph
            
            # Build and execute supervisor graph
            supervisor_graph = build_supervisor_graph(llm)
            
            result = await supervisor_graph.ainvoke({
                "messages": state.messages,
                "session_id": state.session_id,
                "intent": state.intent,
                "intent_confidence": state.intent_confidence
            })
            
            logger.info("supervisor_pipeline_completed", 
                       intent=state.intent,
                       session_id=state.session_id)
            
            return {
                "messages": result.get("messages", []),
                "task_type": result.get("task_type"),
                "assigned_agent": result.get("assigned_agent"),
                "supervisor_reasoning": result.get("supervisor_reasoning")
            }
            
        except Exception as e:
            logger.error("supervisor_pipeline_failed", error=str(e))
            # Fallback to direct response
            return {"messages": []}

    # Build the graph
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("detect_intent", _detect_intent)
    graph.add_node("extract_entities", _extract_entities)
    graph.add_node("resolve_context", _resolve_context)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("retrieve_documents", _retrieve_documents)
    graph.add_node("handle_supervisor_pipeline", _handle_supervisor_pipeline)

    # Set entry point
    graph.set_entry_point("detect_intent")

    # Add conditional routing after intent detection with complexity evaluation
    graph.add_conditional_edges(
        "detect_intent",
        _evaluate_complexity_and_route,
        {
            "rag_pipeline": "extract_entities",
            "direct_response": END,
            "supervisor_pipeline": "handle_supervisor_pipeline",
        },
    )

    # For RAG pipeline: intent -> entities -> context -> rewrite -> retrieve -> END
    graph.add_edge("extract_entities", "resolve_context")
    graph.add_edge("resolve_context", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", END)
    
    # For supervisor pipeline: intent -> supervisor -> END
    graph.add_edge("handle_supervisor_pipeline", END)

    return graph.compile(name="UnifiedAgentGraph")


# Backward compatibility - keep the original function name
def build_intent_graph(llm) -> CompiledStateGraph:
    """Legacy function name for backward compatibility."""
    return build_unified_agent_graph(llm)