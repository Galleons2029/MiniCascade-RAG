# -*- coding: utf-8 -*-
# @Time   : 2025/8/31 22:15
# @Author : ggbond
# @File   : supervisor_agent.py

"""
Lightweight Supervisor Agent for MiniCascade-RAG.

This module implements a simplified supervisor for task routing and coordination
between different agents. It provides core task classification and routing logic
without complex multi-agent orchestration.
"""

from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.logger_utils import logger
from app.models import GraphState
from app.core.agent.graph.intent_agent import build_unified_agent_graph


# Simple task classification keywords
TASK_KEYWORDS = {
    "simple_qa": ["什么是", "谁是", "何时", "哪里", "多少", "定义", "解释"],
    "complex_research": ["分析", "比较", "研究", "调查", "综合", "详细分析"],
    "multi_step": ["步骤", "工作流", "流程", "如何", "首先", "然后"],
    "tool_heavy": ["计算", "处理数据", "分析文件", "生成报告"]
}

def classify_task_simple(user_input: str) -> tuple[str, float]:
    """Simple keyword-based task classification."""
    user_input_lower = user_input.lower()
    scores = {}

    for task_type, keywords in TASK_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword.lower() in user_input_lower:
                score += 1.0
        scores[task_type] = score / len(keywords) if keywords else 0.0

    if not scores or max(scores.values()) == 0:
        return "simple_qa", 0.5  # Default fallback

    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]


class SupervisorState(GraphState):
    """Extended state for supervisor operations."""
    # Inherit all fields from GraphState
    task_type: Optional[str] = None
    task_confidence: Optional[float] = None
    assigned_agent: Optional[str] = None
    supervisor_reasoning: Optional[str] = None


def build_supervisor_graph(llm) -> CompiledStateGraph:
    """Build a lightweight supervisor graph for task routing.

    Args:
        llm: A chat model instance for the supervisor

    Returns:
        CompiledStateGraph: A compiled supervisor graph
    """

    async def _classify_task(state: SupervisorState) -> dict:
        """Step 1: Classify the task type using simple keyword matching."""
        latest_user = _get_latest_user_message(state.messages)
        if not latest_user:
            return {"task_type": "simple_qa", "task_confidence": 0.5}

        # Use simple keyword-based classification
        task_type, confidence = classify_task_simple(latest_user)

        logger.info("task_classified",
                   task_type=task_type,
                   confidence=confidence,
                   user_input=latest_user[:50] + "..." if len(latest_user) > 50 else latest_user)

        return {
            "task_type": task_type,
            "task_confidence": confidence,
            "supervisor_reasoning": f"Classified as {task_type} with confidence {confidence:.2f}"
        }
    
    async def _route_task(state: SupervisorState) -> dict:
        """Step 2: Simple task routing - currently only routes to unified_agent."""
        task_type = state.task_type or "simple_qa"
        task_confidence = state.task_confidence or 0.5

        # For now, always route to unified_agent
        # This can be extended later to route to other agents
        assigned_agent = "unified_agent"
        reasoning = f"Routing {task_type} task to {assigned_agent} (confidence: {task_confidence:.2f})"

        logger.info("task_routed",
                   task_type=task_type,
                   assigned_agent=assigned_agent,
                   confidence=task_confidence,
                   reasoning=reasoning)

        return {
            "assigned_agent": assigned_agent,
            "supervisor_reasoning": reasoning
        }
    
    async def _execute_agent(state: SupervisorState) -> Command:
        """Step 3: Execute the assigned agent."""
        assigned_agent = state.assigned_agent or "unified_agent"

        # Route to the appropriate worker node
        if assigned_agent == "unified_agent":
            return Command(goto="unified_agent_worker")
        else:
            # For future agents, you can add more routing logic here
            logger.warning("unknown_agent", agent=assigned_agent)
            return Command(goto="unified_agent_worker")  # Fallback
    
    async def _unified_agent_worker(state: SupervisorState) -> dict:
        """Worker node for unified agent execution."""
        # Build the unified agent graph
        unified_graph = build_unified_agent_graph(llm)

        # Execute the unified agent
        try:
            result = await unified_graph.ainvoke({
                "messages": state.messages,
                "session_id": state.session_id
            })

            logger.info("agent_completed",
                       agent_name="unified_agent",
                       task_type=state.task_type,
                       confidence=state.task_confidence)

            # Return the result with supervisor metadata
            return {
                "messages": result.get("messages", []),
                "task_type": state.task_type,
                "task_confidence": state.task_confidence,
                "assigned_agent": state.assigned_agent,
                "supervisor_reasoning": state.supervisor_reasoning
            }
        except Exception as e:
            logger.error("agent_execution_failed", agent_name="unified_agent", error=str(e))
            return {
                "messages": [],
                "error": str(e),
                "task_type": state.task_type,
                "assigned_agent": state.assigned_agent
            }
    
    def _get_latest_user_message(messages) -> Optional[str]:
        """Extract the latest user message from the conversation."""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
            elif isinstance(m, HumanMessage):
                return m.content
        return None
    
    # Build the supervisor graph
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("classify_task", _classify_task)
    graph.add_node("route_task", _route_task)
    graph.add_node("execute_agent", _execute_agent)
    graph.add_node("unified_agent_worker", _unified_agent_worker)

    # Add edges
    graph.add_edge(START, "classify_task")
    graph.add_edge("classify_task", "route_task")
    graph.add_edge("route_task", "execute_agent")
    graph.add_edge("unified_agent_worker", END)

    return graph.compile(name="LightweightSupervisorAgent")
