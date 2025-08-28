# -*- coding: utf-8 -*-

"""
Query rewrite sub-graph.

Rewrite user intent + entities + context_frame into a precise, retriever-friendly query.
"""



from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.models import GraphState


def build_rewrite_graph(llm) -> CompiledStateGraph:
    async def _rewrite(state: GraphState) -> dict:
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

        user_latest = None
        for m in reversed(state.messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "user":
                user_latest = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                break

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
            return {"rewritten_query": rewritten}
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e))
            return {}

    g = StateGraph(GraphState)
    g.add_node("rewrite", _rewrite)
    g.set_entry_point("rewrite")
    g.set_finish_point("rewrite")
    return g.compile(name="QueryRewriteGraph")


