# -*- coding: utf-8 -*-

"""
Context/coreference resolution sub-graph.

Uses previous context_frame to fill omitted slots (subject, filters) and
normalize relative time expressions (e.g., 上周→绝对时间区间)。
"""

from datetime import datetime, timedelta
from typing import Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


from app.models import GraphState


def _normalize_time(time_text: Optional[str]) -> dict | None:
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


def build_context_graph() -> CompiledStateGraph:
    async def _resolve(state: GraphState) -> dict:
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
        return {"context_frame": new_frame, "time_range": time_range}

    g = StateGraph(GraphState)
    g.add_node("resolve", _resolve)
    g.set_entry_point("resolve")
    g.set_finish_point("resolve")
    return g.compile(name="ContextResolutionGraph")


