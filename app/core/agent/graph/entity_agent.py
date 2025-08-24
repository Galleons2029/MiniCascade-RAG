# -*- coding: utf-8 -*-

"""
Entity extraction sub-graph.

Extract key slots/entities (subject, action, customer, amount, etc.) and
time expressions from the latest user message into GraphState.
"""

from typing import Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.models import GraphState


def build_entity_graph(llm) -> CompiledStateGraph:
    async def _extract(state: GraphState) -> dict:
        latest_user: Optional[str] = None
        for m in reversed(state.messages):
            try:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "user":
                    latest_user = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    break
            except Exception:
                continue

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
            return {"entities": entities, "time_text": time_text}
        except Exception as e:
            logger.warning("entity_extraction_failed", error=str(e))
            return {}

    g = StateGraph(GraphState)
    g.add_node("extract", _extract)
    g.set_entry_point("extract")
    g.set_finish_point("extract")
    return g.compile(name="EntityExtractionGraph")


