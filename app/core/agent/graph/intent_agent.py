# -*- coding: utf-8 -*-
# @Time   : 2025/8/17 10:12
# @Author : Galleons
# @File   : intent_agent.py

"""
Intent detection sub-graph for LangGraph.

One graph per file: this module defines a minimal graph that detects the
user intent from the latest user message and writes `intent` and
`intent_confidence` into the `GraphState`.
"""

from typing import Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.models import GraphState


def build_intent_graph(llm) -> CompiledStateGraph:
    """Build a compiled sub-graph that performs intent detection.

    Args:
        llm: A chat model instance exposing `ainvoke` compatible with LangChain's ChatOpenAI.

    Returns:
        CompiledStateGraph: A compiled LangGraph sub-graph.
    """

    async def _detect_intent(state: GraphState) -> dict:
        """Detect user intent from the latest user message.

        Produces keys: `intent` (str) and `intent_confidence` (float [0,1]).
        """
        detected_intent: str = "other"
        confidence: float = 0.5

        try:
            # Find latest user message
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
        except Exception as e:
            logger.warning("intent_detection_failed", error=str(e))

        return {"intent": detected_intent, "intent_confidence": confidence}

    g = StateGraph(GraphState)
    g.add_node("detect_intent", _detect_intent)
    g.set_entry_point("detect_intent")
    g.set_finish_point("detect_intent")
    return g.compile(name="IntentDetectionGraph")


