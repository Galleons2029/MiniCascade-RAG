# -*- coding: utf-8 -*-

"""
RAG retrieval sub-graph.

Use VectorRetriever to expand query, retrieve from Qdrant, rerank and produce
top passages into GraphState.context_docs, and optionally inject a system
message for grounding.
"""

from typing import Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.core.rag.retriever import VectorRetriever
from app.pipeline.inference_pipeline.config import settings as rag_settings
from app.models import GraphState


def build_rag_graph() -> CompiledStateGraph:
    async def _retrieve(state: GraphState) -> dict:
        # Choose query: prefer rewritten_query, fallback to latest user
        query = state.rewritten_query
        if not query:
            for m in reversed(state.messages):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "user":
                    query = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    break

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
            return {"context_docs": passages, "messages": [system_msg] if passages else []}
        except Exception as e:
            logger.warning("rag_retrieval_failed", error=str(e))
            return {}

    g = StateGraph(GraphState)
    g.add_node("retrieve", _retrieve)
    g.set_entry_point("retrieve")
    g.set_finish_point("retrieve")
    return g.compile(name="RAGRetrievalGraph")


