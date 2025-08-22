# -*- coding: utf-8 -*-
# @Time   : 2025/8/16 23:38
# @Author : Galleons
# @File   : graph.py

"""
This file contains the graph models for the application.
"""

import re
import uuid
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    session_id: str = Field(..., description="The unique identifier for the conversation session")
    intent: str | None = Field(default=None, description="Detected user intent label")
    intent_confidence: float | None = Field(default=None, description="Confidence of detected intent [0,1]")
    # Entity & context resolution
    entities: dict | None = Field(default=None, description="Extracted entities/slots from user message")
    time_text: str | None = Field(default=None, description="Normalized time expression text (e.g., 上个月/上周)")
    time_range: dict | None = Field(default=None, description="Resolved time range, e.g., {start,end,granularity}")
    context_frame: dict | None = Field(default=None, description="Conversation frame for multi-turn coreference")
    # Query rewrite
    rewritten_query: str | None = Field(default=None, description="Rewritten query for retrieval/execution")
    # RAG context
    context_docs: list[str] | None = Field(default=None, description="Top passages for grounding")
    doc_names: list[str] | None = Field(default=None, description="Target collections/namespaces for retrieval")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that the session ID is a valid UUID or follows safe pattern.

        Args:
            v: The thread ID to validate

        Returns:
            str: The validated session ID

        Raises:
            ValueError: If the session ID is not valid
        """
        # Try to validate as UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a UUID, check for safe characters only
            if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
                raise ValueError("Session ID must contain only alphanumeric characters, underscores, and hyphens")
            return v
