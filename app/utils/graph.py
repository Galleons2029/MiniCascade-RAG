# -*- coding: utf-8 -*-
# @Time   : 2025/8/17 11:45
# @Author : Galleons
# @File   : graph.py

"""
This file contains the graph utilities for the application.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import trim_messages as _trim_messages

from app.configs.agent_config import settings
from app.models import Message


def dump_messages(messages) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages: The messages to dump (can be Message objects or dicts).

    Returns:
        list[dict]: The dumped messages.
    """
    result = []
    for message in messages:
        if isinstance(message, dict):
            # Already a dict, use as-is
            result.append(message)
        elif hasattr(message, "model_dump"):
            # Message object, convert to dict
            result.append(message.model_dump())
        else:
            # Fallback: assume it's a string content
            result.append({"role": "user", "content": str(message)})
    return result


def prepare_messages(messages: list[Message], llm: BaseChatModel, system_prompt: str) -> list[Message]:
    """Prepare the messages for the LLM.

    Args:
        messages (list[Message]): The messages to prepare.
        llm (BaseChatModel): The LLM to use.
        system_prompt (str): The system prompt to use.

    Returns:
        list[Message]: The prepared messages.
    """
    # Remove token_counter for Qwen models that aren't supported by LangChain
    trimmed_messages = _trim_messages(
        dump_messages(messages),
        strategy="last",
        max_tokens=settings.MAX_TOKENS,
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    return [Message(role="system", content=system_prompt)] + trimmed_messages
