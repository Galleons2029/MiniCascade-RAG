# -*- coding: utf-8 -*-
# @Time   : 2025/8/13 17:32
# @Author : Galleons
# @File   : chief_agent.py

"""
This file contains the LangGraph Agent/workflow and interactions with the LLM.
"""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool

from app.configs import (
    Environment,
    llm_config,
)
from app.configs.agent_config import settings
from app.core.agent.tools import tools
from app.core.agent.graph.intent_agent import build_unified_agent_graph
from app.core.logger_utils import logger

# from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import SYSTEM_PROMPT
from app.models import (
    GraphState,
    Message,
)
from app.utils import (
    dump_messages,
)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Disable tiktoken for unsupported models like Qwen
        import os

        os.environ["TIKTOKEN_CACHE_DIR"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Disable LangChain's automatic token tracking
        os.environ["LANGCHAIN_CALLBACKS_MANAGER"] = "false"

        # CRITICAL: Monkey patch to completely disable token counting for Qwen models
        self._monkey_patch_token_counting()

        # Use environment-specific LLM model
        self.llm = ChatOpenAI(
            model=llm_config.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            base_url=llm_config.SILICON_BASE_URL,
            # Explicitly disable token counting and tracking
            default_headers={"User-Agent": "MiniCascade-RAG/1.0"},
            **self._get_model_kwargs(),
        ).bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

        logger.info("llm_initialized", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Completely disable token counting for Qwen models (not supported by LangChain)
        # This prevents any token-related errors without referencing other models
        model_kwargs["tiktoken_model_name"] = None

        # Development - can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    def _monkey_patch_token_counting(self):
        """Monkey patch to disable token counting for unsupported models like Qwen."""
        try:
            # Patch langchain_openai.chat_models.base to avoid token counting errors
            import langchain_openai.chat_models.base as base_module

            # Store original method
            if not hasattr(base_module, "_original_get_num_tokens_from_messages"):
                base_module._original_get_num_tokens_from_messages = getattr(
                    base_module.ChatOpenAI, "get_num_tokens_from_messages", None
                )

            # Replace with a dummy method that returns 0
            def dummy_get_num_tokens_from_messages(self, messages):
                """Dummy method to avoid token counting errors for unsupported models."""
                return 0

            # Apply the patch
            base_module.ChatOpenAI.get_num_tokens_from_messages = dummy_get_num_tokens_from_messages

            logger.info(
                "token_counting_monkey_patch_applied", message="Successfully disabled token counting for Qwen models"
            )

        except Exception as e:
            logger.warning("token_counting_monkey_patch_failed", error=str(e))

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                # Psycopg AsyncConnectionPool expects a libpq/psycopg DSN (e.g. postgresql:// or key=value),
                # not SQLAlchemy-style "postgresql+psycopg://". Convert if needed, or allow override via env.
                import os

                raw_url = os.getenv("POSTGRES_PG_DSN") or settings.POSTGRES_URL
                conn_dsn = raw_url.replace("+psycopg", "", 1) if "+psycopg" in raw_url else raw_url

                self._connection_pool = AsyncConnectionPool(
                    conn_dsn,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _chat(self, state: GraphState) -> dict:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            dict: Updated state with new messages.
        """
        # Simplified message preparation without token counting (Qwen models not supported)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in state.messages:
            if isinstance(msg, dict):
                # Message is already in dict format
                messages.append(msg)
            elif hasattr(msg, "model_dump"):
                # Message is a Pydantic model, convert to dict
                messages.append(msg.model_dump())
            else:
                # Fallback: convert to user message
                messages.append({"role": "user", "content": str(msg)})

        llm_calls_num = 0

        # Configure retry attempts based on environment
        max_retries = settings.MAX_LLM_CALL_RETRIES

        for attempt in range(max_retries):
            try:
                # with llm_inference_duration_seconds.labels(model=self.llm.model_name).time():
                generated_state = {"messages": [await self.llm.ainvoke(dump_messages(messages))]}
                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.LLM_MODEL,
                    environment=settings.ENVIRONMENT.value,
                )
                return generated_state
            except OpenAIError as e:
                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # In production, we might want to fall back to a more reliable model
                if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                    fallback_model = "gpt-4o"
                    logger.warning("using_fallback_model", model=fallback_model, environment=settings.ENVIRONMENT.value)
                    self.llm.model_name = fallback_model

                continue

        raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts")

    # (Removed: intent detection is now in intent_agent.py)

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> GraphState:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Dict with updated messages containing tool responses.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def _should_continue(self, state: GraphState) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
        """
        messages = state.messages
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)

                # Use the unified agent that handles intent detection, entity extraction,
                # context resolution, query rewrite, and RAG retrieval internally
                unified_agent = build_unified_agent_graph(self.llm)
                graph_builder.add_node("unified_agent", unified_agent)

                # Main chat/tool nodes
                graph_builder.add_node("chat", self._chat)
                graph_builder.add_node("tool_call", self._tool_call)

                # The unified agent handles all RAG processing internally based on intent.
                # After processing, we go directly to chat for final response generation.
                graph_builder.add_edge("unified_agent", "chat")

                # Chat <-> Tool loop and finish at chat
                graph_builder.add_conditional_edges("chat", self._should_continue, {"continue": "tool_call", "end": END})
                graph_builder.add_edge("tool_call", "chat")
                graph_builder.set_entry_point("unified_agent")
                graph_builder.set_finish_point("chat")

                # FORCE DISABLE: Skip PostgreSQL for testing
                checkpointer = None
                logger.warning("graph_creation", message="🔧 FORCED: PostgreSQL checkpointer disabled for testing")

                # TODO: Re-enable when PostgreSQL connection is fixed
                # Get connection pool (maybe None in production if DB unavailable)
                # try:
                #     connection_pool = await self._get_connection_pool()
                #     if connection_pool:
                #         checkpointer = AsyncPostgresSaver(connection_pool)
                #         await checkpointer.setup()
                #     else:
                #         checkpointer = None
                #         if settings.ENVIRONMENT != Environment.PRODUCTION:
                #             raise Exception("Connection pool initialization failed")
                # except Exception as e:
                #     logger.error("postgres_connection_failed", error=str(e))
                #     checkpointer = None

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
            },
        }
        try:
            response = await self._graph.ainvoke({"messages": dump_messages(messages), "session_id": session_id}, config)
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise e

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "session_id": session_id}, config, stream_mode="messages"
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
