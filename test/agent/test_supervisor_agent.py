# -*- coding: utf-8 -*-
# @Time   : 2025/8/31 22:30
# @Author : Galleons
# @File   : test_supervisor_agent.py

"""
Tests for the Supervisor Agent system.

This module contains comprehensive tests for the supervisor-based
multi-agent system, including task classification, routing, and
agent coordination.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

from app.core.agent.graph.supervisor_agent import build_supervisor_graph, SupervisorState, classify_task_simple
from app.core.agent.supervisor_config import TASK_CLASSIFICATION_KEYWORDS, AVAILABLE_AGENTS


class MockLLM:
    """Mock LLM for testing supervisor functionality."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = {}
    
    def with_structured_output(self, output_class):
        """Mock structured output binding."""
        mock_llm = AsyncMock()
        
        async def mock_ainvoke(messages):
            self.call_count += 1
            
            # Extract the user content for routing decisions
            user_content = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg.get("content", "").lower()
                    break
            
            # Mock responses based on output class and content
            if output_class.__name__ == "TaskType":
                if "研究" in user_content or "research" in user_content:
                    return output_class(
                        task_type="complex_research",
                        confidence=0.85,
                        reasoning="Contains research keywords"
                    )
                elif "步骤" in user_content or "step" in user_content:
                    return output_class(
                        task_type="multi_step",
                        confidence=0.8,
                        reasoning="Contains step-by-step keywords"
                    )
                else:
                    return output_class(
                        task_type="simple_qa",
                        confidence=0.9,
                        reasoning="Simple question format"
                    )
            
            elif output_class.__name__ == "SupervisorDecision":
                if "research" in user_content or "研究" in user_content:
                    return output_class(
                        assignments=[
                            output_class.__annotations__["assignments"].__args__[0](
                                agent_name="unified_agent",
                                task_description="Conduct research on the topic",
                                priority=1
                            )
                        ],
                        execution_mode="sequential",
                        reasoning="Research task assigned to unified agent"
                    )
                else:
                    return output_class(
                        assignments=[
                            output_class.__annotations__["assignments"].__args__[0](
                                agent_name="unified_agent",
                                task_description="Answer the question",
                                priority=1
                            )
                        ],
                        execution_mode="sequential",
                        reasoning="Simple QA task assigned to unified agent"
                    )
        
        mock_llm.ainvoke = mock_ainvoke
        return mock_llm


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM."""
    return MockLLM()


@pytest.fixture
def sample_state():
    """Fixture providing a sample supervisor state."""
    return SupervisorState(
        messages=[
            HumanMessage(content="什么是RAG系统？")
        ],
        session_id="test-session"
    )


class TestTaskClassification:
    """Test task classification functionality."""

    def test_simple_classification_qa(self):
        """Test simple classification for QA tasks."""
        user_input = "什么是机器学习？"
        task_type, confidence = classify_task_simple(user_input)

        assert task_type == "simple_qa"
        assert confidence > 0

    def test_simple_classification_research(self):
        """Test simple classification for research tasks."""
        user_input = "请分析比较不同的深度学习框架"
        task_type, confidence = classify_task_simple(user_input)

        assert task_type == "complex_research"
        assert confidence > 0

    def test_simple_classification_multi_step(self):
        """Test simple classification for multi-step tasks."""
        user_input = "请一步步教我如何搭建RAG系统"
        task_type, confidence = classify_task_simple(user_input)

        assert task_type == "multi_step"
        assert confidence > 0

    def test_fallback_classification(self):
        """Test fallback classification for unknown inputs."""
        user_input = "随机的输入内容"
        task_type, confidence = classify_task_simple(user_input)

        assert task_type == "simple_qa"  # Default fallback
        assert confidence == 0.5


class TestSupervisorConfig:
    """Test supervisor configuration functionality."""

    def test_available_agents(self):
        """Test available agents configuration."""
        assert "unified_agent" in AVAILABLE_AGENTS
        assert AVAILABLE_AGENTS["unified_agent"]["name"] == "unified_agent"
        assert "qa" in AVAILABLE_AGENTS["unified_agent"]["capabilities"]

    def test_task_keywords(self):
        """Test task classification keywords."""
        assert "simple_qa" in TASK_CLASSIFICATION_KEYWORDS
        assert "什么是" in TASK_CLASSIFICATION_KEYWORDS["simple_qa"]
        assert "分析" in TASK_CLASSIFICATION_KEYWORDS["complex_research"]


@pytest.mark.asyncio
class TestSupervisorGraph:
    """Test supervisor graph functionality."""
    
    async def test_supervisor_graph_creation(self, mock_llm):
        """Test supervisor graph creation."""
        graph = build_supervisor_graph(mock_llm)
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    async def test_simple_qa_routing(self, mock_llm, sample_state):
        """Test routing for simple QA tasks."""
        graph = build_supervisor_graph(mock_llm)
        
        result = await graph.ainvoke(sample_state)
        
        # Should have task classification
        assert "task_type" in result or result.get("agent_results")
        
        # Should have routed to an agent
        assert "agent_assignments" in result or result.get("agent_results")
    
    async def test_research_task_routing(self, mock_llm):
        """Test routing for research tasks."""
        state = SupervisorState(
            messages=[
                HumanMessage(content="请研究并分析当前AI发展趋势")
            ],
            session_id="test-session"
        )
        
        graph = build_supervisor_graph(mock_llm)
        result = await graph.ainvoke(state)
        
        # Should classify as research task
        # Should route appropriately
        assert result is not None
    
    async def test_multi_step_task_routing(self, mock_llm):
        """Test routing for multi-step tasks."""
        state = SupervisorState(
            messages=[
                HumanMessage(content="请一步步指导我搭建一个RAG系统")
            ],
            session_id="test-session"
        )
        
        graph = build_supervisor_graph(mock_llm)
        result = await graph.ainvoke(state)
        
        # Should classify as multi-step task
        # Should route appropriately
        assert result is not None
    
    async def test_error_handling(self, mock_llm):
        """Test error handling in supervisor system."""
        # Test with empty messages
        state = SupervisorState(
            messages=[],
            session_id="test-session"
        )
        
        graph = build_supervisor_graph(mock_llm)
        result = await graph.ainvoke(state)
        
        # Should handle gracefully
        assert result is not None
    
    async def test_fallback_behavior(self, mock_llm):
        """Test fallback behavior when classification fails."""
        # Mock LLM to raise exception
        mock_llm.with_structured_output = MagicMock(side_effect=Exception("Mock error"))
        
        state = SupervisorState(
            messages=[
                HumanMessage(content="Test question")
            ],
            session_id="test-session"
        )
        
        graph = build_supervisor_graph(mock_llm)
        
        # Should not raise exception, should fallback gracefully
        try:
            result = await graph.ainvoke(state)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Supervisor should handle errors gracefully, but got: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
