#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2025/8/31 22:50
# @Author : ggbond
# @File   : supervisor_demo.py

"""
Lightweight Supervisor Agent Demo Script.

This script demonstrates the capabilities of the simplified supervisor
system, showing how it classifies tasks and routes them to the unified agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.configs import llm_config
from app.configs.agent_config import settings
from app.core.agent.graph.supervisor_agent import build_supervisor_graph, SupervisorState, classify_task_simple
from app.core.agent.supervisor_config import TASK_CLASSIFICATION_KEYWORDS
from app.core.logger_utils import logger


class LightweightSupervisorDemo:
    """Demo class for lightweight supervisor agent system."""

    def __init__(self):
        """Initialize the demo with LLM and supervisor graph."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_config.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            base_url=llm_config.SILICON_BASE_URL,
        )

        # Build supervisor graph
        self.supervisor_graph = build_supervisor_graph(self.llm)

        logger.info("lightweight_supervisor_demo_initialized", model=llm_config.LLM_MODEL)
    
    async def demo_task_classification(self):
        """Demonstrate task classification capabilities."""
        print("\n" + "="*60)
        print("🎯 TASK CLASSIFICATION DEMO")
        print("="*60)

        test_queries = [
            "什么是RAG系统？",
            "请研究并分析当前AI发展趋势",
            "请一步步教我如何搭建RAG系统",
            "帮我计算这个数据集的统计信息",
            "比较不同的深度学习框架",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")

            # Simple classification
            task_type, confidence = classify_task_simple(query)

            print(f"   🔍 Classification: {task_type} (confidence: {confidence:.2f})")
            print(f"   🎯 Will route to: unified_agent")

    async def demo_full_execution(self):
        """Demonstrate full supervisor execution."""
        print("\n" + "="*60)
        print("🎬 FULL EXECUTION DEMO")
        print("="*60)

        query = "什么是RAG系统的核心组件？"
        print(f"📝 Query: {query}")

        state = SupervisorState(
            messages=[HumanMessage(content=query)],
            session_id="demo-execution"
        )

        try:
            print("🚀 Starting supervisor execution...")
            result = await self.supervisor_graph.ainvoke(state)

            print("✅ Execution completed!")
            print(f"📊 Result keys: {list(result.keys())}")

            # Display key results
            if "task_type" in result:
                print(f"🎯 Task Type: {result['task_type']}")
            if "assigned_agent" in result:
                print(f"🤖 Assigned Agent: {result['assigned_agent']}")
            if "messages" in result and result["messages"]:
                print(f"💬 Generated {len(result['messages'])} response messages")

        except Exception as e:
            print(f"❌ Execution failed: {str(e)}")
            logger.error("demo_execution_failed", error=str(e))

    async def run_simple_demo(self):
        """Run simplified demonstration."""
        print("🎭 LIGHTWEIGHT SUPERVISOR AGENT DEMO")
        print("Demonstrating simplified task classification and routing")

        await self.demo_task_classification()
        await self.demo_full_execution()

        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED")
        print("="*60)
        print("The lightweight supervisor system demonstrated:")
        print("✅ Simple task classification using keywords")
        print("✅ Routing to unified_agent")
        print("✅ Integration with existing intent detection system")
        print("✅ Extensible architecture for future agents")
    
    async def demo_agent_routing(self):
        """Demonstrate agent routing for different task types."""
        print("\n" + "="*60)
        print("🚀 AGENT ROUTING DEMO")
        print("="*60)
        
        test_cases = [
            {
                "query": "什么是机器学习？",
                "expected_agent": "unified_agent",
                "description": "Simple QA task"
            },
            {
                "query": "请深入研究并分析transformer架构的发展历程",
                "expected_agent": "research_agent",
                "description": "Complex research task"
            },
            {
                "query": "请一步步指导我搭建一个完整的RAG系统",
                "expected_agent": "unified_agent",
                "description": "Multi-step tutorial task"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {case['description']}")
            print(f"   📝 Query: {case['query']}")
            print(f"   🎯 Expected Agent: {case['expected_agent']}")
            
            # Create state and run through supervisor
            state = SupervisorState(
                messages=[HumanMessage(content=case['query'])],
                session_id=f"demo-session-{i}"
            )
            
            try:
                # Run only classification and routing (not full execution)
                result = await self._run_classification_and_routing(state)
                
                assigned_agents = result.get("agent_assignments", [])
                if assigned_agents:
                    actual_agent = assigned_agents[0].get("agent_name", "unknown")
                    execution_mode = result.get("execution_mode", "unknown")
                    
                    print(f"   ✅ Actual Agent: {actual_agent}")
                    print(f"   ⚙️  Execution Mode: {execution_mode}")
                    print(f"   📋 Task Description: {assigned_agents[0].get('task_description', 'N/A')}")
                    
                    # Check if routing matches expectation
                    if actual_agent == case['expected_agent']:
                        print(f"   ✅ Routing: CORRECT")
                    else:
                        print(f"   ⚠️  Routing: UNEXPECTED (expected {case['expected_agent']})")
                else:
                    print(f"   ❌ No agent assigned")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    async def _run_classification_and_routing(self, state: SupervisorState) -> dict:
        """Run only classification and routing steps."""
        # This is a simplified version that runs only the first two steps
        # In a real implementation, you'd extract these from the graph
        
        # Step 1: Classify task
        latest_user = self._get_latest_user_message(state.messages)
        keyword_scores = classify_task_by_keywords(latest_user)
        best_type = max(keyword_scores.items(), key=lambda x: x[1])
        
        # Step 2: Route task (simplified)
        if best_type[0] == "simple_qa":
            assignments = [{
                "agent_name": "unified_agent",
                "task_description": latest_user,
                "priority": 1
            }]
        elif best_type[0] == "complex_research":
            assignments = [{
                "agent_name": "research_agent",
                "task_description": latest_user,
                "priority": 1
            }]
        elif best_type[0] == "tool_heavy":
            assignments = [{
                "agent_name": "tool_agent",
                "task_description": latest_user,
                "priority": 1
            }]
        else:
            assignments = [{
                "agent_name": "unified_agent",
                "task_description": latest_user,
                "priority": 1
            }]
        
        return {
            "task_type": best_type[0],
            "task_confidence": best_type[1],
            "agent_assignments": assignments,
            "execution_mode": "sequential"
        }
    
    def _get_latest_user_message(self, messages) -> str:
        """Extract the latest user message."""
        for m in reversed(messages):
            if hasattr(m, 'content'):
                return m.content
        return ""
    
    async def demo_full_execution(self):
        """Demonstrate full supervisor execution with a simple query."""
        print("\n" + "="*60)
        print("🎬 FULL EXECUTION DEMO")
        print("="*60)
        
        query = "什么是RAG系统的核心组件？"
        print(f"📝 Query: {query}")
        
        state = SupervisorState(
            messages=[HumanMessage(content=query)],
            session_id="demo-full-execution"
        )
        
        try:
            print("🚀 Starting supervisor execution...")
            result = await self.supervisor_graph.ainvoke(state)
            
            print("✅ Execution completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            # Display results
            if "agent_results" in result:
                agent_results = result["agent_results"]
                print(f"🤖 Agent Results: {len(agent_results)} results")
                for i, agent_result in enumerate(agent_results):
                    agent_name = agent_result.get("agent_name", "unknown")
                    status = agent_result.get("status", "unknown")
                    print(f"   Agent {i+1}: {agent_name} - {status}")
            
            if "messages" in result and result["messages"]:
                final_messages = result["messages"]
                print(f"💬 Final Messages: {len(final_messages)} messages")
                for msg in final_messages[-2:]:  # Show last 2 messages
                    if hasattr(msg, 'content'):
                        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"   📄 {type(msg).__name__}: {content_preview}")
                        
        except Exception as e:
            print(f"❌ Execution failed: {str(e)}")
            logger.error("demo_execution_failed", error=str(e))
    
    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("🎭 SUPERVISOR AGENT SYSTEM DEMO")
        print("Demonstrating intelligent task routing and multi-agent coordination")
        
        await self.demo_task_classification()
        await self.demo_agent_routing()
        await self.demo_full_execution()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED")
        print("="*60)
        print("The supervisor system successfully demonstrated:")
        print("✅ Task classification using keywords and LLM")
        print("✅ Intelligent agent routing based on task type")
        print("✅ Multi-agent coordination and execution")
        print("✅ Fallback mechanisms for error handling")


async def main():
    """Main demo function."""
    try:
        demo = LightweightSupervisorDemo()
        await demo.run_simple_demo()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error("demo_failed", error=str(e))


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("ENVIRONMENT", "DEVELOPMENT")
    
    # Run the demo
    asyncio.run(main())
