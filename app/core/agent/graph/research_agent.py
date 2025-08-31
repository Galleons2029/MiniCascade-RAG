# -*- coding: utf-8 -*-
# @Time   : 2025/8/31 22:35
# @Author : ggbond
# @File   : research_agent.py

"""
Research Agent for MiniCascade-RAG.

This module implements a specialized research agent that focuses on
deep research, analysis, and comprehensive information gathering tasks.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph

from app.core.logger_utils import logger
from app.models import GraphState
from app.core.rag.retriever import VectorRetriever
from app.configs.rag_config import rag_settings


class ResearchQuery(BaseModel):
    """Research query for information gathering."""
    query: str = Field(description="Search query for research")
    focus_area: str = Field(description="Specific focus area or aspect")
    priority: int = Field(description="Priority level (1-5)")


class ResearchPlan(BaseModel):
    """Research plan with multiple queries."""
    queries: List[ResearchQuery] = Field(description="List of research queries")
    research_approach: str = Field(description="Overall research approach")
    expected_depth: str = Field(description="Expected depth of research (shallow/moderate/deep)")


class ResearchFindings(BaseModel):
    """Research findings and analysis."""
    summary: str = Field(description="Summary of research findings")
    key_points: List[str] = Field(description="Key points discovered")
    sources: List[str] = Field(description="Source URLs or references")
    confidence_score: float = Field(description="Confidence in findings (0-1)")


class ResearchState(GraphState):
    """Extended state for research operations."""
    research_plan: Optional[Dict[str, Any]] = None
    research_queries: List[Dict[str, Any]] = Field(default_factory=list)
    research_results: List[Dict[str, Any]] = Field(default_factory=list)
    research_findings: Optional[Dict[str, Any]] = None
    research_status: str = "planning"  # planning, researching, analyzing, completed


def build_research_agent_graph(llm) -> CompiledStateGraph:
    """Build a specialized research agent graph.
    
    Args:
        llm: A chat model instance for the research agent
        
    Returns:
        CompiledStateGraph: A compiled research agent graph
    """
    
    async def _plan_research(state: ResearchState) -> dict:
        """Step 1: Plan the research approach and queries."""
        latest_user = _get_latest_user_message(state.messages)
        if not latest_user:
            return {"research_status": "failed"}
        
        system_prompt = """You are a research planning expert. Given a research request, create a \
comprehensive research plan.

Your task is to:
1. Analyze the research request and identify key aspects to investigate
2. Generate 3-5 specific research queries that will gather comprehensive information
3. Determine the appropriate research approach and depth

Guidelines:
- Create diverse queries that cover different aspects of the topic
- Prioritize queries by importance (1 = highest priority)
- Consider both broad overview and specific details
- Focus on authoritative and recent information

Research approaches:
- **broad_survey**: Wide coverage of the topic with multiple perspectives
- **deep_dive**: Focused, in-depth analysis of specific aspects
- **comparative**: Comparison between different options/approaches
- **trend_analysis**: Analysis of trends and developments over time"""
        
        user_prompt = f"Create a research plan for: {latest_user}"
        
        try:
            response = await llm.with_structured_output(ResearchPlan).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            research_plan = {
                "queries": [q.dict() for q in response.queries],
                "approach": response.research_approach,
                "depth": response.expected_depth,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info("research_planned", 
                       num_queries=len(response.queries),
                       approach=response.research_approach,
                       depth=response.expected_depth)
            
            return {
                "research_plan": research_plan,
                "research_queries": [q.dict() for q in response.queries],
                "research_status": "researching"
            }
        except Exception as e:
            logger.error("research_planning_failed", error=str(e))
            return {"research_status": "failed"}
    
    async def _execute_research(state: ResearchState) -> dict:
        """Step 2: Execute research queries and gather information."""
        queries = state.research_queries or []
        if not queries:
            return {"research_status": "failed"}
        
        research_results = []
        
        for query_info in queries:
            query = query_info.get("query", "")
            focus_area = query_info.get("focus_area", "")
            
            try:
                # Use VectorRetriever for research
                retriever = VectorRetriever(query=query)
                
                # Generate multiple search queries
                generated_queries = retriever.multi_query(to_expand_to_n_queries=3)
                if hasattr(generated_queries, '__iter__') and not isinstance(generated_queries, (str, bytes)):
                    generated_queries = list(generated_queries)
                
                # Retrieve documents
                hits = retriever.retrieve_top_k(
                    k=rag_settings.TOP_K,
                    collections=["default"],  # Use default collection
                    generated_queries=generated_queries
                )
                
                # Rerank results
                passages = retriever.rerank(hits=hits, keep_top_k=rag_settings.KEEP_TOP_K)
                if hasattr(passages, '__iter__') and not isinstance(passages, (str, bytes)):
                    passages = list(passages)
                
                result = {
                    "query": query,
                    "focus_area": focus_area,
                    "passages": passages,
                    "num_results": len(passages),
                    "status": "completed"
                }
                
                research_results.append(result)
                
                logger.info("research_query_completed", 
                           query=query,
                           num_results=len(passages))
                
            except Exception as e:
                logger.error("research_query_failed", query=query, error=str(e))
                research_results.append({
                    "query": query,
                    "focus_area": focus_area,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "research_results": research_results,
            "research_status": "analyzing"
        }
    
    async def _analyze_findings(state: ResearchState) -> dict:
        """Step 3: Analyze research results and synthesize findings."""
        research_results = state.research_results or []
        if not research_results:
            return {"research_status": "failed"}
        
        # Combine all research passages
        all_passages = []
        successful_queries = []
        
        for result in research_results:
            if result.get("status") == "completed":
                passages = result.get("passages", [])
                all_passages.extend(passages)
                successful_queries.append(result.get("query", ""))
        
        if not all_passages:
            return {"research_status": "failed"}
        
        # Create analysis prompt
        context = "\n\n".join(all_passages[:10])  # Limit context size
        original_request = _get_latest_user_message(state.messages)
        
        system_prompt = """You are a research analyst. Analyze the gathered research information and \
provide comprehensive findings.

Your task is to:
1. Synthesize the information from multiple sources
2. Identify key points and insights
3. Provide a clear summary of findings
4. Assess the confidence level of your analysis

Guidelines:
- Focus on the most relevant and important information
- Identify patterns and connections across sources
- Be objective and evidence-based
- Note any limitations or gaps in the research"""
        
        user_prompt = f"""Original research request: {original_request}

Research queries executed: {', '.join(successful_queries)}

Research context:
{context}

Please analyze these findings and provide a comprehensive summary."""
        
        try:
            response = await llm.with_structured_output(ResearchFindings).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            findings = {
                "summary": response.summary,
                "key_points": response.key_points,
                "sources": response.sources,
                "confidence_score": response.confidence_score,
                "num_sources_analyzed": len(all_passages),
                "successful_queries": len(successful_queries),
                "completed_at": datetime.now().isoformat()
            }
            
            # Create final response message
            final_response = f"""# 研究分析结果

## 摘要
{response.summary}

## 关键发现
{chr(10).join(f"• {point}" for point in response.key_points)}

## 信息来源
分析了 {len(all_passages)} 个信息源，执行了 {len(successful_queries)} 个研究查询。

置信度评分: {response.confidence_score:.2f}/1.0"""
            
            logger.info("research_analysis_completed", 
                       confidence=response.confidence_score,
                       num_key_points=len(response.key_points),
                       num_sources=len(all_passages))
            
            return {
                "research_findings": findings,
                "research_status": "completed",
                "messages": [AIMessage(content=final_response)]
            }
            
        except Exception as e:
            logger.error("research_analysis_failed", error=str(e))
            return {"research_status": "failed"}
    
    def _get_latest_user_message(messages) -> Optional[str]:
        """Extract the latest user message from the conversation."""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
            elif isinstance(m, HumanMessage):
                return m.content
        return None
    
    # Build the research agent graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("plan_research", _plan_research)
    graph.add_node("execute_research", _execute_research)
    graph.add_node("analyze_findings", _analyze_findings)
    
    # Add edges
    graph.add_edge(START, "plan_research")
    graph.add_edge("plan_research", "execute_research")
    graph.add_edge("execute_research", "analyze_findings")
    graph.add_edge("analyze_findings", END)
    
    return graph.compile(name="ResearchAgent")
