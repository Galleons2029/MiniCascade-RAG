# -*- coding: utf-8 -*-
# @Time   : 2025/8/31 22:20
# @Author : ggbond
# @File   : supervisor_config.py

"""
Lightweight configuration for the Supervisor Agent system.

This module contains basic configuration settings for task classification
and agent routing in the simplified supervisor system.
"""

from typing import Dict, List

# Task classification keywords for different task types
TASK_CLASSIFICATION_KEYWORDS = {
    "simple_qa": [
        "what is", "who is", "when", "where", "how much", "define", "explain briefly",
        "什么是", "谁是", "何时", "哪里", "多少", "定义", "简单解释"
    ],
    "complex_research": [
        "analyze", "compare", "research", "investigate", "comprehensive", "detailed analysis",
        "pros and cons", "advantages and disadvantages", "in-depth",
        "分析", "比较", "研究", "调查", "综合", "详细分析", "优缺点", "深入"
    ],
    "multi_step": [
        "step by step", "workflow", "process", "procedure", "how to", "tutorial",
        "first", "then", "next", "finally", "sequence",
        "步骤", "工作流", "流程", "程序", "如何", "教程", "首先", "然后", "接下来", "最后", "顺序"
    ],
    "tool_heavy": [
        "calculate", "compute", "process data", "analyze file", "generate report",
        "extract", "transform", "load", "api", "database",
        "计算", "处理数据", "分析文件", "生成报告", "提取", "转换", "加载", "接口", "数据库"
    ]
}

# Available agents and their descriptions
AVAILABLE_AGENTS = {
    "unified_agent": {
        "name": "unified_agent",
        "description": "General-purpose agent with intent detection, RAG, and context awareness",
        "capabilities": ["qa", "research", "context_aware", "rag", "multi_step"]
    }
    # Future agents can be added here:
    # "custom_agent": {
    #     "name": "custom_agent",
    #     "description": "Custom agent for specific tasks",
    #     "capabilities": ["custom_capability"]
    # }
}


