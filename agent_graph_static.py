#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态Agent图可视化

基于代码分析生成MiniCascade-RAG Agent图的Mermaid可视化，
无需运行实际的agent实例。
"""

def generate_mermaid_graph():
    """基于代码分析生成Mermaid图定义"""
    
    mermaid_def = """graph TD
    %% MiniCascade-RAG Agent Flow Diagram
    
    Start([开始]) --> DetectIntent[detect_intent<br/>意图检测]
    
    %% 意图路由
    DetectIntent --> RouteDecision{路由决策}
    
    %% QA/Write路径 - 完整RAG流程
    RouteDecision -->|qa/write| RouteRAG[route_rag<br/>RAG路由]
    RouteRAG --> EntityExtraction[entity_extraction<br/>实体提取]
    EntityExtraction --> ContextResolution[context_resolution<br/>上下文解析]
    ContextResolution --> QueryRewrite[query_rewrite<br/>查询改写]
    QueryRewrite --> RAGRetrieval[rag_retrieval<br/>RAG检索]
    RAGRetrieval --> Chat[chat<br/>对话生成]
    
    %% 其他路径 - 直接到对话
    RouteDecision -->|search| RouteSearch[route_search<br/>搜索路由]
    RouteDecision -->|exec| RouteExec[route_exec<br/>执行路由]
    RouteDecision -->|smalltalk| RouteSmallTalk[route_smalltalk<br/>闲聊路由]
    RouteDecision -->|other| RouteOther[route_other<br/>其他路由]
    
    RouteSearch --> Chat
    RouteExec --> Chat
    RouteSmallTalk --> Chat
    RouteOther --> Chat
    
    %% 工具调用循环
    Chat --> ShouldContinue{需要工具?}
    ShouldContinue -->|是| ToolCall[tool_call<br/>工具调用]
    ShouldContinue -->|否| End([结束])
    ToolCall --> Chat
    
    %% 样式定义
    classDef intentNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef ragNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef routeNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef chatNode fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef toolNode fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef startEndNode fill:#fce4ec,stroke:#880e4f,stroke-width:3px
    
    %% 应用样式
    class DetectIntent intentNode
    class EntityExtraction,ContextResolution,QueryRewrite,RAGRetrieval ragNode
    class RouteRAG,RouteSearch,RouteExec,RouteSmallTalk,RouteOther routeNode
    class Chat chatNode
    class ToolCall toolNode
    class Start,End startEndNode
    
    %% 添加注释
    subgraph Legend [图例]
        direction TB
        L1[意图检测] :::intentNode
        L2[RAG流程] :::ragNode
        L3[路由节点] :::routeNode
        L4[对话生成] :::chatNode
        L5[工具调用] :::toolNode
    end"""
    
    return mermaid_def

def generate_simple_ascii():
    """生成简化的ASCII图"""
    
    ascii_art = """
MiniCascade-RAG Agent 流程图:

┌─────────────┐
│    开始     │
└──────┬──────┘
       │
┌──────▼──────┐
│  意图检测   │
└──────┬──────┘
       │
┌──────▼──────┐
│  路由决策   │
└─┬─┬─┬─┬─┬───┘
  │ │ │ │ │
  │ │ │ │ └─── 其他路径 ───┐
  │ │ │ └───── 闲聊路径 ───┤
  │ │ └─────── 执行路径 ───┤
  │ └───────── 搜索路径 ───┤
  │                       │
  │ QA/Write 路径:        │
  │                       │
┌─▼──────────┐            │
│ 实体提取   │            │
└─┬──────────┘            │
  │                       │
┌─▼──────────┐            │
│ 上下文解析 │            │
└─┬──────────┘            │
  │                       │
┌─▼──────────┐            │
│ 查询改写   │            │
└─┬──────────┘            │
  │                       │
┌─▼──────────┐            │
│ RAG检索    │            │
└─┬──────────┘            │
  │                       │
┌─▼──────────┐◄───────────┘
│  对话生成  │
└─┬──────────┘
  │
┌─▼──────────┐
│ 需要工具？ │
└─┬────────┬─┘
  │是      │否
┌─▼──────┐ │
│工具调用│ │
└─┬──────┘ │
  │        │
  └────────┤
           │
        ┌──▼──┐
        │结束 │
        └─────┘
"""
    return ascii_art

def analyze_agent_components():
    """分析agent组件"""
    
    components = {
        "子图模块": [
            "intent_agent.py - 意图检测子图",
            "entity_agent.py - 实体提取子图", 
            "context_agent.py - 上下文解析子图",
            "rewrite_agent.py - 查询改写子图",
            "rag_agent.py - RAG检索子图"
        ],
        "路由节点": [
            "route_rag - QA/写作任务路由",
            "route_search - 搜索任务路由",
            "route_exec - 执行任务路由", 
            "route_smalltalk - 闲聊路由",
            "route_other - 其他任务路由"
        ],
        "核心节点": [
            "chat - 对话生成和管理",
            "tool_call - 工具调用处理"
        ],
        "流程特点": [
            "只有qa/write意图走完整RAG流程",
            "其他意图直接进入对话生成",
            "支持多轮工具调用",
            "使用LangGraph状态管理"
        ]
    }
    
    return components

def main():
    """主函数"""
    print("🎯 MiniCascade-RAG Agent图结构静态分析")
    print("=" * 60)
    
    # 生成Mermaid图
    print("\n📊 生成Mermaid图定义...")
    mermaid = generate_mermaid_graph()
    
    # 保存Mermaid文件
    with open("agent_graph_static.mermaid", "w", encoding="utf-8") as f:
        f.write(mermaid)
    
    print("✅ Mermaid图已保存到: agent_graph_static.mermaid")
    
    # 显示ASCII图
    print("\n📋 ASCII流程图:")
    ascii_art = generate_simple_ascii()
    print(ascii_art)
    
    # 分析组件
    print("\n🔍 Agent组件分析:")
    components = analyze_agent_components()
    
    for category, items in components.items():
        print(f"\n{category}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
    
    # 生成分析报告
    print("\n📝 生成分析报告...")
    report = """# MiniCascade-RAG Agent架构分析报告

## 概述
MiniCascade-RAG采用LangGraph框架构建的多智能体RAG系统，具有清晰的意图路由和处理流程。

## 核心流程

### 1. 意图检测阶段
- 入口点：`detect_intent` 
- 功能：分析用户输入，识别意图类型
- 输出：intent字段用于后续路由

### 2. 路由分发阶段
根据意图类型进行不同路由：
- **qa/write** → 完整RAG流程
- **search** → 直接对话
- **exec** → 直接对话  
- **smalltalk** → 直接对话
- **other** → 直接对话

### 3. RAG处理链（仅qa/write）
1. **entity_extraction**: 提取关键实体
2. **context_resolution**: 解析上下文信息
3. **query_rewrite**: 改写优化查询
4. **rag_retrieval**: 执行向量检索

### 4. 对话生成阶段
- **chat**: 基于上下文和检索结果生成回复
- **tool_call**: 必要时调用外部工具
- 支持多轮工具调用循环

## 设计特点

1. **意图驱动**: 不同意图类型采用不同处理策略
2. **模块化设计**: 每个子功能独立为子图
3. **灵活路由**: 支持多种任务类型
4. **工具集成**: 内置工具调用机制
5. **状态管理**: 使用LangGraph进行状态流转

## 使用建议

1. **性能优化**: 对于非qa/write任务，避免不必要的RAG流程
2. **扩展性**: 可以轻松添加新的意图类型和处理路径
3. **监控**: 建议添加每个节点的性能监控
4. **缓存**: 考虑在RAG检索阶段添加缓存机制

生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open("agent_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ 分析报告已保存到: agent_analysis_report.md")
    
    print("\n💾 生成的文件:")
    print("  - agent_graph_static.mermaid (Mermaid图定义)")
    print("  - agent_analysis_report.md (架构分析报告)")
    
    print("\n💡 使用提示:")
    print("  1. 复制Mermaid代码到 https://mermaid.live 查看交互图")
    print("  2. 查看分析报告了解详细架构说明")
    print("  3. 基于静态分析理解agent工作流程")
    
    print("\n" + "=" * 60)
    print("🎉 静态分析完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
