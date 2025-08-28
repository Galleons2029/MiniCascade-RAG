#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化MiniCascade-RAG Agent图结构

这个脚本用于可视化LangGraph agent的连接关系和工作流程
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.core.agent.graph.chief_agent import LangGraphAgent
    from app.models import GraphState
    print("✅ 成功导入依赖")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保项目依赖已正确安装")
    sys.exit(1)


async def visualize_agent_graph():
    """可视化agent图结构"""
    print("🚀 开始可视化Agent图结构...")
    
    try:
        # 创建LangGraph Agent实例
        agent = LangGraphAgent()
        print("✅ Agent实例创建成功")
        
        # 创建图
        graph = await agent.create_graph()
        if graph is None:
            print("❌ 图创建失败")
            return
            
        print("✅ 图创建成功")
        
        # 尝试不同的可视化方法
        print("\n📊 开始生成可视化...")
        
        # 方法1: ASCII可视化
        try:
            print("\n=== ASCII 图形 ===")
            ascii_graph = graph.get_graph().draw_ascii()
            print(ascii_graph)
        except Exception as e:
            print(f"ASCII可视化失败: {e}")
        
        # 方法2: Mermaid图
        try:
            print("\n=== Mermaid 图定义 ===")
            mermaid_def = graph.get_graph().draw_mermaid()
            print(mermaid_def)
            
            # 保存Mermaid图到文件
            with open("agent_graph.mermaid", "w", encoding="utf-8") as f:
                f.write(mermaid_def)
            print("✅ Mermaid图已保存到 agent_graph.mermaid")
            
        except Exception as e:
            print(f"Mermaid可视化失败: {e}")
        
        # 方法3: 如果有PNG依赖，尝试生成PNG
        try:
            print("\n=== 尝试生成PNG图片 ===")
            png_data = graph.get_graph().draw_mermaid_png()
            
            with open("agent_graph.png", "wb") as f:
                f.write(png_data)
            print("✅ PNG图片已保存到 agent_graph.png")
            
        except Exception as e:
            print(f"PNG生成失败: {e}")
            print("💡 提示: 安装 `playwright` 和 `kaleido` 可支持PNG生成")
        
        # 显示图的基本信息
        print(f"\n📈 图结构信息:")
        graph_info = graph.get_graph()
        print(f"  - 节点数量: {len(graph_info.nodes)}")
        print(f"  - 边数量: {len(graph_info.edges)}")
        print(f"  - 入口点: {graph_info.first_node}")
        
        print(f"\n🔍 节点列表:")
        for node_id in graph_info.nodes:
            print(f"  - {node_id}")
            
        print(f"\n🔗 边连接:")
        for edge in graph_info.edges:
            print(f"  - {edge.source} → {edge.target}")
        
    except Exception as e:
        print(f"❌ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    print("🎯 MiniCascade-RAG Agent图结构可视化工具")
    print("=" * 50)
    
    await visualize_agent_graph()
    
    print("\n" + "=" * 50)
    print("🎉 可视化完成!")
    print("\n💡 使用建议:")
    print("1. 查看生成的 agent_graph.mermaid 文件")
    print("2. 将Mermaid代码复制到 https://mermaid.live 查看交互式图形")
    print("3. 如果生成了PNG文件，可以直接查看图片")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
