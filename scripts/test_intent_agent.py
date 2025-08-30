#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Agent 测试运行脚本

这个脚本提供了多种测试 Intent Agent 的方式：
1. 基础单元测试
2. LangSmith 集成测试
3. 性能测试
4. 交互式测试

使用方法:
python scripts/test_intent_agent.py --test-type basic
python scripts/test_intent_agent.py --test-type langsmith
python scripts/test_intent_agent.py --test-type performance
python scripts/test_intent_agent.py --test-type interactive
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()


async def run_basic_tests():
    """运行基础单元测试"""
    print("🧪 运行基础单元测试...")
    
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "test/agent/test_intent_agent.py", 
        "-v", "--tb=short"
    ], cwd=project_root)
    
    return result.returncode == 0


async def run_langsmith_tests():
    """运行 LangSmith 集成测试"""
    print("🔍 运行 LangSmith 集成测试...")
    
    # 检查 LangSmith 配置
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ 请设置 LANGCHAIN_API_KEY 环境变量")
        print("   获取 API Key: https://smith.langchain.com/")
        return False
    
    try:
        from test.agent.test_intent_agent_langsmith import LangSmithTestRunner
        
        runner = LangSmithTestRunner()
        results = await runner.run_intent_test_suite()
        runner.generate_test_report(results)
        
        return True
    except Exception as e:
        print(f"❌ LangSmith 测试失败: {e}")
        return False


async def run_performance_tests():
    """运行性能测试"""
    print("⚡ 运行性能测试...")
    
    try:
        from test.agent.test_intent_performance import run_performance_tests
        
        results = await run_performance_tests()
        return results is not None
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


async def run_interactive_test():
    """运行交互式测试"""
    print("💬 启动交互式测试模式...")
    print("输入 'quit' 或 'exit' 退出")
    
    try:
        from langchain_openai import ChatOpenAI
        from app.core.agent.graph.intent_agent import build_unified_agent_graph
        
        # 初始化 LLM
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 需要设置 LLM_API_KEY 或 OPENAI_API_KEY")
            return False
        
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL")
        )
        
        graph = build_unified_agent_graph(llm)
        
        print("\n" + "="*50)
        print("🤖 Intent Agent 交互式测试")
        print("="*50)
        print("输入您的消息，系统将分析意图并显示处理结果")
        print()
        
        session_id = f"interactive-{asyncio.get_event_loop().time()}"
        context_frame = None
        
        while True:
            try:
                user_input = input("👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 分析中...")
                
                # 构建消息历史
                messages = [{"role": "user", "content": user_input}]
                
                # 调用 agent
                result = await graph.ainvoke({
                    "messages": messages,
                    "session_id": session_id,
                    "context_frame": context_frame
                })
                
                # 显示结果
                print("\n📊 分析结果:")
                print(f"   意图: {result.get('intent', 'unknown')}")
                print(f"   置信度: {result.get('intent_confidence', 0):.2f}")
                
                if result.get('entities'):
                    print(f"   实体: {result['entities']}")
                
                if result.get('time_text'):
                    print(f"   时间表达: {result['time_text']}")
                
                if result.get('rewritten_query'):
                    print(f"   改写查询: {result['rewritten_query']}")
                
                if result.get('context_docs'):
                    print(f"   检索文档数: {len(result['context_docs'])}")
                
                # 更新上下文
                context_frame = result.get('context_frame')
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ 交互式测试初始化失败: {e}")
        return False


def check_environment():
    """检查环境配置"""
    print("🔧 检查环境配置...")
    
    required_vars = ["LLM_API_KEY"]
    optional_vars = ["LANGCHAIN_API_KEY", "LANGFUSE_PUBLIC_KEY"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        print(f"❌ 缺少必需的环境变量: {', '.join(missing_required)}")
        return False
    
    print("✅ 必需的环境变量已设置")
    
    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_optional:
        print(f"⚠️  可选环境变量未设置: {', '.join(missing_optional)}")
        print("   这些变量用于高级功能（LangSmith 追踪、Langfuse 监控）")
    
    return True


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Intent Agent 测试工具")
    parser.add_argument(
        "--test-type", 
        choices=["basic", "langsmith", "performance", "interactive", "all"],
        default="basic",
        help="测试类型"
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="跳过环境检查"
    )
    
    args = parser.parse_args()
    
    print("🚀 Intent Agent 测试工具")
    print("=" * 40)
    
    # 环境检查
    if not args.skip_env_check:
        if not check_environment():
            print("\n💡 请检查 .env 文件或设置相应的环境变量")
            return
    
    # 运行测试
    success = True
    
    if args.test_type == "basic" or args.test_type == "all":
        success &= await run_basic_tests()
    
    if args.test_type == "langsmith" or args.test_type == "all":
        success &= await run_langsmith_tests()
    
    if args.test_type == "performance" or args.test_type == "all":
        success &= await run_performance_tests()
    
    if args.test_type == "interactive":
        success &= await run_interactive_test()
    
    if success:
        print("\n✅ 所有测试完成")
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
