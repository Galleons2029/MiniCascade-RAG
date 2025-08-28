#!/usr/bin/env python3
"""
测试你项目中真正的LangGraph Agent
适配你的实际代码结构
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('ENVIRONMENT', 'development')

def load_env():
    """手动加载.env文件"""
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ 已加载环境变量: {env_file}")
    else:
        print(f"⚠️ 未找到.env文件: {env_file}")

async def test_your_real_agent():
    """测试你项目中真正的LangGraph Agent"""
    
    # 加载环境变量
    load_env()
    
    print("🤖 测试真实LangGraph Agent")
    print("=" * 60)
    
    try:
        # 导入你的真实Agent
        from app.core.agent.graph.chief_agent import LangGraphAgent
        from app.models import Message
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 测试Agent结构
    print("1️⃣ 测试Agent结构...")
    try:
        agent = LangGraphAgent()
        graph = await agent.create_graph()
        
        if graph is not None:
            print("   ✅ Agent创建成功")
            print(f"   ✅ 节点: {sorted(list(graph.nodes))}")
            
            # 检查预期节点
            expected_nodes = ['chat', 'tool_call']
            actual_nodes = list(graph.nodes)
            
            if all(node in actual_nodes for node in expected_nodes):
                print("   ✅ 所有预期节点都存在")
            else:
                missing = [n for n in expected_nodes if n not in actual_nodes]
                print(f"   ⚠️ 缺少节点: {missing}")
        else:
            print("   ❌ Agent创建失败")
            return
            
    except Exception as e:
        print(f"   ❌ Agent结构测试失败: {e}")
        return
    
    # 测试基本对话功能
    print("\n2️⃣ 测试基本对话...")
    simple_tests = [
        "你好",
        "你是谁？",
        "今天天气怎么样？",
        "请介绍一下自己"
    ]
    
    for i, test_msg in enumerate(simple_tests, 1):
        print(f"   [{i}] 测试: {test_msg}")
        try:
            messages = [Message(role="user", content=test_msg)]
            response = await agent.get_response(
                messages=messages,
                session_id=f"test-basic-{i}"
            )
            
            if response and len(response) > 0:
                content = response[-1].get('content', '')[:100]
                print(f"       ✅ 响应: {content}...")
            else:
                print("       ❌ 无响应")
                
        except Exception as e:
            print(f"       ❌ 错误: {str(e)[:100]}...")
    
    # 测试工具调用功能
    print("\n3️⃣ 测试工具调用...")
    tool_tests = [
        "搜索Python编程教程",
        "查找人工智能最新新闻",
        "搜索今天的天气",
    ]
    
    for i, test_msg in enumerate(tool_tests, 1):
        print(f"   [{i}] 测试: {test_msg}")
        try:
            messages = [Message(role="user", content=test_msg)]
            response = await agent.get_response(
                messages=messages,
                session_id=f"test-tool-{i}"
            )
            
            if response and len(response) > 0:
                content = response[-1].get('content', '')
                # 简单检查是否可能调用了工具
                has_search_content = any(keyword in content.lower() 
                                       for keyword in ['search', '搜索', 'found', '找到', 'result'])
                
                if has_search_content:
                    print("       ✅ 可能使用了搜索工具")
                else:
                    print("       ⚠️ 未明显使用搜索工具")
                    
                print(f"       响应: {content[:100]}...")
            else:
                print("       ❌ 无响应")
                
        except Exception as e:
            print(f"       ❌ 错误: {str(e)[:100]}...")
    
    # 测试会话历史
    print("\n4️⃣ 测试会话历史...")
    try:
        # 先发送几条消息
        messages1 = [Message(role="user", content="我叫张三")]
        await agent.get_response(messages1, "history-test")
        
        messages2 = [Message(role="user", content="我喜欢编程")]
        await agent.get_response(messages2, "history-test")
        
        # 获取历史
        history = await agent.get_chat_history("history-test")
        
        if history and len(history) > 0:
            print(f"   ✅ 历史记录: {len(history)} 条消息")
            for msg in history[-2:]:  # 显示最后2条
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:50]
                print(f"       {role}: {content}...")
        else:
            print("   ❌ 无历史记录")
            
    except Exception as e:
        print(f"   ❌ 历史记录测试失败: {e}")
    
    # 测试流式输出
    print("\n5️⃣ 测试流式输出...")
    try:
        messages = [Message(role="user", content="请写一首短诗")]
        print("   流式响应: ", end="", flush=True)
        
        response_parts = []
        async for token in agent.get_stream_response(
            messages=messages,
            session_id="stream-test"
        ):
            if token:
                print(token, end="", flush=True)
                response_parts.append(token)
        
        if response_parts:
            print(f"\n   ✅ 流式输出成功，共 {len(response_parts)} 个token")
        else:
            print("\n   ❌ 流式输出失败")
            
    except Exception as e:
        print(f"\n   ❌ 流式输出测试失败: {e}")
    
    print("\n📊 测试完成！")
    print("=" * 60)

async def quick_structure_test():
    """快速结构测试 - 等价于你之前的Docker命令"""
    try:
        from app.core.agent.graph.chief_agent import LangGraphAgent
        
        ag = LangGraphAgent()
        g = await ag.create_graph()
        
        print('compiled:', g is not None)
        if g:
            print('nodes:', sorted(list(g.nodes)))
        
    except Exception as e:
        print(f'❌ 错误: {e}')

def main():
    """主函数"""
    print("🚀 选择测试模式:")
    print("1. 快速结构测试 (等价于Docker命令)")
    print("2. 完整功能测试")
    
    choice = input("请选择 (1/2): ").strip()
    
    try:
        if choice == "1":
            print("\n🔧 运行快速结构测试...")
            asyncio.run(quick_structure_test())
        elif choice == "2":
            print("\n🧪 运行完整功能测试...")
            asyncio.run(test_your_real_agent())
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()