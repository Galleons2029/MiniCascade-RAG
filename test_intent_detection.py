#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别测试脚本

测试 MiniCascade-RAG 项目中的意图识别功能
包括各种类型的用户输入和边界情况
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量（如果需要）
os.environ.setdefault("ENVIRONMENT", "development")

try:
    from app.core.agent.graph.intent_agent import build_intent_graph
    from app.models.graph import GraphState
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本，并且已安装所有依赖")
    sys.exit(1)

# 加载环境变量
load_dotenv()

class IntentTestCase:
    """意图测试用例"""
    
    def __init__(self, input_text: str, expected_intent: str, description: str):
        self.input_text = input_text
        self.expected_intent = expected_intent
        self.description = description

class IntentTester:
    """意图识别测试器"""
    
    def __init__(self):
        # 初始化 LLM
        self.llm = self._init_llm()
        # 构建意图识别图
        self.intent_graph = build_intent_graph(self.llm)
        
        # 定义测试用例
        self.test_cases = [
            # QA 类型
            IntentTestCase("什么是人工智能？", "qa", "知识问答"),
            IntentTestCase("请解释一下机器学习的原理", "qa", "技术解释"),
            IntentTestCase("北京的天气怎么样？", "qa", "信息查询"),
            
            # Search 类型  
            IntentTestCase("搜索最新的AI论文", "search", "搜索请求"),
            IntentTestCase("帮我找一下关于深度学习的资料", "search", "资料查找"),
            IntentTestCase("查找Python教程", "search", "教程搜索"),
            
            # Write 类型
            IntentTestCase("帮我写一个Python函数", "write", "代码编写"),
            IntentTestCase("写一份项目报告", "write", "文档写作"),
            IntentTestCase("生成一个邮件模板", "write", "模板生成"),
            
            # Exec 类型
            IntentTestCase("执行这段代码", "exec", "代码执行"),
            IntentTestCase("运行数据分析脚本", "exec", "脚本运行"),
            IntentTestCase("计算这个表达式", "exec", "计算请求"),
            
            # Smalltalk 类型
            IntentTestCase("你好", "smalltalk", "打招呼"),
            IntentTestCase("今天心情不错", "smalltalk", "闲聊"),
            IntentTestCase("谢谢你的帮助", "smalltalk", "感谢"),
            
            # Other 类型
            IntentTestCase("aslkjdflaskjdf", "other", "无意义输入"),
            IntentTestCase("", "other", "空输入"),
            IntentTestCase("？？？", "other", "模糊输入"),
        ]
    
    def _init_llm(self):
        """初始化 LLM"""
        try:
            # 尝试从环境变量获取配置
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("SILICON_KEY")
            model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            base_url = os.getenv("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
            
            if not api_key:
                print("警告: 未找到 API Key，使用模拟 LLM")
                return MockLLM()
            
            print(f"初始化 LLM: 模型={model}, API Base={base_url}")
            
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.1,
            )
        except Exception as e:
            print(f"LLM 初始化失败: {e}")
            # 创建一个模拟的 LLM 用于测试
            return MockLLM()
    
    async def test_single_intent(self, test_case: IntentTestCase) -> dict:
        """测试单个意图识别"""
        try:
            # 构建状态
            state = GraphState(
                messages=[{"role": "user", "content": test_case.input_text}],
                session_id="test-session"
            )
            
            # 执行意图识别
            result = await self.intent_graph.ainvoke(state.model_dump())
            
            detected_intent = result.get("intent", "unknown")
            confidence = result.get("intent_confidence", 0.0)
            
            return {
                "input": test_case.input_text,
                "expected": test_case.expected_intent,
                "detected": detected_intent,
                "confidence": confidence,
                "match": detected_intent == test_case.expected_intent,
                "description": test_case.description
            }
        except Exception as e:
            return {
                "input": test_case.input_text,
                "expected": test_case.expected_intent,
                "detected": "error",
                "confidence": 0.0,
                "match": False,
                "description": test_case.description,
                "error": str(e)
            }
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🤖 意图识别测试开始")
        print("=" * 60)
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i:2d}] 测试: {test_case.description}")
            print(f"     输入: '{test_case.input_text}'")
            
            result = await self.test_single_intent(test_case)
            results.append(result)
            
            # 显示结果
            status = "✅" if result["match"] else "❌"
            print(f"     预期: {result['expected']}")
            print(f"     检测: {result['detected']} (置信度: {result['confidence']:.2f})")
            print(f"     结果: {status}")
            
            if "error" in result:
                print(f"     错误: {result['error']}")
        
        # 统计结果
        self._print_summary(results)
    
    def _print_summary(self, results: list):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        
        total = len(results)
        passed = sum(1 for r in results if r["match"])
        failed = total - passed
        
        print(f"总测试数: {total}")
        print(f"通过数: {passed}")
        print(f"失败数: {failed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        # 按意图类型统计
        intent_stats = {}
        for result in results:
            intent = result["expected"]
            if intent not in intent_stats:
                intent_stats[intent] = {"total": 0, "passed": 0}
            intent_stats[intent]["total"] += 1
            if result["match"]:
                intent_stats[intent]["passed"] += 1
        
        print("\n各意图类型准确率:")
        for intent, stats in intent_stats.items():
            accuracy = stats["passed"] / stats["total"] * 100
            print(f"  {intent:10s}: {stats['passed']}/{stats['total']} ({accuracy:.1f}%)")
        
        # 显示失败案例
        failed_cases = [r for r in results if not r["match"]]
        if failed_cases:
            print("\n❌ 失败案例:")
            for case in failed_cases:
                print(f"  输入: '{case['input']}'")
                print(f"  预期: {case['expected']}, 检测: {case['detected']}")
                if "error" in case:
                    print(f"  错误: {case['error']}")
                print()

class MockLLM:
    """模拟 LLM（用于测试网络不可用的情况）"""
    
    async def ainvoke(self, messages):
        """模拟 LLM 调用"""
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "").lower()
                break
        
        # 简单的规则匹配
        if any(word in user_msg for word in ["什么", "怎么", "如何", "解释"]):
            intent = "qa"
        elif any(word in user_msg for word in ["搜索", "查找", "找"]):
            intent = "search"
        elif any(word in user_msg for word in ["写", "生成", "创建"]):
            intent = "write"
        elif any(word in user_msg for word in ["执行", "运行", "计算"]):
            intent = "exec"
        elif any(word in user_msg for word in ["你好", "谢谢", "心情"]):
            intent = "smalltalk"
        else:
            intent = "other"
        
        return MockResponse(f'{{"intent": "{intent}", "confidence": 0.8}}')

class MockResponse:
    """模拟 LLM 响应"""
    def __init__(self, content):
        self.content = content

async def main():
    """主函数"""
    try:
        tester = IntentTester()
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 启动意图识别测试...")
    asyncio.run(main())
