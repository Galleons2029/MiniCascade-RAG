# -*- coding: utf-8 -*-
"""
LangSmith 集成测试 - Intent Agent 监控和调试

这个文件专门用于测试 Intent Agent 并通过 LangSmith 进行监控和调试。
运行前请确保设置了 LangSmith 环境变量。
"""

import os
import pytest
import asyncio
from datetime import datetime

# LangSmith 相关导入
from langsmith import Client
from langchain_openai import ChatOpenAI

from app.core.agent.graph.intent_agent import build_unified_agent_graph


class LangSmithTestRunner:
    """LangSmith 测试运行器"""

    def __init__(self):
        self.client = None
        self.setup_langsmith()

    def setup_langsmith(self):
        """设置 LangSmith 客户端"""
        try:
            # 检查环境变量
            if not os.getenv("LANGCHAIN_API_KEY"):
                print("⚠️  LANGCHAIN_API_KEY 未设置，将跳过 LangSmith 集成")
                return

            # 设置 LangSmith 环境变量
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "MiniCascade-RAG-Intent-Testing"

            # 获取项目信息用于生成正确的 URL
            self.project_name = "MiniCascade-RAG-Intent-Testing"

            self.client = Client()
            print("✅ LangSmith 客户端初始化成功")

        except Exception as e:
            print(f"❌ LangSmith 初始化失败: {e}")

    def create_test_llm(self):
        """创建用于测试的 LLM"""
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要设置 LLM_API_KEY 或 OPENAI_API_KEY")

        # 使用 LangSmith 包装的 LLM
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),  # 支持自定义 base_url
        )

        return llm

    async def run_intent_test_suite(self):
        """运行完整的意图测试套件"""
        if not self.client:
            print("⚠️  LangSmith 未配置，跳过测试")
            return

        llm = self.create_test_llm()
        graph = build_unified_agent_graph(llm)

        # 测试用例
        test_cases = [
            {"name": "QA_Intent_Simple", "message": "什么是RAG系统？", "expected_intent": "qa"},
            {"name": "QA_Intent_Complex", "message": "上周的销售数据显示了什么趋势？", "expected_intent": "qa"},
            {"name": "Write_Intent", "message": "请帮我写一份关于AI发展的报告", "expected_intent": "write"},
            {"name": "Search_Intent", "message": "搜索最新的机器学习论文", "expected_intent": "search"},
            {"name": "Exec_Intent", "message": "执行数据备份任务", "expected_intent": "exec"},
            {"name": "Smalltalk_Intent", "message": "你好，今天天气怎么样？", "expected_intent": "smalltalk"},
            {
                "name": "Multi_Turn_Context",
                "messages": [
                    {"role": "user", "content": "这个月的销售额是多少？"},
                    {"role": "assistant", "content": "这个月的销售额是100万元"},
                    {"role": "user", "content": "上个月的呢？"},
                ],
                "expected_intent": "qa",
            },
        ]

        results = []

        for test_case in test_cases:
            print(f"🧪 运行测试: {test_case['name']}")

            try:
                # 准备输入
                if "messages" in test_case:
                    messages = test_case["messages"]
                else:
                    messages = [{"role": "user", "content": test_case["message"]}]

                # 运行测试（会自动记录到 LangSmith）
                # 生成符合验证规则的 session_id（只包含字母数字、下划线和连字符）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 移除微秒的后3位
                session_id = f"test-{test_case['name']}-{timestamp}"

                result = await graph.ainvoke({"messages": messages, "session_id": session_id})

                # 验证结果
                actual_intent = result.get("intent", "unknown")
                expected_intent = test_case["expected_intent"]

                test_result = {
                    "test_name": test_case["name"],
                    "expected_intent": expected_intent,
                    "actual_intent": actual_intent,
                    "confidence": result.get("intent_confidence", 0),
                    "success": actual_intent == expected_intent,
                    "full_result": result,
                }

                results.append(test_result)

                status = "✅" if test_result["success"] else "❌"
                print(f"{status} {test_case['name']}: {actual_intent} (置信度: {test_result['confidence']:.2f})")

            except Exception as e:
                print(f"❌ 测试失败 {test_case['name']}: {e}")
                results.append({"test_name": test_case["name"], "error": str(e), "success": False})

        return results

    def generate_test_report(self, results):
        """生成测试报告"""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get("success", False))

        print("\n" + "=" * 60)
        print("📊 Intent Agent 测试报告")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {total_tests - successful_tests}")
        print(f"成功率: {successful_tests / total_tests * 100:.1f}%")

        print("\n📋 详细结果:")
        for result in results:
            if result.get("success"):
                print(f"✅ {result['test_name']}: {result['actual_intent']} (置信度: {result.get('confidence', 0):.2f})")
            else:
                expected = result.get("expected_intent")
                actual = result.get("actual_intent")
                error_msg = result.get("error", f"预期: {expected}, 实际: {actual}")
                print(f"❌ {result['test_name']}: {error_msg}")

        if self.client:
            try:
                # 获取正确的项目 URL
                project_id = self.project_name.replace("-", "_").lower()
                project_url = f"https://smith.langchain.com/o/default/projects/p/{project_id}"
                print(f"\n🔗 查看详细追踪: {project_url}")
                print("💡 如果链接无法访问，请直接访问 https://smith.langchain.com/ 并查找项目")
            except Exception:
                print("\n🔗 查看详细追踪: https://smith.langchain.com/")
                print("💡 请在 LangSmith Dashboard 中查找项目: MiniCascade-RAG-Intent-Testing")


@pytest.mark.asyncio
async def test_intent_agent_with_langsmith():
    """使用 LangSmith 测试 Intent Agent"""
    runner = LangSmithTestRunner()

    if not runner.client:
        pytest.skip("LangSmith 未配置，跳过测试")

    results = await runner.run_intent_test_suite()
    runner.generate_test_report(results)

    # 断言至少有一些测试成功
    successful_tests = sum(1 for r in results if r.get("success", False))
    assert successful_tests > 0, "至少应该有一个测试成功"


@pytest.mark.asyncio
async def test_rag_pipeline_tracing():
    """测试完整 RAG 流程的追踪"""
    runner = LangSmithTestRunner()

    if not runner.client:
        pytest.skip("LangSmith 未配置，跳过测试")

    llm = runner.create_test_llm()

    # Mock VectorRetriever for testing
    from unittest.mock import patch, MagicMock

    with patch("app.core.agent.graph.intent_agent.VectorRetriever") as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.multi_query.return_value = ["扩展查询1", "扩展查询2", "扩展查询3"]
        mock_retriever.retrieve_top_k.return_value = ["文档片段1", "文档片段2"]
        mock_retriever.rerank.return_value = ["重排后片段1", "重排后片段2"]
        mock_retriever_class.return_value = mock_retriever

        graph = build_unified_agent_graph(llm)

        # 运行完整的 RAG 流程
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "上周的销售数据分析报告"}],
                "session_id": f"rag-pipeline-test-{timestamp}",
            }
        )

        # 验证 RAG 流程的各个步骤
        assert "intent" in result
        assert "entities" in result
        assert "context_frame" in result
        assert "rewritten_query" in result
        assert "context_docs" in result

        print("✅ RAG 流程追踪测试完成")


if __name__ == "__main__":
    """直接运行测试"""

    async def main():
        runner = LangSmithTestRunner()
        results = await runner.run_intent_test_suite()
        runner.generate_test_report(results)

    asyncio.run(main())
