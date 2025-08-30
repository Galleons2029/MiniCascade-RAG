# -*- coding: utf-8 -*-
"""
Intent Agent 性能测试

测试 Intent Agent 的响应时间、准确率和资源使用情况
"""

import time
import asyncio
import statistics
from typing import List, Dict
from datetime import datetime

from app.core.agent.graph.intent_agent import build_unified_agent_graph
from langchain_openai import ChatOpenAI


class PerformanceTestSuite:
    """性能测试套件"""

    def __init__(self, llm):
        self.llm = llm
        self.graph = build_unified_agent_graph(llm)
        self.results = []

    async def test_response_time(self, test_cases: List[Dict], iterations: int = 5):
        """测试响应时间"""
        print(f"🚀 开始响应时间测试 ({iterations} 次迭代)")

        for test_case in test_cases:
            case_times = []

            for i in range(iterations):
                start_time = time.time()

                try:
                    await self.graph.ainvoke(
                        {"messages": [{"role": "user", "content": test_case["message"]}], "session_id": f"perf-test-{i}"}
                    )

                    end_time = time.time()
                    response_time = end_time - start_time
                    case_times.append(response_time)

                except Exception as e:
                    print(f"❌ 测试失败: {e}")
                    continue

            if case_times:
                avg_time = statistics.mean(case_times)
                min_time = min(case_times)
                max_time = max(case_times)

                self.results.append(
                    {
                        "test_case": test_case["name"],
                        "message": test_case["message"],
                        "avg_response_time": avg_time,
                        "min_response_time": min_time,
                        "max_response_time": max_time,
                        "iterations": len(case_times),
                    }
                )

                print(f"📊 {test_case['name']}: 平均 {avg_time:.2f}s (最小: {min_time:.2f}s, 最大: {max_time:.2f}s)")

    async def test_accuracy(self, test_cases: List[Dict]):
        """测试准确率"""
        print("🎯 开始准确率测试")

        correct_predictions = 0
        total_predictions = 0

        for test_case in test_cases:
            try:
                result = await self.graph.ainvoke(
                    {
                        "messages": [{"role": "user", "content": test_case["message"]}],
                        "session_id": f"accuracy-test-{total_predictions}",
                    }
                )

                predicted_intent = result.get("intent", "unknown")
                expected_intent = test_case["expected_intent"]

                if predicted_intent == expected_intent:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"

                total_predictions += 1

                print(f"{status} {test_case['name']}: 预期 {expected_intent}, 实际 {predicted_intent}")

            except Exception as e:
                print(f"❌ 测试失败 {test_case['name']}: {e}")
                total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"📈 总体准确率: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

        return accuracy

    async def test_concurrent_requests(self, message: str, concurrent_count: int = 10):
        """测试并发请求处理能力"""
        print(f"⚡ 开始并发测试 ({concurrent_count} 个并发请求)")

        async def single_request(request_id: int):
            start_time = time.time()
            try:
                result = await self.graph.ainvoke(
                    {"messages": [{"role": "user", "content": message}], "session_id": f"concurrent-test-{request_id}"}
                )
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time": end_time - start_time,
                    "intent": result.get("intent", "unknown"),
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e),
                }

        # 创建并发任务
        tasks = [single_request(i) for i in range(concurrent_count)]

        # 执行并发请求
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # 分析结果
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        if successful_requests:
            avg_response_time = statistics.mean([r["response_time"] for r in successful_requests])
            throughput = len(successful_requests) / total_time
        else:
            avg_response_time = 0
            throughput = 0

        print("📊 并发测试结果:")
        print(f"   总请求数: {concurrent_count}")
        print(f"   成功请求: {len(successful_requests)}")
        print(f"   失败请求: {len(failed_requests)}")
        print(f"   平均响应时间: {avg_response_time:.2f}s")
        print(f"   吞吐量: {throughput:.2f} 请求/秒")
        print(f"   总耗时: {total_time:.2f}s")

        return {
            "total_requests": concurrent_count,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "avg_response_time": avg_response_time,
            "throughput": throughput,
            "total_time": total_time,
        }

    def generate_performance_report(self):
        """生成性能报告"""
        print("\n" + "=" * 60)
        print("📊 Intent Agent 性能测试报告")
        print("=" * 60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.results:
            print("\n⏱️  响应时间统计:")
            for result in self.results:
                print(f"   {result['test_case']}: {result['avg_response_time']:.2f}s")

            # 计算总体统计
            all_avg_times = [r["avg_response_time"] for r in self.results]
            overall_avg = statistics.mean(all_avg_times)
            print(f"\n📈 总体平均响应时间: {overall_avg:.2f}s")

        print("\n💡 性能优化建议:")
        if self.results:
            slowest = max(self.results, key=lambda x: x["avg_response_time"])
            if slowest["avg_response_time"] > 2.0:
                print("   - 考虑优化 LLM 调用次数")
                print("   - 检查网络延迟")
                print("   - 考虑使用缓存机制")

        print("   - 监控内存使用情况")
        print("   - 考虑异步处理优化")


async def run_performance_tests():
    """运行完整的性能测试套件"""

    # 初始化 LLM
    import os

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 需要设置 LLM_API_KEY 或 OPENAI_API_KEY")
        return

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.1,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # 创建测试套件
    test_suite = PerformanceTestSuite(llm)

    # 测试用例
    test_cases = [
        {"name": "简单QA", "message": "什么是AI？", "expected_intent": "qa"},
        {"name": "复杂QA", "message": "上周的销售数据显示了什么趋势？", "expected_intent": "qa"},
        {"name": "写作任务", "message": "请帮我写一份报告", "expected_intent": "write"},
        {"name": "搜索任务", "message": "搜索最新论文", "expected_intent": "search"},
        {"name": "执行任务", "message": "执行备份", "expected_intent": "exec"},
        {"name": "闲聊", "message": "你好", "expected_intent": "smalltalk"},
    ]

    # 运行测试
    await test_suite.test_response_time(test_cases, iterations=3)
    accuracy = await test_suite.test_accuracy(test_cases)
    concurrent_result = await test_suite.test_concurrent_requests("什么是RAG系统？", concurrent_count=5)

    # 生成报告
    test_suite.generate_performance_report()

    return {"response_time_results": test_suite.results, "accuracy": accuracy, "concurrent_result": concurrent_result}


if __name__ == "__main__":
    """直接运行性能测试"""
    asyncio.run(run_performance_tests())
