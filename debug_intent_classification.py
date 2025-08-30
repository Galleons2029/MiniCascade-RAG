#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试意图分类功能

这个脚本用于单独测试和调试意图分类功能，帮助理解为什么所有测试都返回 'other'。
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


async def test_intent_classification():
    """测试意图分类功能"""
    
    # 初始化 LLM
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 需要设置 LLM_API_KEY 或 OPENAI_API_KEY")
        return
    
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.1,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL")
    )
    
    print("🤖 测试意图分类功能")
    print("=" * 50)
    
    # 测试用例
    test_messages = [
        "什么是RAG系统？",
        "上周的销售数据显示了什么趋势？", 
        "请帮我写一份关于AI发展的报告",
        "搜索最新的机器学习论文",
        "执行数据备份任务",
        "你好，今天天气怎么样？"
    ]
    
    expected_intents = ["qa", "qa", "write", "search", "exec", "smalltalk"]
    
    for i, message in enumerate(test_messages):
        print(f"\n🧪 测试 {i+1}: {message}")
        print(f"   预期意图: {expected_intents[i]}")
        
        # 构建提示词
        system = (
            "You are an intent classifier. Classify the user's latest message into one of: "
            "qa, write, search, exec, smalltalk, other. "
            "Return a JSON object with keys: intent, confidence (0-1)."
        )
        user = f"Message: {message}\nRespond with JSON only."
        
        try:
            # 调用 LLM
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            
            # 获取响应内容
            content = getattr(resp, "content", "") or ""
            print(f"   LLM 原始响应: {content}")
            
            # 解析 JSON
            try:
                data = json.loads(content) if isinstance(content, str) else {}
                detected_intent = str(data.get("intent", "other"))
                confidence = float(data.get("confidence", 0.5))
                
                print(f"   解析结果: intent={detected_intent}, confidence={confidence}")
                
                # 验证结果
                if detected_intent == expected_intents[i]:
                    print("   ✅ 分类正确")
                else:
                    print(f"   ❌ 分类错误 (预期: {expected_intents[i]}, 实际: {detected_intent})")
                    
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON 解析失败: {e}")
                
                # 尝试备用解析方法
                lc = content.lower() if isinstance(content, str) else ""
                detected_intent = "other"
                for k in ["qa", "write", "search", "exec", "smalltalk"]:
                    if k in lc:
                        detected_intent = k
                        break
                
                print(f"   备用解析结果: {detected_intent}")
                
        except Exception as e:
            print(f"   ❌ LLM 调用失败: {e}")


async def test_improved_prompts():
    """测试改进的提示词"""
    
    # 初始化 LLM
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 需要设置 LLM_API_KEY 或 OPENAI_API_KEY")
        return
    
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.1,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL")
    )
    
    print("\n" + "=" * 50)
    print("🚀 测试改进的提示词")
    print("=" * 50)
    
    # 改进的提示词
    improved_system = """你是一个意图分类器。请将用户的消息分类到以下类别之一：

1. qa - 问答类：用户询问问题，需要获取信息或解释
2. write - 写作类：用户要求写作、总结、创作内容
3. search - 搜索类：用户要求搜索、查找特定信息
4. exec - 执行类：用户要求执行特定任务或操作
5. smalltalk - 闲聊类：日常对话、问候、闲聊
6. other - 其他：不属于以上任何类别

请返回JSON格式：{"intent": "分类结果", "confidence": 置信度(0-1之间的数字)}

示例：
- "什么是人工智能？" → {"intent": "qa", "confidence": 0.9}
- "帮我写一份报告" → {"intent": "write", "confidence": 0.9}
- "搜索最新新闻" → {"intent": "search", "confidence": 0.9}"""
    
    test_messages = [
        "什么是RAG系统？",
        "请帮我写一份关于AI发展的报告", 
        "搜索最新的机器学习论文",
        "你好"
    ]
    
    for message in test_messages:
        print(f"\n🧪 测试消息: {message}")
        
        try:
            resp = await llm.ainvoke([
                {"role": "system", "content": improved_system},
                {"role": "user", "content": message},
            ])
            
            content = getattr(resp, "content", "") or ""
            print(f"   LLM 响应: {content}")
            
            # 解析结果
            try:
                data = json.loads(content)
                print(f"   ✅ 解析成功: {data}")
            except:
                print(f"   ❌ JSON 解析失败")
                
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")


if __name__ == "__main__":
    async def main():
        await test_intent_classification()
        await test_improved_prompts()
    
    asyncio.run(main())
