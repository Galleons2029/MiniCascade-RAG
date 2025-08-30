# Intent Agent 测试指南

本文档详细介绍如何测试和监控 MiniCascade-RAG 项目中的 Intent Agent 组件。

## 🎯 测试目标

1. **功能正确性**: 验证意图识别的准确性
2. **性能表现**: 测试响应时间和并发处理能力
3. **流程追踪**: 通过 LangSmith 监控整个处理流程
4. **错误处理**: 验证异常情况下的系统行为

## 🛠️ 环境准备

### 1. 基础环境变量

```bash
# 必需配置
LLM_API_KEY=your_api_key_here
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BASE_URL=https://api.siliconflow.cn/v1  # 可选，用于自定义 API 端点

# 项目配置
PROJECT_NAME=MiniCascade-RAG
ENVIRONMENT=development
```

### 2. LangSmith 配置（可选，用于高级监控）

```bash
# LangSmith 追踪
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=MiniCascade-RAG-Intent-Testing
```

获取 LangSmith API Key: https://smith.langchain.com/

### 3. Langfuse 配置（可选，用于 LLM 监控）

```bash
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 🧪 测试类型

### 1. 基础单元测试

测试 Intent Agent 的核心功能：

```bash
# 运行基础测试
python scripts/test_intent_agent.py --test-type basic

# 或使用 pytest 直接运行
pytest test/agent/test_intent_agent.py -v
```

**测试内容**:
- 意图分类准确性
- 实体提取功能
- 上下文继承机制
- 错误处理能力

### 2. LangSmith 集成测试

通过 LangSmith 进行详细的流程追踪：

```bash
# 运行 LangSmith 测试
python scripts/test_intent_agent.py --test-type langsmith
```

**功能特点**:
- 完整的调用链追踪
- LLM 输入输出记录
- 性能指标监控
- 错误堆栈追踪

**查看结果**: 测试完成后，访问 [LangSmith Dashboard](https://smith.langchain.com/) 查看详细追踪信息。

### 3. 性能测试

测试系统的性能表现：

```bash
# 运行性能测试
python scripts/test_intent_agent.py --test-type performance
```

**测试指标**:
- 平均响应时间
- 并发处理能力
- 吞吐量统计
- 资源使用情况

### 4. 交互式测试

实时测试和调试：

```bash
# 启动交互式测试
python scripts/test_intent_agent.py --test-type interactive
```

**使用方式**:
1. 输入测试消息
2. 查看实时分析结果
3. 验证多轮对话上下文
4. 输入 `quit` 退出

## 📊 测试用例

### 意图分类测试用例

| 意图类型 | 测试消息 | 预期结果 |
|---------|---------|---------|
| qa | "什么是RAG系统？" | intent: qa |
| qa | "上周的销售数据显示了什么趋势？" | intent: qa |
| write | "请帮我写一份关于AI发展的报告" | intent: write |
| search | "搜索最新的机器学习论文" | intent: search |
| exec | "执行数据备份任务" | intent: exec |
| smalltalk | "你好，今天天气怎么样？" | intent: smalltalk |

### 多轮对话测试

```
用户: "上周的账单情况如何？"
系统: [分析] intent: qa, entities: {subject: "账单"}, time_text: "上周"

用户: "上个月的呢？"
系统: [分析] intent: qa, 继承 subject: "账单", 更新 time_text: "上个月"
```

## 🔍 监控和调试

### 1. LangSmith 监控

LangSmith 提供了强大的监控能力：

- **调用链追踪**: 查看每个 LLM 调用的详细信息
- **性能分析**: 响应时间、Token 使用量统计
- **错误追踪**: 详细的错误堆栈和上下文
- **A/B 测试**: 比较不同版本的性能

### 2. 日志分析

系统使用结构化日志记录关键信息：

```python
# 意图检测日志
logger.info("intent_detected", intent=detected_intent, confidence=confidence)

# 实体提取日志
logger.info("entities_extracted", entities=entities, time_text=time_text)

# 上下文解析日志
logger.info("context_resolved", context_frame=new_frame)
```

### 3. 性能指标

关键性能指标：

- **响应时间**: < 2秒（目标）
- **准确率**: > 90%（目标）
- **并发处理**: 支持 10+ 并发请求
- **错误率**: < 5%

## 🚀 快速开始

1. **环境配置**:
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置必要的 API Key
   ```

2. **安装依赖**:
   ```bash
   uv sync
   ```

3. **运行基础测试**:
   ```bash
   python scripts/test_intent_agent.py --test-type basic
   ```

4. **启用 LangSmith 监控**:
   ```bash
   # 设置 LANGCHAIN_API_KEY
   python scripts/test_intent_agent.py --test-type langsmith
   ```

5. **交互式测试**:
   ```bash
   python scripts/test_intent_agent.py --test-type interactive
   ```

## 🔧 故障排除

### 常见问题

1. **API Key 错误**:
   - 检查 `.env` 文件中的 API Key 配置
   - 确认 API Key 有效且有足够的配额

2. **网络连接问题**:
   - 检查网络连接
   - 验证 API 端点是否可访问

3. **依赖缺失**:
   ```bash
   uv sync  # 重新同步依赖
   ```

4. **LangSmith 连接失败**:
   - 验证 LANGCHAIN_API_KEY 设置
   - 检查网络是否能访问 smith.langchain.com

### 调试技巧

1. **启用详细日志**:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **使用交互式测试**:
   - 实时查看处理结果
   - 逐步验证每个组件

3. **查看 LangSmith 追踪**:
   - 详细的调用链信息
   - LLM 输入输出记录

## 📈 持续改进

1. **定期运行测试**: 建议每次代码变更后运行完整测试套件
2. **监控性能指标**: 关注响应时间和准确率趋势
3. **收集用户反馈**: 基于实际使用情况优化意图分类
4. **更新测试用例**: 根据新的业务场景添加测试用例

## 🤝 贡献指南

如果您想改进测试套件：

1. 添加新的测试用例到相应的测试文件
2. 更新文档说明新的测试功能
3. 确保所有测试都能通过
4. 提交 Pull Request

---

有问题或建议？请创建 Issue 或联系开发团队。
