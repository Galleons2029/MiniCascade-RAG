# .gitignore 配置说明

## 📋 忽略的文件类型

### 🔧 开发环境相关
- **IDE配置**: `.idea/`, `.vscode/`, `*.swp`
- **编辑器临时文件**: `*~`, `*.swo`

### 🐍 Python项目相关
- **字节码**: `__pycache__/`, `*.pyc`, `*.pyo`
- **构建文件**: `build/`, `dist/`, `*.egg-info/`
- **虚拟环境**: `venv/`, `.env`, `.venv`

### 🧪 测试文件相关 (选择性忽略)
**已忽略的测试文件**:
- `test_unified_agent.py` - 临时测试脚本
- `test_intent_simple.py` - 简单测试脚本  
- `test_intent_detection.py` - 意图检测测试
- `*_test_temp.py` - 临时测试文件

**保留的测试文件**:
- `test/` 目录下的正式测试文件
- `app/api/test_main.py` 等核心测试

### 📊 数据和模型文件
- **AI模型**: `*.pkl`, `*.pickle`, `models/`, `weights/`
- **向量数据库**: `qdrant_storage/`, `vector_store/`
- **数据库文件**: `*.db`, `*.sqlite3`

### 📈 可视化文件 (选择性忽略)
**已忽略**:
- `agent_graph.png` - 生成的图片
- `agent_graph_static.mermaid` - 生成的Mermaid文件
- `*.tmp.png`, `*.tmp.mermaid` - 临时图片

**保留**: 重要的设计图表和文档图片

### 📝 文档和报告 (选择性忽略)
**已忽略**:
- `AGENT_MERGE_SUMMARY.md` - 临时合并报告
- `agent_analysis_report.md` - 临时分析报告
- `*_temp.md`, `*_draft.md` - 临时文档

### 🗂️ 备份和临时文件
- **备份目录**: `backup/`
- **临时文件**: `tmp/`, `temp/`, `*.tmp`, `*.bak`
- **系统文件**: `.DS_Store`, `Thumbs.db`

### ⚙️ 配置文件
- **环境变量**: `.env` (保留 `.env.example`)
- **配置文件**: `config.yaml`, `secrets.json`

### 📔 Jupyter Notebook
- **检查点**: `.ipynb_checkpoints/`
- **保留**: 重要的 `.ipynb` 文件

## 🎯 当前状态

根据新的 `.gitignore` 规则，以下文件将被忽略而不会上传到Git：

### 即将被忽略的未跟踪文件:
- `test_unified_agent.py` ✅
- `test_intent_simple.py` ✅  
- `test_intent_detection.py` ✅
- `agent_graph_static.mermaid` ✅
- `AGENT_MERGE_SUMMARY.md` ✅
- `agent_analysis_report.md` ✅

### 保留在版本控制中的重要文件:
- `agent_graph_visualization.ipynb` ✅
- `visualize_agent_graph.py` ✅
- `app/core/agent/graph/intent_agent.py` ✅
- `test/` 目录下的正式测试 ✅

## 🔧 自定义调整

如果您需要调整某些规则，可以：

### 保留特定文件
在文件名前添加 `!` 来强制包含：
```gitignore
# 例如，要保留特定的测试文件
!test_important.py
```

### 忽略特定文件
添加完整的文件路径：
```gitignore
# 例如，忽略特定的配置文件
config/local_settings.py
```

### 临时忽略已跟踪的文件
```bash
# 临时忽略已跟踪的文件修改
git update-index --skip-worktree filename
```

## 📁 推荐的项目文件结构

```
MiniCascade-RAG/
├── app/                    # 主应用代码 ✅ 跟踪
├── test/                   # 正式测试套件 ✅ 跟踪  
├── docs/                   # 项目文档 ✅ 跟踪
├── README.md               # 项目说明 ✅ 跟踪
├── pyproject.toml          # 项目配置 ✅ 跟踪
├── .env.example            # 环境变量示例 ✅ 跟踪
├── .gitignore              # 忽略规则 ✅ 跟踪
│
├── .env                    # 实际环境变量 ❌ 忽略
├── backup/                 # 备份文件 ❌ 忽略
├── logs/                   # 日志文件 ❌ 忽略
├── test_*.py               # 临时测试 ❌ 忽略
└── *.tmp.*                 # 临时文件 ❌ 忽略
```

## 💡 使用建议

1. **定期检查**: 使用 `git status` 检查哪些文件被忽略
2. **团队协作**: 确保 `.gitignore` 规则对团队所有成员都适用
3. **敏感信息**: 确保所有包含密钥、密码的文件都被忽略
4. **临时文件**: 养成为临时文件添加特定后缀的习惯 (如 `.tmp`, `_temp`)

---
*更新时间: 2025-08-28*
