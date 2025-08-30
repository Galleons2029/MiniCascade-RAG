#!/bin/bash

# 清理已被 git 跟踪的测试和调试文件
# 这个脚本会将这些文件从 git 跟踪中移除，但保留本地文件

echo "🧹 清理已被 git 跟踪的测试和调试文件..."

# 要移除跟踪的文件列表
files_to_untrack=(
    "debug_intent_classification.py"
    "test_intent_simple.py" 
    "test_intent_detection.py"
    "test_unified_agent.py"
)

# 检查文件是否存在并移除跟踪
for file in "${files_to_untrack[@]}"; do
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        echo "📝 移除 git 跟踪: $file"
        git rm --cached "$file"
    else
        echo "⚠️  文件未被跟踪: $file"
    fi
done

echo ""
echo "✅ 清理完成！"
echo ""
echo "📋 接下来的步骤："
echo "1. 检查 git status 确认更改"
echo "2. 提交更改: git commit -m 'Remove test and debug files from tracking'"
echo "3. 这些文件现在会被 .gitignore 忽略"
echo ""
echo "💡 提示: 这些文件仍然存在于你的本地目录中，只是不再被 git 跟踪"
