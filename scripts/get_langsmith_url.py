#!/usr/bin/env python3
"""
获取正确的 LangSmith 项目 URL

这个脚本会帮助你找到正确的 LangSmith 项目链接
"""

import os
import sys
from langsmith import Client

def get_langsmith_project_url():
    """获取 LangSmith 项目 URL"""
    
    # 检查环境变量
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ 请先设置 LANGCHAIN_API_KEY 环境变量")
        print("   获取 API Key: https://smith.langchain.com/")
        return None
    
    try:
        # 初始化客户端
        client = Client()
        
        # 项目名称
        project_name = "MiniCascade-RAG-Intent-Testing"
        
        print(f"🔍 查找项目: {project_name}")
        
        # 尝试获取项目信息
        try:
            projects = list(client.list_projects())
            
            print(f"📋 找到 {len(projects)} 个项目:")
            
            target_project = None
            for project in projects:
                project_display_name = getattr(project, 'name', 'Unknown')
                project_id = getattr(project, 'id', 'Unknown')
                
                print(f"   - {project_display_name} (ID: {project_id})")
                
                if project_display_name == project_name:
                    target_project = project
            
            if target_project:
                project_id = getattr(target_project, 'id', None)
                if project_id:
                    url = f"https://smith.langchain.com/o/default/projects/p/{project_id}"
                    print(f"\n✅ 找到目标项目!")
                    print(f"🔗 项目 URL: {url}")
                    return url
            else:
                print(f"\n⚠️  未找到项目 '{project_name}'")
                print("💡 可能的原因:")
                print("   1. 项目尚未创建（运行一次测试后会自动创建）")
                print("   2. 项目名称不匹配")
                
        except Exception as e:
            print(f"❌ 获取项目列表失败: {e}")
            
        # 提供通用链接
        print(f"\n🌐 通用链接:")
        print(f"   LangSmith Dashboard: https://smith.langchain.com/")
        print(f"   在 Dashboard 中搜索项目: {project_name}")
        
        return "https://smith.langchain.com/"
        
    except Exception as e:
        print(f"❌ LangSmith 客户端初始化失败: {e}")
        return None

if __name__ == "__main__":
    print("🔧 LangSmith URL 获取工具")
    print("=" * 50)
    
    url = get_langsmith_project_url()
    
    if url:
        print(f"\n📋 使用说明:")
        print(f"1. 复制上面的 URL")
        print(f"2. 在浏览器中打开")
        print(f"3. 查看 Intent Agent 的调用追踪")
    else:
        print(f"\n❌ 无法获取项目 URL")
        print(f"请检查 LANGCHAIN_API_KEY 设置")
