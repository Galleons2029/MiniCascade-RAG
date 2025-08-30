# -*- coding: utf-8 -*-
# @Time   : 2025/8/14 09:46
# @Author : Galleons
# @File   : duckduckgo_search.py

"""DuckDuckGo search tool for LangGraph.

This module provides a DuckDuckGo search tool that can be used with LangGraph
to perform web searches. It returns up to 10 search results and handles errors
gracefully.
"""

from langchain_community.tools import DuckDuckGoSearchResults

# Expose a ready-to-use DuckDuckGo search tool for LangGraph
duckduckgo_search_tool = DuckDuckGoSearchResults(
    num_results=10,
    handle_tool_error=True,
)


if __name__ == "__main__":
    # 创建一个详细搜索实例
    search = DuckDuckGoSearchResults()

    # 执行搜索
    detailed_result = search.invoke("Obama")
    print(detailed_result)
    # from duckduckgo_search import search
    # from duckduckgo_search import duckduckgo_search
    # result = search("What is Machine Learning?")
