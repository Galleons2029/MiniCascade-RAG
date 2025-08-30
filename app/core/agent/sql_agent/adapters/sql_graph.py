# -*- coding: utf-8 -*-
# @Time    : 2025/8/28 15:39 
# @Author  : zqh
# @File    : sql_graph.py
import re
from typing import Any, Literal, List, TypedDict, Annotated

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import BaseMessage, HumanMessage
from app.configs import postgres_config
from app.core.agent.sql_agent.adapters.sql_config import DECOMPOSE_QUESTION_PROMPT, WRITE_QUERY_PROMPT, \
    MERGE_RESULTS_PROMPT, CHECK_QUERY_PROMPT, REWRITE_QUERY_PROMPT
from app.core.agent.sql_agent.adapters.sql_graph_verl import evaluate_query
from app.core.config import settings

pg_host = postgres_config.PG_HOST
pg_port = postgres_config.PG_PORT
pg_user = postgres_config.PG_USER
pg_password = postgres_config.PG_PASSWORD
pg_db = postgres_config.PG_DB
db_uri = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
db=SQLDatabase.from_uri(db_uri)
API_KEY = "sk-wdsylcaafxprwpvyrlmpvhsrpjzgrdnftpstmpgzeknwzpsq"
BASE_URL = settings.Silicon_base_url

sql_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0
)
class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[BaseMessage]
    table_info: str

def parse_query(message: BaseMessage) -> str | None:
    result = None
    for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
        result = match.group(1).strip()
    return result


def get_table_info(db) -> str:
    """Get the table information in a human-readable format."""
    table_info = db.get_table_info()
    return table_info


def write_query(state: State):
    """
    1. 拆解问题 -> 子问题列表
    2. 对每个子问题生成 SQL 并执行
    3. 用 LLM 合并结果，生成最终 SQL
    """

    # ---------- 1. 拆解问题 ----------
    decompose_prompt = DECOMPOSE_QUESTION_PROMPT.invoke({"input": state["question"]})
    decompose_res = invoke_prompt(decompose_prompt)
    sub_questions: List[str] = [q.strip() for q in decompose_res.content.splitlines() if q.strip() and q.strip()[0].isdigit()]
    # if not sub_questions:               # 兜底：拆不出时退化成原问题
    #     sub_questions = [state["question"]]
    # ---------- 2. 子问题逐个生成 & 执行 ----------
    sub_details = []
    execute_tool = QuerySQLDatabaseTool(db=db)
    for sq in sub_questions:
        sq_prompt = WRITE_QUERY_PROMPT.invoke({
            "dialect": "postgresql",
            "input": sq,
            "table_info": get_table_info(db),
        })
        sq_llm_res = invoke_prompt(sq_prompt)
        sq_sql = parse_query(sq_llm_res) or sq_llm_res.content

        # 执行子查询
        try:
            sq_exec = execute_tool.invoke(sq_sql)
        except Exception as e:
            sq_exec = f"Error: {e}"

        sub_details.append(f"Sub-question: {sq}\nSQL:\n{sq_sql}\nResult:\n{sq_exec}")
    # ---------- 3. 生成最终合并 SQL ----------
    merge_prompt = MERGE_RESULTS_PROMPT.invoke({
        "question": state["question"],
        "sub_details": "\n\n".join(sub_details)
    })
    final_res = invoke_prompt(merge_prompt)
    final_sql = parse_query(final_res) or final_res.content

    # ---------- 4. 返回状态 ----------
    messages = [
        *decompose_prompt.messages, decompose_res,
        *merge_prompt.messages, final_res
    ]
    #

    return {
        **state,
        "query": final_sql,
        "num_turns": 1,
        "messages": messages,
    }


def execute_query(state: State) -> State:
    """Execute SQL query."""

    execute_query_tool = QuerySQLDatabaseTool(db=db)
    execution_result = execute_query_tool.invoke(state["query"])
    if not isinstance(execution_result, str):
        # Convert to string if it's not already
        execution_result = str(execution_result)
    return {**state, "execution": execution_result}


def truncate_execuion(execution: str) -> str:
    """Truncate the execution result to a reasonable length."""
    if len(execution) > 2048:
        return execution[: 2048] + "\n... (truncated)"
    return execution


def invoke_prompt(prompt: Any) -> BaseMessage:
    try:
        # # 使用结构化输出 - 确保模型返回JSON
        # structured_llm = sql_llm.with_structured_output(
        #     schema=QueryOutput,
        #     method="json_mode"
        # )
        result = sql_llm.invoke(prompt)
    except Exception as e:
        #logger.error(f"Failed to invoke prompt: {e}")
        print(f"Failed to invoke prompt: {e}")
        # FIXME: fallback to create a random trajectory
        result = sql_llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])

    return result
def check_query(state: State) -> State:
    """Check the SQL query for correctness."""

    prompt = CHECK_QUERY_PROMPT.invoke(
        {
            "dialect": "postgresql",
            "input": state["question"],
            "query": state["query"],
            "execution": truncate_execuion(state["execution"]),
            "table_info": get_table_info(db),
        }
    )

    result = invoke_prompt(prompt)


    res = {
        **state,
        "feedback": result.content,
        "messages": [*state.get("messages", []), *prompt.messages, result],
    }
    return res

def rewrite_query(state: State) -> State:
    """Rewrite SQL query if necessary."""

    prompt = REWRITE_QUERY_PROMPT.invoke(
        {
            "dialect": "postgresql",
            "input": state["question"],
            "query": state["query"],
            "execution": truncate_execuion(state["execution"]),
            "feedback": state["feedback"],
            "table_info": get_table_info(db),
        }
    )
    result = invoke_prompt(prompt)

    rewritten_query = parse_query(result)

    return {
        **state,
        "query": rewritten_query or state["query"],
        "num_turns": state.get("num_turns", 0) + 1,
        "messages": [*prompt.messages, result],  # clear previous prompts
    }


def should_continue(state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
    """Determine if the agent should continue based on the result."""
    if state.get("num_turns", 0) >= 5:
        return END

    if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
        last_message = state["messages"][-1]
        if "THE QUERY IS CORRECT" in last_message.content:
            if "THE QUERY IS INCORRECT" in last_message.content:
                # Both correct and incorrect messages found
                # See which is the last one
                correct_index = last_message.content.rfind("THE QUERY IS CORRECT")
                incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")
                if correct_index > incorrect_index:
                    return END
            else:
                if state["execution"] == "":
                    return "rewrite_query"
                return END
    return "rewrite_query"

builder = StateGraph(State)
builder.add_node(write_query)
builder.add_node(execute_query)
builder.add_node(check_query)
builder.add_node(rewrite_query)

builder.add_edge(START, "write_query")
builder.add_edge("write_query", "execute_query")
builder.add_edge("execute_query", "check_query")
builder.add_conditional_edges(
    "check_query",
    should_continue,
)
builder.add_edge("rewrite_query", "execute_query")
sql_graph_test=builder.compile()


if __name__ == "__main__":
    question = "客户的姓名、国家、客户购买过的艺术家数量。"
    try:
        result = sql_graph_test.invoke({"question": question})
        print("\n" + "=" * 50)
        print(f"问题: {result['question']}")
        print("-" * 50)
        print(f"生成的SQL查询:\n{result['query']}")
        print("-" * 50)
        print(f"执行结果:\n{result['execution']}")
        print("=" * 50)
        ground_truth = """SELECT artist.artistid, artist.name, COUNT(track.trackid) \
                          FROM artist \
                                   JOIN album ON artist.artistid = album.artistid \
                                   JOIN track ON album.albumid = track.albumid \
                                   JOIN genre ON track.genreid = genre.genreid \
                          WHERE genre.name = 'Rock' \
                          GROUP BY artist.artistid, artist.name \
                          ORDER BY COUNT(track.trackid) DESC LIMIT 1;"""
        db_path = "/opt/MiniCascade-RAG/app/core/agent/sql_agent/mydata.sqlite"
        # 评估查询准确性
        accuracy = evaluate_query(
            result['query'],
            ground_truth,
            db_path,
            raise_on_error=False
        )
        print(f"查询准确度: {accuracy:.2f}")
    except Exception as e:
        print(f"执行出错: {str(e)}")