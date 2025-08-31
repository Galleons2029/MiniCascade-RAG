# -*- coding: utf-8 -*-
# @Time    : 2025/8/28 15:39 
# @Author  : zqh
# @File    : sql_graph.py
import re
from typing import Any, Literal, List

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import BaseMessage, HumanMessage
from app.configs import postgres_config
from app.core.agent.sql_agent.adapters.sql_config import DECOMPOSE_QUESTION_PROMPT, WRITE_QUERY_PROMPT, \
    MERGE_RESULTS_PROMPT, CHECK_QUERY_PROMPT, REWRITE_QUERY_PROMPT
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
num_turns = 1
sql_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0
)
class State(MessagesState):
    research_topic: str
    raw_notes: str
    compressed_research: str
    #feedback: str
    #num_turns: int
    researcher_messages: list[BaseMessage]


def parse_query(message: BaseMessage) -> str | None:
    result = None
    for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
        result = match.group(1).strip()
    return result


def get_table_info() -> str:
    """Parse the table information from a string and return it in a dictionary format."""
    import psycopg2
    # 连接到数据库
    conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        dbname=pg_db
    )

    # 创建游标
    cur = conn.cursor()

    # 查询数据库中所有表及其列信息
    cur.execute("""
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                """)

    # 获取查询结果
    rows = cur.fetchall()

    # 将表及其列信息保存到字典中
    tables_columns = {}
    for row in rows:
        table_name, column_name = row
        if table_name not in tables_columns:
            tables_columns[table_name] = []
        tables_columns[table_name].append(column_name)

    # 关闭游标和连接
    cur.close()
    conn.close()
    # 将字典转换为格式化的字符串
    formatted_str = ""
    for table, columns in tables_columns.items():
        formatted_str += f"Table: {table}\n"
        formatted_str += "Columns: " + ", ".join(columns) + "\n\n"

    return formatted_str


def write_query(state: State):
    """
    1. 拆解问题 -> 子问题列表
    2. 对每个子问题生成 SQL 并执行
    3. 用 LLM 合并结果，生成最终 SQL
    """

    # ---------- 1. 拆解问题 ----------
    decompose_prompt = DECOMPOSE_QUESTION_PROMPT.invoke({"input": state["research_topic"]})
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
            "table_info": get_table_info(),
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
        "question": state["research_topic"],
        "sub_details": "\n\n".join(sub_details)
    })
    final_res = invoke_prompt(merge_prompt)
    final_sql = parse_query(final_res) or final_res.content
    messages = [
        *decompose_prompt.messages, decompose_res,
        *merge_prompt.messages, final_res
    ]
    return {
        **state,
        "raw_notes": final_sql,
        "researcher_messages": messages,
    }


def execute_query(state: State) -> State:
    """Execute SQL query."""

    execute_query_tool = QuerySQLDatabaseTool(db=db)
    execution_result = execute_query_tool.invoke(state["raw_notes"])
    if not isinstance(execution_result, str):
        # Convert to string if it's not already
        execution_result = str(execution_result)
    return {**state, "compressed_research": execution_result}


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
            "input": state["research_topic"],
            "query": state["raw_notes"],
            "execution": truncate_execuion(state["compressed_research"]),
            "table_info": get_table_info(),
        }
    )

    result = invoke_prompt(prompt)
    res = {
        **state,
        "researcher_messages": [*state.get("researcher_messages", []), *prompt.messages, result],
    }
    return res

def rewrite_query(state: State) -> State:
    """Rewrite SQL query if necessary."""
    global num_turns
    num_turns = num_turns+1
    prompt = REWRITE_QUERY_PROMPT.invoke(
        {
            "dialect": "postgresql",
            "input": state["research_topic"],
            "query": state["raw_notes"],
            "execution": truncate_execuion(state["compressed_research"]),
            "table_info": get_table_info(),
        }
    )
    result = invoke_prompt(prompt)

    rewritten_query = parse_query(result)

    return {
        **state,
        "raw_notes": rewritten_query or state["raw_notes"],
        "compressed_research": [*prompt.messages, result],  # clear previous prompts
    }


def should_continue(state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
    """Determine if the agent should continue based on the result."""
    global num_turns
    if num_turns > 5:
        return END
    if state["researcher_messages"] and isinstance(state["researcher_messages"][-1], BaseMessage):
        last_message = state["researcher_messages"][-1]
        if "THE QUERY IS CORRECT" in last_message.content:
            if "THE QUERY IS INCORRECT" in last_message.content:
                # Both correct and incorrect messages found
                # See which is the last one
                correct_index = last_message.content.rfind("THE QUERY IS CORRECT")
                incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")
                if correct_index > incorrect_index:
                    return END
            else:
                if state["compressed_research"] == "":
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
    question = "Find the first names of the faculty members who are playing Canoeing or Kayaking."
    try:
        result = sql_graph_test.invoke({"research_topic": question})
        print("\n" + "=" * 50)
        print(f"问题: {result['research_topic']}")
        print("-" * 50)
        print(f"生成的SQL查询:\n{result['raw_notes']}")
        print("-" * 50)
        print(f"执行结果:\n{result['compressed_research']}")
        print("=" * 50)
    except Exception as e:
        print(f"执行出错: {str(e)}")
