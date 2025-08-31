# -*- coding: utf-8 -*-
# @Time    : 2025/8/25 02:39 
# @Author  : zqh
# @File    : sql_graph_verl.py
 # 配置大模型
import json
import os

from langchain_openai import ChatOpenAI
import re
from typing import Any, Literal, TypedDict, Annotated

from langgraph.graph.state import CompiledStateGraph

from app.configs import postgres_config
from app.core.agent.sql_agent.adapters.sql_prompt import DECOMPOSE_QUESTION_PROMPT
from app.core.agent.sql_agent.adapters.sql_prompt import WRITE_QUERY_PROMPT
from app.core.agent.sql_agent.adapters.sql_prompt import MERGE_RESULTS_PROMPT
from app.core.agent.sql_agent.adapters.sql_prompt import CHECK_QUERY_PROMPT
from app.core.agent.sql_agent.adapters.sql_prompt import REWRITE_QUERY_PROMPT
from app.core.agent.sql_agent.exec_eval_match import eval_exec_match
from app.core.config import settings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.utilities import SQLDatabase

from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
API_KEY = "sk-wdsylcaafxprwpvyrlmpvhsrpjzgrdnftpstmpgzeknwzpsq"
BASE_URL = settings.Silicon_base_url

sql_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0
)
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]
# PostgreSQL 连接配置
pg_host = postgres_config.PG_HOST
pg_port = postgres_config.PG_PORT
pg_user = postgres_config.PG_USER
pg_password = postgres_config.PG_PASSWORD
pg_db = postgres_config.PG_DB
db_uri = f"postgresql+psycopg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

# 在测试文件开头设置
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pr-clear-fly-70"
class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[BaseMessage]
    user_input_table: str



class SQLAgent:
    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        db_schema: str="public" ,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)
        self.db_schema = db_schema
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
    def parse_query(self,message: BaseMessage) -> str | None:
        result = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
            result = match.group(1).strip()
        return result


    # def get_table_info(self) -> str:
    #     """Get the table information in a human-readable format."""
    #     table_info = self.db.get_table_info()
    #     return table_info
    import re


    def get_table_info(self) -> str:
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

    def write_query(self,state: State):
        """
        1. 拆解问题 -> 子问题列表
        2. 对每个子问题生成 SQL 并执行
        3. 用 LLM 合并结果，生成最终 SQL
        """

        # ---------- 1. 拆解问题 ----------
        decompose_prompt = DECOMPOSE_QUESTION_PROMPT.invoke({"input": state["question"]})
        decompose_res = self.invoke_prompt(decompose_prompt)
        sub_questions = [
            q.strip()
            for q in decompose_res.content.splitlines()
            if q.strip() and q.strip()[0].isdigit()
        ]
        # if not sub_questions:               # 兜底：拆不出时退化成原问题
        #     sub_questions = [state["question"]]
        # ---------- 2. 子问题逐个生成 & 执行 ----------
        sub_details = []
        execute_tool = QuerySQLDatabaseTool(db=self.db)
        for sq in sub_questions:
            sq_prompt = WRITE_QUERY_PROMPT.invoke({
                "dialect": "postgresql",
                "input": sq,
                "table_info": self.get_table_info(),
            })
            sq_llm_res = self.invoke_prompt(sq_prompt)
            sq_sql = self.parse_query(sq_llm_res) or sq_llm_res.content

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
        final_res = self.invoke_prompt(merge_prompt)
        final_sql = self.parse_query(final_res) or final_res.content

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


    def execute_query(self,state: State) -> State:
        """Execute SQL query."""

        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        execution_result = execute_query_tool.invoke(state["query"])
        if not isinstance(execution_result, str):
            # Convert to string if it's not already
            execution_result = str(execution_result)
        return {**state, "execution": execution_result}


    def truncate_execuion(self,execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > 2048:
            return execution[: 2048] + "\n... (truncated)"
        return execution


    def invoke_prompt(self,prompt: Any) -> BaseMessage:
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
    def check_query(self,state: State) -> State:
        """Check the SQL query for correctness."""

        prompt = CHECK_QUERY_PROMPT.invoke(
            {
                "dialect": "postgresql",
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "table_info": self.get_table_info(),
            }
        )

        result = self.invoke_prompt(prompt)


        res = {
            **state,
            "feedback": result.content,
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        return res

    def rewrite_query(self,state: State) -> State:
        """Rewrite SQL query if necessary."""

        prompt = REWRITE_QUERY_PROMPT.invoke(
            {
                "dialect": "postgresql",
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)

        rewritten_query = self.parse_query(result)

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
        }


    def should_continue(self,state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if state.get("num_turns", 0) >= self.max_turns:
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

    def graph(self) -> CompiledStateGraph:
        builder = StateGraph(State)
        builder.add_node(self.write_query)
        builder.add_node(self.execute_query)
        builder.add_node(self.check_query)
        builder.add_node(self.rewrite_query)

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,
        )
        builder.add_edge("rewrite_query", "execute_query")
        sql_graph=builder.compile()
        return sql_graph

    def run(self, question: str) -> dict:
        initial_state = {
            "question": question,
            "query": "",
            "execution": "",
            "answer": "",
            "feedback": "",
            "num_turns": 0,
            "messages": [],
            "user_input_table":"Activity,Addresses,Allergy_Type,Apartment_Bookings,Apartment_Buildings,Apartments,Assessment_Notes,Assets,Behavior_Incident,Detention,Faculty,Faculty_Participates_in,Fault_Log,Guests,Maintenance_Contracts,Part_Faults,Participates_in,Parts,Ref_Address_Types,Ref_Detention_Type,Ref_Incident_Type,Skills,Staff,Student,Student_Addresses,Students,Teachers,Third_Party_Companies,accelerator_compatible_browser,aircraft,airport,airport_aircraft,architect,author,battle,body_builder,book,bridge,browser,conference,death,domain,journal,keyword,match,mill,organization,people,pilot,publication,ship,station,status,web_client_accelerator"
        }
        final_state = self.graph().invoke(initial_state)  # 调用graph()方法获取编译后的图
        return {
            "question": final_state["question"],
            "query": final_state["query"],
            "execution": final_state["execution"]
        }
def evaluate_query(query: str, ground_truth: str, raise_on_error: bool = True) -> float:

    try:
        # Parameters following the default setting
        exec_score = eval_exec_match(
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            print(f"Error evaluating query: {e}")
            return 0.0


def main():
    # 创建SQLAgent实例时传入数据库连接URI
    agent = SQLAgent(db=db_uri)
    # 读取 test.json
    with open("../test.json", "r") as f:
        test_data = json.load(f)

    # 提取所有 question
    questions = [item["question"] for item in test_data]
    truly_truth = [item["query"] for item in test_data]
    question_test = questions[119]
    ground_truth = truly_truth[119]
    try:
        result = agent.run(question_test)
        print("\n" + "=" * 50)
        print(f"问题: {result['question']}")
        print("-" * 50)
        print(f"生成的SQL查询:\n{result['query']}")
        print("-" * 50)
        print(f"执行结果:\n{result['execution']}")
        print("=" * 50)
        # 评估查询准确性
        accuracy = evaluate_query(
            result['query'],
            ground_truth,
            raise_on_error=False
        )
        print(f"查询准确度: {accuracy:.2f}")
    except Exception as e:
        print(f"执行出错: {str(e)}")

if __name__ == "__main__":

    main()

