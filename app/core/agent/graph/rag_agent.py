# -*- coding: utf-8 -*-
# @Time   : 2025/8/26 21:51
# @Author : Galleons
# @File   : rag_agent.py

"""
这里是文件说明
"""

from typing import Annotated, Sequence, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from app.core.rag.retriever import VectorRetriever
from app.configs import llm_config, app_config
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
  secret_key="sk-lf-cbd943ab-4879-44e0-8c1f-5495cbfc0a47",
  public_key="pk-lf-49f12594-88c3-4991-be28-1e4eb4570a0d",
  host="https://cloud.langfuse.com"
)

langfuse_handler = CallbackHandler()

QDRANT_COLLECTION = "zsk_test1"


client=QdrantClient(
    host='localhost',
    port=6333,
)
embedding_model = OpenAIEmbeddings(api_key=llm_config.SILICON_KEY,
                                    base_url=llm_config.SILICON_BASE_URL,
                                   model="BAAI/bge-m3")


# @tool("search_tool")
# def search_tool(query: str):
#     """Qdrant search tool"""
#     retriever = VectorRetriever(query)
#
#     search = retriever
#     return search.run(query)

@tool("get relevant chunk")
def retrieve_content(query: str):
    """Retrieve information related to a query."""
    retriever = VectorRetriever(query)
    retrieved_docs = retriever.retrieve_top_k(k=4,collections=['zsk_test1'],)
    context = retriever.rerank(hits=retrieved_docs, keep_top_k=3)

    # serialized = "\n\n".join(
    #     (f"Source: {doc.metadata}\nContent: {doc.page_content}")
    #     for doc in retrieved_docs
    # )
    return context

tools = [retrieve_content]



class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]



### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str | int = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    # model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    model = ChatOpenAI(model=llm_config.FREE_LLM_MODEL,
               api_key=llm_config.SILICON_KEY,
               base_url="https://api.siliconflow.cn/v1",
               temperature=0,streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) then grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. /no_think""",
        input_variables=["context", "question"],
    )

    # Chain
    # chain = prompt | llm_with_tool
    chain = prompt | model

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    print("question: ", question)
    print("context: ", docs)
    scored_result = chain.invoke({"question": question, "context": docs})

    #score = scored_result.binary_score

    if "yes" in scored_result.content:
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"


### Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = ChatOpenAI(model='deepseek-ai/DeepSeek-V3',
               api_key=llm_config.SILICON_KEY,
               base_url="https://api.siliconflow.cn/v1",
               temperature=0)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    # model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    model = ChatOpenAI(model='deepseek-ai/DeepSeek-V3',
               api_key=llm_config.SILICON_KEY,
               base_url="https://api.siliconflow.cn/v1",
               temperature=0, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \n
        Question: {question} \n
        Context: {context} \n
        Answer: """,
        input_variables=["context", "question"],
    )

    # LLM
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    llm = ChatOpenAI(model='deepseek-ai/DeepSeek-V3',
               api_key=llm_config.SILICON_KEY,
               base_url="https://api.siliconflow.cn/v1",
               temperature=0, streaming=True)
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    print("context: ", docs)
    print("question: ", question)
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}



# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retrieve_content])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
rag_agent = workflow.compile()

if __name__ == "__main__":
    state = {
        "messages": []
    }

    model = ChatOpenAI(model=llm_config.FREE_LLM_MODEL,
               api_key=llm_config.SILICON_KEY,
               base_url="https://api.siliconflow.cn/v1",
               temperature=0,streaming=True)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) then grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. /no_think""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | model

    question = """
    什么是数据科学？
    """
    docs = """
    《数据科学》考试大纲 # 一、考试要求 1. 要求考生掌握统计学的基本原理,掌握数据收集和处理的基本分析方法,具备运用统计方法分析数据和解释数据的基本能力。 2. 要求考生掌握数据结构的基本概念、基本原理和基本方法。掌握数据的逻辑结构、存储结构及基本操作的实现,能够对算法进行基本的时间与空间复杂度的分析;能够运用数据结构基本原理和方法进行问题的分析与求解,具备一定算法实现的能力。 3. 要求考生掌握常用数值计算方法的基本原理,掌握求解线性和非线性代数方程组、插值与拟合、积分方程、微分方程等问题的基本方法。 # 二、考试内容 # 2.1 统计学 (1)导论:统计学的应用领域;数据的分类;统计学中的基本概念,如总体、个体、样本、变量等。 (2)数据的搜集:常见的调查方法,如概率抽样、非概率抽样;统计误差的主要来源;统计数据的质量要求。 (3)数据的图表展示:常用统计图,如条形图、帕累托图、饼图、环形图、直方图、箱型图、散点图等。 (4)数据的概括性度量:众数、中位数、平均数、四分位数、离散系数等的概念;不同类型数据的概括性度量。"
    """
    for i in range(10):
        response = chain.invoke({"question": question, "context": docs}, config={"callbacks": [langfuse_handler]})

    print(type(response.content))
    print(response.content)
    print(response)


    # # Data model
    # class grade(BaseModel):
    #     """Binary score for relevance check."""
    #
    #     binary_score: str | int = Field(description="Relevance score 'yes' or 'no'")
    #
    # # LLM with tool and validation
    # llm_with_tool = model.with_structured_output(grade)
    #
    # chain2 = prompt | llm_with_tool
    #
    # response2 = chain2.invoke({"question": question, "context": docs})
    #
    # print(response2)
    # #print()