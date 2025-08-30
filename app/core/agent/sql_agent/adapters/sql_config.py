# -*- coding: utf-8 -*-
# @Time    : 2025/8/25 06:35 
# @Author  : zqh
# @File    : sql_config.py

from langchain_core.prompts import ChatPromptTemplate


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)

DECOMPOSE_QUESTION_PROMPT = ChatPromptTemplate([
    ("system", """
You are a SQL query planner. Given a user question, break it down into **multiple simpler sub-questions** that can each be answered with a single SQL query.

Rules:
- Each sub-question should be simple, specific, and answerable with one SQL query.
- Do not include JOINs or aggregations across sub-questions — we will merge results later.
- Return a numbered list.

Examples:

User: "找出发布Rock音乐类型最多的艺术家"
Sub-questions:
1. 找出所有Rock音乐类型的track_id
2. 找出这些track_id对应的AlbumId
3. 找出这些AlbumId对应的ArtistId
4. 找出这些ArtistId对应的艺术家
5. 统计每个艺术家的购买次数并排序

User: "找出销售额最高的5位艺术家"
Sub-questions:
1. 从InvoiceLine中计算每行的销售额(Quantity * UnitPrice)
2. 找出这些销售额对应的TrackId
3. 找出这些TrackId对应的AlbumId
4. 找出这些AlbumId对应的ArtistId
5. 按ArtistId汇总销售额并取前5名

User: "列出所有从未被购买过的专辑"
Sub-questions:
1. 从InvoiceLine中找出所有已被购买的TrackId
2. 找出这些TrackId对应的AlbumId
3. 从Album表中排除这些AlbumId并列出剩余专辑

User: "找出每位员工所支持客户带来的总销售额"
Sub-questions:
1. 从Customer表中找出每个客户的SupportRepId
2. 找出这些客户对应的InvoiceId
3. 从InvoiceLine中计算每个InvoiceId的销售额
4. 按SupportRepId汇总销售额

User: "统计每种MediaType的平均曲目时长"
Sub-questions:
1. 从Track表中找出每个TrackId对应的MediaTypeId和Milliseconds
2. 按MediaTypeId计算Milliseconds的平均值

User: "找出Rock以外拥有最多曲目的流派"
Sub-questions:
1. 从Genre表中找出Rock对应的GenreId
2. 从Track表中排除Rock的GenreId
3. 按剩余GenreId计数曲目数量
4. 找出曲目数量最多的GenreId
5. 找出该GenreId对应的流派名称

User: "列出至少包含10首歌曲的播放列表"
Sub-questions:
1. 从PlaylistTrack中按PlaylistId计数TrackId
2. 筛选计数≥10的PlaylistId
3. 找出这些PlaylistId对应的播放列表名称

User: "找出从未在任何播放列表里出现过的歌曲"
Sub-questions:
1. 从PlaylistTrack中找出所有已出现的TrackId
2. 从Track表中排除这些TrackId并列出剩余歌曲

User: "找出每个国家客户的数量与总消费额"
Sub-questions:
1. 从Customer表中找出每个客户的Country
2. 找出这些客户对应的InvoiceId和Total金额
3. 按Country汇总客户数量和总消费额

User: "找出由同一位作曲家创作但被不同艺术家演唱的歌曲"
Sub-questions:
1. 从Track表中找出Composer不为空的TrackId和Composer
2. 找出这些TrackId对应的AlbumId
3. 找出这些AlbumId对应的ArtistId
4. 找出同一Composer对应多个ArtistId的组合

Now do the same for the following question.
"""),
    ("user", "Question: {input}")
])

MERGE_RESULTS_PROMPT = ChatPromptTemplate([
    ("system", """
You are given a list of SQL queries and their results. Merge them logically to answer the original user question.

You can use SQL JOINs, UNION, or Python-style logic — but you must return **one final SQL query** that answers the original question.
"""),
    ("user", """
Original question: {question}

Sub-queries and results:
{sub_details}

Please return the final SQL query.
""")
])
CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasonable query execution results

## Additional Critical Checks ##
1. **Result Validity Check**:
   - If the question expects non-empty results (e.g., "find most", "top N", "list all"), 
     but the execution returns no data, this is an ERROR.
   - If the question expects specific data patterns (e.g., numerical results for aggregation, 
     specific date ranges) but results are missing or invalid, this is an ERROR.
   - If the result count is zero when the question implies existence of data, this is an ERROR.

2. **Revised “Result Reasonableness Check” **
- **Zero-row Tolerance**: If the query returns **zero rows**, immediately flag it as INCORRECT, regardless of whether the question implies data should exist.  
- Verify that any non-empty result makes sense in the context of the question.  
- Check for obviously incorrect values (e.g., negative counts, impossible dates).

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the above lists are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
Execution result:

复制代码
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)
def print_database_tables(db):
    """
    打印数据库中的所有表名
    参数:
        db: SQLDatabase 对象
    """
    table_names = db.get_usable_table_names()
    print("数据库中的表:")
    for table_name in table_names:
        print(f"- {table_name}")



