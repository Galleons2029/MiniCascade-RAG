# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 06:55 
# @Author  : zqh
# @File    : sql_prompt.py
# -*- coding: utf-8 -*-
# @Time    : 2025/8/25 06:35
# @Author  : zqh
# @File    : sql_prompt.py

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
You are a SQL query planner. Given a user question, break it down into **multiple simpler sub-questions** 
that can each be answered with a single SQL query.

Rules:
- Each sub-question should be simple, specific, and answerable with one SQL query.
- Do not include JOINs or aggregations across sub-questions — we will merge results later.
- Return a numbered list.

Now do the same for the following question.
"""),
    ("user", "Question: {input}")
])

MERGE_RESULTS_PROMPT = ChatPromptTemplate([
    ("system", """
You are given a list of SQL queries and their results. Merge them logically to answer the original user question.

You can use SQL JOINs, UNION, or Python-style logic — but you must return **one final SQL query** 
that answers the original question.
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
- **Zero-row Tolerance**: If the query returns **zero rows**, immediately flag it as INCORRECT, 
regardless of whether the question implies data should exist.  
- Verify that any non-empty result makes sense in the context of the question.  
- Check for obviously incorrect values (e.g., negative counts, impossible dates).

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the above lists are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases 
in all caps and without surrounding quotes:
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


Please rewrite the query.""",
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
