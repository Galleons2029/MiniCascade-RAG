# -*- coding: utf-8 -*-
"""
批量 RAG 测评脚本
- 从 .env 读取配置
- 直接用 ragas 默认 OpenAI LLM（支持代理 API_BASE）
- Langfuse trace
"""

import os
import json
import time
import pandas as pd
from langfuse import Langfuse
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

#配置
load_dotenv()
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

EVAL_JSONL_FILE = Path(os.getenv("EVAL_JSONL_FILE"))
RAG_API_URL = os.getenv("RAG_API_URL")


def load_crag_jsonl(jsonl_path: Path):
    data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            data.append({
                "interaction_id": item.get("interaction_id", ""),
                "question": item.get("query", ""),
                "ground_truth": item.get("answer", "")
            })
    return data

def ensure_list_of_str(x):
    if x is None: return []
    if isinstance(x, str): return [x]
    if isinstance(x, list):
        out = []
        for c in x:
            if isinstance(c, str): out.append(c)
            elif isinstance(c, dict): out.append(c.get("content") or c.get("text") or str(c))
            else: out.append(str(c))
        return out
    return [str(x)]

#调用rag API
def call_rag_api(question: str, retries: int = 3, delay: float = 0.5):
    import requests
    payload = {"query": question}
    for attempt in range(1, retries + 1):
        try:
            start_time = time.time()
            resp = requests.post(RAG_API_URL, json=payload, timeout=600)
            latency = time.time() - start_time
            resp.raise_for_status()
            res_json = resp.json()
            answer = res_json.get("answer", "")
            contexts_raw = res_json.get("contexts")
            contexts = ensure_list_of_str(contexts_raw)
            return answer, contexts, latency
        except Exception as e:
            print(f"[Warn] RAG API 请求失败({attempt}/{retries}): {e}")
            time.sleep(delay)
    return "", [], None


def main():
    questions = load_crag_jsonl(EVAL_JSONL_FILE)
    if not questions:
        print("未读取到任何条目")
        return

    #选择测评数据
    questions = [questions[9], questions[10]]
    data_records, trace_records = [], []

    for item in tqdm(questions, desc="评测中"):
        q_text, gt, interaction_id = item["question"], item.get("ground_truth",""), item.get("interaction_id","")
        answer, contexts, latency = call_rag_api(q_text)

        print("Question:", q_text)
        print("Answer:", answer)
        print("Contexts count:", len(contexts))
        for i, c in enumerate(contexts):
            print(f"\n--- Context {i + 1} ---")
            print(c[:1000])
            print(f"Length: {len(c)} chars")
        print("\nGround truth:", gt)
        print("Latency:", latency)

        try:
            span = langfuse.start_span(name="RAG Evaluation")
            span.input = q_text
            span.output = answer
            span.metadata = {"contexts": contexts, "ground_truth": gt, "interaction_id": interaction_id}
            span.end()
            langfuse.flush()

            record = {
                "interaction_id": interaction_id,"question": q_text,"answer": answer,
                "contexts": contexts,"ground_truth": gt,"ground_truths":[gt] if gt else [],"latency": latency
            }
            data_records.append(record)
            trace_records.append((span, record))
        except Exception as e:
            print(f"[Warn] Langfuse trace 失败：{e}")
            data_records.append({
                "interaction_id": interaction_id,"question": q_text,"answer": answer,
                "contexts": contexts,"ground_truth": gt,"ground_truths":[gt] if gt else [],"latency": latency
            })

    dataset = Dataset.from_list(data_records)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    results = evaluate(dataset=dataset, metrics=metrics)
    results_df = results.to_pandas()

    #评测指标
    metric_names = ["faithfulness","answer_relevancy","context_precision","context_recall"]
    for span, record in trace_records:
        try:
            qid = record["interaction_id"]
            row = results_df[results_df.get("interaction_id", pd.Series([None]*len(results_df)))==qid]
            if row.empty:
                row = results_df[results_df.get("question", pd.Series([None]*len(results_df)))==record["question"]]
            if not row.empty:
                row = row.iloc[0]
                for m in metric_names:
                    val = row.get(m)
                    if val is not None and not pd.isna(val):
                        span.score(name=m, value=float(val))
            if record["latency"] is not None:
                span.score(name="latency", value=float(record["latency"]))
        except Exception as e:
            print(f"[Warn] Langfuse 写分失败：{e}")
        finally:
            span.end()

    print("\n=== RAG 评估结果（整体平均值） ===")
    cols = [c for c in metric_names if c in results_df.columns]
    if "latency" in results_df.columns: cols.append("latency")
    print(results_df[cols].mean(numeric_only=True))

    out_path = Path("rag_evaluation_results.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n评估结果已保存到 {out_path.resolve()}")

if __name__ == "__main__":
    main()


