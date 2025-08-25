# -*- coding: utf-8 -*-
"""
批量 RAG 测评脚本
- 从 .env 读取配置
- 使用 SiliconFlow 替换 Ragas 默认 OpenAI
- Langfuse trace + span + 打分
"""

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ==================== dotenv ====================
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件

# ==================== Langfuse ====================
from langfuse import Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# ==================== 配置 ====================
EVAL_JSONL_FILE = Path(os.getenv("EVAL_JSONL_FILE"))
RAG_API_URL = os.getenv("RAG_API_URL")
SILICON_API_KEY = os.getenv("SILICON_API_KEY")
SILICON_API_BASE = os.getenv("SILICON_API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")

# ==================== SiliconFlow LLM ====================
from typing import List, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class SiliconChatLLM(BaseChatModel):
    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [], "temperature": 0.2}
        for m in messages:
            role, content = "user", str(m)
            if isinstance(m, HumanMessage): role, content = "user", m.content
            elif isinstance(m, SystemMessage): role, content = "system", m.content
            elif isinstance(m, AIMessage): role, content = "assistant", m.content
            payload["messages"].append({"role": role, "content": content})
        resp = requests.post(f"{SILICON_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        text = self._call(messages, stop=stop, **kwargs)
        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "siliconflow-chat"

# 覆盖 Ragas 默认 LLM
import ragas.llms.base as ragas_llms_base
ragas_llms_base.llm_factory = lambda: SiliconChatLLM()

# ==================== 辅助函数 ====================
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
            elif isinstance(c, dict): out.append(c.get("content") or c.get("text") or json.dumps(c, ensure_ascii=False))
            else: out.append(str(c))
        return out
    return [str(x)]

def call_rag_api(question: str, retries: int = 3, delay: float = 0.5):
    payload = {"query": question}
    for attempt in range(1, retries + 1):
        try:
            start_time = time.time()
            resp = requests.post(RAG_API_URL, json=payload, timeout=60)
            latency = time.time() - start_time
            resp.raise_for_status()
            res_json = resp.json()
            answer = res_json.get("answer", "") or res_json.get("output", "")
            contexts_raw = res_json.get("contexts") or res_json.get("context") or []
            contexts = ensure_list_of_str(contexts_raw)
            return answer, contexts, latency
        except Exception as e:
            print(f"[Warn] RAG API 请求失败({attempt}/{retries}): {e}")
            time.sleep(delay)
    return "", [], None

# ==================== 主流程 ====================
def main():
    questions = load_crag_jsonl(EVAL_JSONL_FILE)
    if not questions:
        print("未读取到任何条目")
        return

    questions = questions[:2]  # 仅评测前2条
    data_records, trace_records = [], []

    for item in tqdm(questions, desc="评测中"):
        q_text, gt, interaction_id = item["question"], item.get("ground_truth",""), item.get("interaction_id","")
        answer, contexts, latency = call_rag_api(q_text)

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
    results = evaluate(dataset=dataset, metrics=metrics, llm=SiliconChatLLM())
    results_df = results.to_pandas()

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
