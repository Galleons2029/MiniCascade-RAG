# -*- coding: utf-8 -*-
"""
批量 RAG 测评脚本
- 使用 SiliconFlow 替换 Ragas 默认 OpenAI
- Langfuse trace 正常调用（手动 start/end + 打分）
- 不使用 Embedding（后续可加）
- 仅评测前2条
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

# ==================== Langfuse ====================
from langfuse import Langfuse

LF_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-398ba093-b8c9-490d-81fd-83a699963578").strip()
LF_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-c65955a8-2a9b-4853-8485-5ed80949af11").strip()
LF_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip()

langfuse = Langfuse(public_key=LF_PUBLIC, secret_key=LF_SECRET, host=LF_HOST)

# ==================== 配置 ====================
EVAL_JSONL_FILE = Path(
    os.getenv("EVAL_JSONL_FILE", r"D:\研究生资料\研0\RAG项目\data\task1\crag_task_1_dev_v4_release.jsonl")
)
RAG_API_URL = os.getenv("RAG_API_URL", "").strip()  # 可为空，不调用 API
SILICON_API_KEY = os.getenv("SILICON_API_KEY", "sk-tqtgmwndmtjghgewhdbjzklfofgmttixpytngwnmtpyptooj").strip()
SILICON_API_BASE = os.getenv("SILICON_API_BASE", "https://api.siliconflow.cn/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B").strip()

# ==================== SiliconFlow LLM ====================
from typing import List, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class SiliconChatLLM(BaseChatModel):
    """Ragas 内部调用的 Chat 模型：请求 SiliconFlow /chat/completions"""
    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [], "temperature": 0.2}
        for m in messages:
            role, content = "user", str(m)
            if isinstance(m, HumanMessage):
                role, content = "user", m.content
            elif isinstance(m, SystemMessage):
                role, content = "system", m.content
            elif isinstance(m, AIMessage):
                role, content = "assistant", m.content
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
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                out.append(c.get("content") or c.get("text") or json.dumps(c, ensure_ascii=False))
            else:
                out.append(str(c))
        return out
    return [str(x)]

def call_rag_api(question: str, retries: int = 1, delay: float = 0.5):
    if not RAG_API_URL:
        return "", [], None
    payload = {"query": question}
    for attempt in range(1, retries+1):
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

        # Langfuse trace
        try:
            span = langfuse.start_span(name="RAG Evaluation")
            span.input = q_text
            span.output = answer
            span.metadata = {"contexts": contexts, "ground_truth": gt, "interaction_id": interaction_id}
            span.end()
            langfuse.flush()

            record = {"interaction_id": interaction_id,"question": q_text,"answer": answer,
                      "contexts": contexts,"ground_truth": gt,"ground_truths":[gt] if gt else [],"latency": latency}
            data_records.append(record)
            trace_records.append((span, record))
        except Exception as e:
            print(f"[Warn] Langfuse trace 失败：{e}")
            data_records.append({"interaction_id": interaction_id,"question": q_text,"answer": answer,
                                 "contexts": contexts,"ground_truth": gt,"ground_truths":[gt] if gt else [],"latency": latency})

    # Ragas 评估
    dataset = Dataset.from_list(data_records)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    results = evaluate(dataset=dataset, metrics=metrics, llm=SiliconChatLLM())

    results_df = results.to_pandas()

    # Langfuse 写分 + 结束 span
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

    # 输出
    print("\n=== RAG 评估结果（整体平均值） ===")
    cols = [c for c in metric_names if c in results_df.columns]
    if "latency" in results_df.columns:
        cols.append("latency")
    print(results_df[cols].mean(numeric_only=True))

    out_path = Path("rag_evaluation_results.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n评估结果已保存到 {out_path.resolve()}")

if __name__ == "__main__":
    main()
