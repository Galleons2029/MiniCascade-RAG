# -*- coding: utf-8 -*-
"""
批量 RAG 测评脚本（按 question_type 分组，每种类型选取 50 条）
输出 JSONL 文件包含：
- interaction_id, question, answer, ground_truths, contexts, question_type
"""

import os
import json
import time
import logging
from pathlib import Path
from pprint import pformat
from collections import defaultdict

import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---- 日志配置 ----
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "evaluation.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

# ---- 配置 ----
load_dotenv("/home/liujialong/project/MiniCascade-RAG/test/evaluation/rag_evaluate/baseline/.env")

EVAL_JSONL_FILE = Path(os.getenv("CRAG_JSONL"))
RAG_API_URL = os.getenv("RAG_API_URL")

# ---- 本地 embedding 模型 ----
embedder = SentenceTransformer(
    "/home/liujialong/project/MiniCascade-RAG/test/evaluation/rag_evaluate/baseline/all-MiniLM-L6-v2"
)

def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]
    return embedder.encode(texts, convert_to_numpy=True)

# ---- 工具函数 ----
def ensure_list_of_str(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for c in x:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                out.append(c.get("content") or c.get("text") or str(c))
            else:
                out.append(str(c))
        return out
    return [str(x)]

def sanitize_record(record: dict):
    for key in ["interaction_id", "question", "answer", "question_type"]:
        val = record.get(key)
        record[key] = "" if val is None else str(val)
    for key in ["contexts", "ground_truths"]:
        val = record.get(key)
        if val is None:
            record[key] = []
        elif isinstance(val, list):
            record[key] = [str(x) for x in val]
        else:
            record[key] = [str(val)]
    record["latency"] = float(record.get("latency") or 0.0)
    return record

def load_crag_jsonl(jsonl_path: Path):
    data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSONDecodeError at line {i}: {e}")
                continue
            data.append({
                "interaction_id": str(item.get("interaction_id", "")),
                "question": str(item.get("query", "")),
                "ground_truths": ensure_list_of_str([item.get("answer")] + item.get("alt_ans", [])),
                "question_type": str(item.get("question_type", "unknown"))
            })
    return data

# ---- 调用 RAG API ----
def call_rag_api(question: str, interaction_id: str = "", metadata: dict = None,
                 retries: int = 3, delay: float = 0.5, api_delay: float = 0.5, timeout: float = 600.0):
    payload = {"query": question, "interaction_id": interaction_id}
    if metadata:
        payload["metadata"] = metadata

    for attempt in range(1, retries + 1):
        try:
            start_time = time.time()
            resp = requests.post(RAG_API_URL, json=payload, timeout=timeout)
            latency = time.time() - start_time
            resp.raise_for_status()
            res_json = resp.json()
            answer = str(res_json.get("answer", ""))
            contexts_raw = res_json.get("contexts")
            contexts = ensure_list_of_str(contexts_raw)

            time.sleep(api_delay)
            return answer, contexts, latency
        except Exception as e:
            logger.warning(f"RAG API 请求失败({attempt}/{retries}): {e}")
            time.sleep(delay)
    return "", [], 0.0

# ---- 保存 JSONL ----
def save_jsonl(data_records, output_path="results/baseline_task1_output_by_type.jsonl"):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in data_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Raw outputs saved to {output_path}")

# ---- 主函数 ----
def main():
    all_questions = load_crag_jsonl(EVAL_JSONL_FILE)
    if not all_questions:
        logger.warning("未读取到任何条目")
        return

    # 按 question_type 分组
    type_dict = defaultdict(list)
    for q in all_questions:
        type_dict[q["question_type"]].append(q)

    # 每种类型取前 50 条
    selected_questions = []
    for q_type, q_list in type_dict.items():
        selected_questions.extend(q_list[:50])

    data_records = []

    for item in tqdm(selected_questions, desc="评测中"):
        q_text = item["question"]
        gt_list = item.get("ground_truths", [])
        interaction_id = item["interaction_id"]
        question_type = item.get("question_type", "")

        answer, contexts, latency = call_rag_api(
            question=q_text,
            interaction_id=interaction_id,
            metadata={"ground_truths": gt_list, "question_type": question_type}
        )

        record = sanitize_record({
            "interaction_id": interaction_id,
            "question_type": question_type,
            "question": q_text,
            "answer": answer,
            "ground_truths": gt_list,
            "contexts": contexts 
        })
        data_records.append(record)

        logger.info("=" * 60)
        logger.info(f"Interaction ID: {interaction_id}")
        logger.info(f"Question: {q_text}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Question Type: {question_type}")
        logger.info("Contexts:\n" + pformat(contexts))
        logger.info(f"Latency: {latency:.3f}s")
        logger.info("=" * 60)

    save_jsonl(data_records)

if __name__ == "__main__":
    main()
