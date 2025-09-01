import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import torch

# 自动加载 .env 文件
load_dotenv()

# ===== 配置 =====
CRAG_JSONL = Path(os.environ.get("CRAG_JSONL", "").strip()) if os.environ.get("CRAG_JSONL") else None
SLICE_SIZE = int(os.environ.get("SLICE_SIZE", 1000))

MODEL_NAME = os.environ["MODEL_NAME"]
SILICON_API_KEY = os.environ["SILICON_API_KEY"]
SILICON_API_BASE = os.environ["SILICON_API_BASE"]

# 指定 embedding GPU 卡号
EMBEDDER_GPU = os.environ.get("EMBEDDER_GPU", "0")  # 默认 cuda:0
device = f"cuda:{EMBEDDER_GPU}" if torch.cuda.is_available() else "cpu"

# 使用本地模型路径，并指定 GPU
embedder: SentenceTransformer = SentenceTransformer(r"./all-MiniLM-L6-v2", device=device)

# ===== 工具函数 =====
def clean_html(raw_html: str) -> str:
    """清洗 HTML，去掉 script/style，只保留正文"""
    soup = BeautifulSoup(raw_html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    txt = soup.get_text(" ")
    return " ".join(txt.split())

def slice_text(text: str, slice_size: int = SLICE_SIZE) -> List[str]:
    return [text[i:i + slice_size] for i in range(0, len(text), slice_size)]

def build_kb_from_example(example: Dict[str, Any]):
    """从一条 CRAG 数据构建临时知识库"""
    docs, metadatas = [], []

    for r in example.get("search_results", []) or []:
        raw_html = r.get("page_result") or ""
        if not raw_html.strip():
            continue
        text = clean_html(raw_html)
        if not text:
            continue
        for chunk in slice_text(text):
            docs.append(chunk)
            metadatas.append({
                "page_name": r.get("page_name", ""),
                "page_url": r.get("page_url", ""),
            })

    if not docs:
        return None, None, []

    # embedding 在 GPU
    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)

    # FAISS 在 CPU
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, docs, metadatas

def retrieve(query: str, index, docs, metadatas, top_k: int = 5):
    """检索相关片段"""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return [
        {
            "text": docs[idx],
            "score": float(score),
            **metadatas[idx],
        }
        for score, idx in zip(D[0], I[0])
    ]

def generate_answer(query: str, contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    """根据 query 和上下文调用 LLM"""
    if contexts:
        context_texts = "\n\n".join([c["text"][:SLICE_SIZE] for c in contexts])
        prompt = f"""你是一个智能问答助手。
用户问题：{query}
以下是相关的网页资料：
{context_texts}

请基于资料用英文回答用户问题，如果资料不足，请明确说明。"""
    else:
        prompt = f"""你是一个智能问答助手。
用户问题：{query}

请直接回答问题，如果无法确定，请说明不知道。"""

    headers = {
        "Authorization": f"Bearer {SILICON_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个可靠的问答助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(f"{SILICON_API_BASE}/chat/completions",
                         headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ===== FastAPI 服务 =====
app = FastAPI(title="LLM RAG API", version="0.5")

class QueryRequest(BaseModel):
    query: str
    interaction_id: str  # 必须指定 interaction_id
    top_k: Optional[int] = 5

@app.post("/answer")
def answer(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return {"answer": "", "contexts": []}

    if not CRAG_JSONL or not CRAG_JSONL.exists():
        # 无知识库，直接调用 LLM
        answer_text = generate_answer(query, None)
        return {"answer": answer_text, "contexts": []}

    # 查找指定 interaction_id
    example = None
    with CRAG_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("interaction_id") == req.interaction_id:
                example = item
                break

    if not example:
        return {"answer": f"未找到 interaction_id={req.interaction_id} 的数据。", "contexts": []}

    # 构建 KB 并检索
    index, docs, metadatas = build_kb_from_example(example)
    if not docs:
        return {"answer": "该条数据没有可用的 search_results。", "contexts": []}

    contexts = retrieve(query, index, docs, metadatas, top_k=req.top_k or 5)
    answer_text = generate_answer(query, contexts)
    return {"answer": answer_text, "contexts": contexts}
