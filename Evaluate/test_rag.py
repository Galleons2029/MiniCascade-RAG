import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import requests

# ===== 配置 =====
CRAG_JSONL = Path(os.getenv("CRAG_JSONL", r"D:\研究生资料\研0\RAG项目\data\task1\crag_task_1_dev_v4_release.jsonl"))
MAX_DOCS = int(os.getenv("MAX_DOCS", "2000"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "10"))
SLICE_SIZE = int(os.getenv("SLICE_SIZE", "512"))  # 每条切片字符数

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
SILICON_API_KEY = os.getenv("SILICON_API_KEY", "sk-tqtgmwndmtjghgewhdbjzklfofgmttixpytngwnmtpyptooj")
SILICON_API_BASE = os.getenv("SILICON_API_BASE", "https://api.siliconflow.cn/v1")

# ===== 数据和索引懒加载 =====
docs: List[str] = []
metadatas: List[Dict[str, Any]] = []
index: Optional[faiss.IndexFlatIP] = None
embedder: Optional[SentenceTransformer] = None
data_initialized = False

MAX_DOCS = int(os.getenv("MAX_DOCS", "2"))  # 只取前 N 条

# ===== 工具函数 =====
def slice_text(text: str, slice_size: int = SLICE_SIZE) -> List[str]:
    return [text[i:i+slice_size] for i in range(0, len(text), slice_size)]

# ===== 初始化数据 =====
def init_data():
    global docs, metadatas, index, embedder, data_initialized
    if data_initialized:
        return

    if not CRAG_JSONL.exists():
        raise FileNotFoundError(f"CRAG JSONL file not found: {CRAG_JSONL}")

    print(f"Loading CRAG jsonl from: {CRAG_JSONL}")
    docs.clear()
    metadatas.clear()
    num_items = 0

    with CRAG_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            num_items += 1
            if num_items > MAX_DOCS:
                break
            interaction_id = item.get("interaction_id", "")
            results = item.get("search_results") or []
            for r in results:
                text = r.get("page_result") or ""
                if not text.strip():
                    continue
                for chunk in slice_text(text):
                    docs.append(chunk)
                    metadatas.append({
                        "interaction_id": interaction_id,
                        "page_name": r.get("page_name", ""),
                        "page_url": r.get("page_url", "")
                    })

    print(f"Loaded {len(docs)} slices from {num_items} documents.")

    # 构建向量索引
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    data_initialized = True
    print("Knowledge base and index initialized.")

# ===== 向量检索 =====
def retrieve(query: str, top_k: int = 4):
    if not data_initialized:
        init_data()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    contexts = []
    for score, idx in zip(D[0], I[0]):
        contexts.append({
            "text": docs[idx],
            "score": float(score),
            **metadatas[idx]
        })
    return contexts

# ===== 调用大模型生成答案 =====
def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    # 每条切片最多 512 字符
    context_texts = "\n\n".join([c["text"][:SLICE_SIZE] for c in contexts])
    prompt = f"""你是一个智能问答助手。
用户问题：{query}
以下是相关的资料：
{context_texts}

请基于资料回答用户问题，如果资料不足，请明确说明。"""

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
        "temperature": 0.2
    }
    resp = requests.post(f"{SILICON_API_BASE}/chat/completions", headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ===== FastAPI 服务 =====
app = FastAPI(title="LLM RAG API", version="0.2")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.get("/")
def root():
    return {"status": "ok", "docs": len(docs)}

@app.post("/init")
def init_endpoint():
    init_data()
    return {"status": "initialized", "docs": len(docs)}

@app.post("/answer")
def answer(req: QueryRequest):
    if not data_initialized:
        init_data()
    query = (req.query or "").strip()
    if not query:
        return {"answer": "", "contexts": []}
    top_k = req.top_k or TOP_K_DEFAULT
    contexts = retrieve(query, top_k=top_k)
    answer_text = generate_answer(query, contexts)
    return {"answer": answer_text, "contexts": contexts}
