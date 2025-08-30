# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 15:58
# @Author  : Galleons
# @File    : routers.py

"""
RAG 知识库平台 API 终端
"""

from typing import Annotated
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.api.v1 import inference_v1, chat_v1
from contextlib import asynccontextmanager
from app.core.db.postgre import engine


# 配置日志格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 可在此放置启动检查，如 ping DB
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: None)  # 简单握手
    yield
    await engine.dispose()  # 释放连接池


api_router = FastAPI(
    title="Cascade-RAG",
    summary="多智体级联 RAG 后端",
    version="1.1.0",
    lifespan=lifespan,
)


api_router.include_router(inference_v1.router, prefix="/inference", tags=["inference-v1"])
api_router.include_router(chat_v1.router, prefix="/api", tags=["chat-v1"])

# api_router.include_router(chat_v3.router, prefix="/v3", tags=["chat-v3"])
# api_router.include_router(chat_v2.router, prefix="/v1", tags=["chat-v2"])


fake_secret_token = "coneofsilence"

fake_db = {
    "foo": {"id": "foo", "title": "Foo", "description": "There goes my hero"},
    "bar": {"id": "bar", "title": "Bar", "description": "The bartenders"},
}


class Item(BaseModel):
    id: str
    title: str
    description: str | None = None


@api_router.get("/items/{item_id}", response_model=Item)
async def read_main(item_id: str, x_token: Annotated[str, Header()]):
    if x_token != fake_secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]


@api_router.post("/items/", response_model=Item)
async def create_item(item: Item, x_token: Annotated[str, Header()]):
    if x_token != fake_secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item.id in fake_db:
        raise HTTPException(status_code=409, detail="Item already exists")
    fake_db[item.id] = item
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.router:api_router",
        host="0.0.0.0",
        port=9011,
        # reload=True,
    )
