# -*- coding: utf-8 -*-
# @Time    : 2025/6/5 14:34
# @Author  : Galleons
# @File    : embeddings.py

"""
流式管道——嵌入算子模块
更新：去处本地依赖，全面接入 silicon embedding model API 以方便个人本地化部署
"""

import logging

# from xinference.client import Client
import numpy as np
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# from FlagEmbedding import BGEM3FlagModel
# embed_model_bge = BGEM3FlagModel('/data/model_cache/bge-m3', use_fp16=True)
# from app.pipeline.feature_pipeline.config import settings

# TODO:重写代码以及图像嵌入
# def embedd_repositories(text: str):
#     model = INSTRUCTOR("hkunlp/instructor-xl")
#     sentence = text
#     instruction = "Represent the structure of the repository"
#     return model.encode([instruction, sentence])

# client = Client("http://localhost:9997")
# embed_model_raw = client.get_model(settings.EMBEDDING_MODEL_ID)
# embed_model = client.get_model(settings.EMBEDDING_MODEL_ID)


url = "https://api.siliconflow.cn/v1/embeddings"
headers = {
    "Authorization": "Bearer sk-ulkoelnhlbxhkhpoggtemmamkgnnoshpirggznctlevxlcqy",
    "Content-Type": "application/json",
}


def embedd_text(text: str) -> np.ndarray:
    # embedding_list = embed_model.create_embedding(text)['data'][0]['embedding']
    payload = {"model": "BAAI/bge-m3", "input": text}
    embedding_list = requests.post(url, json=payload, headers=headers).json()["data"][0]["embedding"]
    return np.array(embedding_list)


def embedd_text_tolist(text: str) -> list[int]:
    # embedding_list = embed_model.create_embedding(text)['data'][0]['embedding']
    payload = {"model": "BAAI/bge-m3", "input": text}
    embedding_list = requests.post(url, json=payload, headers=headers).json()["data"][0]["embedding"]
    return embedding_list


def hybrid_embedding(texts: list[str]) -> dict:
    # output = embed_model_bge.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    # idx, vals = zip(*output['lexical_weights'][0].items())
    # return {'dense': output['dense_vecs'][0], 'sparse': models.SparseVector(indices=idx, values=vals)}
    pass


# 代码嵌入模型
# def embedd_repositories(text: str):
#     # TODO：优化代码嵌入模型部分，寻找合适模型
#     embedding_list = embed_model.create_embedding(input)['data'][0]['embedding']
#     embedding_array = np.array(embedding_list)
#     return embedding_array
#
#
#
# class EmbeddingClientManager:
#     _client: Optional[Client] = None
#     _embed_model = None
#
#     @classmethod
#     @contextmanager
#     def get_client_context(cls):
#         """获取 Xinference 客户端的上下文管理器"""
#         if cls._client is None:
#             try:
#                 cls._client = Client("http://192.168.100.111:9997")
#                 cls._embed_model = cls._client.get_model("bge-m3")
#             except Exception as e:
#                 logger.error(f"Failed to initialize Xinference client: {str(e)}")
#                 raise
#
#         try:
#             yield cls._embed_model
#         except Exception as e:
#             logger.error(f"Error during Xinference operation: {str(e)}")
#             cls._client = None
#             cls._embed_model = None
#             raise
#
#     @classmethod
#     def check_health(cls) -> bool:
#         """检查 Xinference 服务是否可用"""
#         try:
#             with cls.get_client_context() as embed_model:
#                 # 尝试进行一个简单的嵌入操作
#                 test_result = embed_model.create_embedding("test")
#                 return bool(test_result and test_result.get('data'))
#         except Exception as e:
#             logger.error(f"Xinference health check failed: {str(e)}")
#             return False
#
#
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10)
# )
# async def vectorize(text: str) -> List[float]:
#     """
#     使用 Xinference 生成文本嵌入向量
#
#     Args:
#         text: 需要生成嵌入向量的文本
#
#     Returns:
#         List[float]: 嵌入向量
#
#     Raises:
#         Exception: 当嵌入生成失败时抛出异常
#     """
#     if not text:
#         return []
#
#     try:
#         with EmbeddingClientManager.get_client_context() as embed_model:
#             result = embed_model.create_embedding(text)
#             if not result or 'data' not in result or not result['data']:
#                 raise ValueError("Invalid embedding result")
#
#             embedding = result['data'][0]['embedding']
#             return embedding
#
#     except Exception as e:
#         logger.error(f"Error generating embedding for text: {str(e)}")
#         raise
#
#
# # 可选：为短文本添加缓存以提高性能
# @lru_cache(maxsize=1000)
# def get_cached_embedding(text: str) -> Coroutine[Any, Any, list[float]]:
#     """
#     为短文本提供缓存的嵌入向量生成
#     仅用于长度小于500字符的文本
#     """
#     if len(text) > 500:
#         return vectorize(text)
#     return vectorize(text)
#
#
# if __name__ == "__main__":
#     import time
#
#     start_time = time.time()
#     for i in range(10):
#         embedd_text('成功连接到数据库,现在开始使用测试样例测试运行时间！')
#     end_time = time.time()
#
#     print(f"函数1执行时间为: {end_time - start_time:.6f} 秒")
#
#     start_time = time.time()
#     for i in range(10):
#         hybrid_embedding(['成功连接到数据库,现在开始使用测试样例测试运行时间！'])
#     end_time = time.time()
#
#     print(f"函数2执行时间为: {end_time - start_time:.6f} 秒")
