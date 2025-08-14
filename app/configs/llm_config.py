# -*- coding: utf-8 -*-
# @Time   : 2025/8/13 15:44
# @Author : Galleons
# @File   : llm_config.py

"""
这里是文件说明
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)+'/.env'


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8", extra='ignore')

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "bge-m3"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 1024
    EMBEDDING_MODEL_DEVICE: str = "gpu"

    # 硅基流动API
    SILICON_KEY: str | None = None
    SILICON_BASE_URL: str | None = "https://api.siliconflow.cn/v1"
    SILICON_EMBEDDING: str | None = "https://api.siliconflow.cn/v1/embeddings"

    ALI_KEY: str | None = None

settings = LLMConfig()


if __name__ == '__main__':
    config = LLMConfig()
    print(config.ALI_KEY)
    print(ROOT_DIR)