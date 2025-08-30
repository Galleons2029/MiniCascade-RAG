# -*- coding: utf-8 -*-
# @Time   : 2025/8/11 15:51
# @Author : Galleons
# @File   : db_config.py

"""
数据库连接配置
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, Field

ROOT_DIR = Path(__file__).resolve().parents[2] / ".env"


class PostgresConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8", extra="ignore")

    DEBUG: bool = True
    POSTGRES_DATABASE_HOST: str = "localhost"
    POSTGRES_DATABASE_PORT: int = 5433
    TIMESCALE_URL: str | None = None

    PG_HOST: str = "localhost"
    PG_PORT: int = 5433
    PG_USER: str = "admin"
    PG_PASSWORD: str = "password"
    PG_DB: str = "postgres"

    # psycopg3 异步
    POSTGRES: PostgresDsn = Field(default="postgresql+psycopg://user:pass@localhost:5432/mydb")
    POSTGRES_URL: str = Field(default="postgresql+psycopg://user:pass@localhost:5432/mydb")

    # 连接池
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_RECYCLE: int = 1800  # 秒，避免空闲连接被中断
    ECHO_SQL: bool = False
    COMMAND_TIMEOUT: float = 5.0  # 秒，asyncpg 全局命令超时

    @property
    def db_url_async(self) -> str:
        return f"postgresql+asyncpg://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8", extra="ignore")

    DEBUG: bool = True
    COLLECTION_TEST: str | None = "multi_demo"

    MULTIMODAL_SIZE: int | None = 1024

    # 连接池
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_RECYCLE: int = 1800  # 秒，避免空闲连接被中断
    ECHO_SQL: bool = False
    COMMAND_TIMEOUT: float = 5.0  # 秒，asyncpg 全局命令超时
