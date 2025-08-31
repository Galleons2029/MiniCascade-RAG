# -*- coding: utf-8 -*-
# @Time   : 8/1/25 3:35 PM
# @Author : Galleons
# @File   : rag_config.py

"""
这里是文件说明
"""

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )

    # RAG Configuration Settings
    TOP_K: int = 10
    KEEP_TOP_K: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SIMILARITY_THRESHOLD: float = 0.7

    # Vector Database Settings
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    COLLECTION_NAME: str = "default"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
        )


# Create a global instance of RAG settings
rag_settings = RAGConfig()
