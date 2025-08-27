from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    PROJECT_NAME: str = 'Cascade-RAG'
    # MongoDB配置
    MONGO_DATABASE_HOST: str = (
        "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    )
    MONGO_DATABASE_NAME: str = "twin"

    # 硅基流动API
    Silicon_base_url: str | None = "https://api.siliconflow.cn/v1"
    SILICON_KEY: str | None = 'EMPTY'

    Silicon_model_v1: str | None = "Qwen/Qwen3-32B"
    Silicon_model_mini: str | None = "Qwen/Qwen2.5-7B-Instruct"
    Silicon_model_rerank: str | None = "BAAI/bge-reranker-v2-m3"

    MODEL_PATH: str = "/data/model_weights/Qwen3-4B"
    LOCAL: str = 'http://localhost:9011/v1'

    # 消息队列配置
    RABBITMQ_DEFAULT_USERNAME: str = "guest"
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"
    RABBITMQ_HOST: str = "localhost" # localhost in local deployment, mq in docker
    RABBITMQ_PORT: int = 5672
    RABBITMQ_QUEUE_NAME: str = "rag_test"

    # QdrantDB配置
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_DATABASE_HOST: str = "localhost" # localhost in local deployment, qdrant in docker
    QDRANT_DATABASE_PORT: int = 6333
    USE_QDRANT_CLOUD: bool = False
    QDRANT_APIKEY: str | None = None

    # OpenAI配置
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # LLM模型配置
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_ID: str = "qwen3"
    DEPLOYMENT_ENDPOINT_NAME: str = "twin"

    MAX_INPUT_TOKENS: int = 1536  # 输入文本的最大长度
    MAX_TOTAL_TOKENS: int = 2048  # 生成文本的最大长度（包括输入文本）
    MAX_BATCH_TOTAL_TOKENS: int = 2048  # 限制生成过程中可以并行处理的token数量

    # 嵌入模型配置
    EMBEDDING_MODEL_ID: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 1024
    EMBEDDING_MODEL_DEVICE: str = "gpu"

    def patch_localhost(self) -> None:
        self.MONGO_DATABASE_HOST = "mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set"
        self.QDRANT_DATABASE_HOST = "localhost"
        self.RABBITMQ_HOST = "localhost"


settings = AppSettings()
