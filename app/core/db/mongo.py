from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.core.config import settings
from app.core.logger_utils import get_logger

logger = get_logger(__file__)


class MongoDatabaseConnector:
    """用于连接MongoDB数据库的单例类。"""

    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.MONGO_DATABASE_HOST)
                # logger.info(
                #     f"成功连接到数据库，URI: {settings.MONGO_DATABASE_HOST}"
                # )
            except ConnectionFailure:
                logger.error("无法连接到数据库。")

                raise

        return cls._instance

    def get_database(self):
        assert self._instance, "数据库连接未初始化"

        return self._instance[settings.MONGO_DATABASE_NAME]

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("数据库连接已关闭。")


connection = MongoDatabaseConnector()
