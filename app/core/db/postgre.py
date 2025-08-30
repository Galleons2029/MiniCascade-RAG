# -*- coding: utf-8 -*-
# @Time   : 2025/8/11 11:40
# @Author : Galleons
# @File   : postgre.py

"""
For PostgreSQL Connect,
使用连接池创建，取消单例类
"""

from app.configs import postgres_config
from app.core.logger_utils import get_logger
from contextlib import contextmanager
from typing import Any, Generator, Optional
from sqlmodel import Session, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = get_logger(__file__)
engine: AsyncEngine = create_async_engine(
    postgres_config.POSTGRES_URL,
    echo=postgres_config.ECHO_SQL,
    pool_size=postgres_config.POOL_SIZE,
    max_overflow=postgres_config.MAX_OVERFLOW,
    pool_timeout=postgres_config.COMMAND_TIMEOUT,
    pool_recycle=postgres_config.POOL_RECYCLE,
    pool_pre_ping=True,  # 断线自动探活
)


async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,  # 提升序列化体验
)


@asynccontextmanager
async def session_scope():
    async with async_session() as session:  # type: AsyncSession
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session():
    async with session_scope() as session:
        yield session


class PostgreSQLConnector:
    """用于连接PostgreSQL数据库的单例类。"""

    _engine: Optional[Any] | None = None

    def __init__(self) -> None:
        if self._engine is None:
            dsn = postgres_config.TIMESCALE_URL
            if not dsn:
                host = postgres_config.HOST
                port = getattr(postgres_config, "POSTGRES_DATABASE_PORT", None)
                user = getattr(postgres_config, "POSTGRES_DATABASE_USER", None)
                password = getattr(postgres_config, "POSTGRES_DATABASE_PASSWORD", None)
                database = getattr(postgres_config, "POSTGRES_DATABASE_NAME", None)

                if all([host, port, user, password, database]):
                    dsn = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
                else:
                    logger.error(
                        "PostgreSQL 配置缺失，无法初始化连接。请提供 POSTGRES_DATABASE_URL 或完整的主机/端口/用户名/密码/数据库名。"  # noqa: E501
                    )
                    raise RuntimeError("PostgreSQL configuration is incomplete")

            try:
                self._engine = create_engine(dsn, pool_pre_ping=True, echo=postgres_config.DEBUG)
                with self._engine.connect() as conn:
                    conn.exec_driver_sql("SELECT 1")
                logger.info("成功初始化 PostgreSQL 连接引擎")
            except Exception as e:
                logger.error(f"初始化 PostgreSQL 连接失败: {str(e)}")
                raise

    def get_engine(self) -> Any:
        assert self._engine is not None, "PostgreSQL 引擎未初始化"
        return self._engine

    def get_session(self) -> Session:
        assert self._engine is not None, "PostgreSQL 引擎未初始化"
        return Session(self._engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute(self, sql: str, params: Optional[dict] = None) -> Any:
        assert self._engine is not None, "PostgreSQL 引擎未初始化"
        with self._engine.connect() as conn:
            return conn.exec_driver_sql(sql, params or {})

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("PostgreSQL 引擎已关闭")
