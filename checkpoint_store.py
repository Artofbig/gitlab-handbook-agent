"""Lifecycle of the asynchronous PostgreSQL LangGraph checkpointer."""
from contextlib import asynccontextmanager
import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

DEFAULT_DATABASE_URL = (
    "postgresql://handbook:handbook_local@localhost:5433/handbook?connect_timeout=10"
)


@asynccontextmanager
async def postgres_checkpointer(database_url=None, *, setup=True):
    url = database_url or os.getenv("CHECKPOINT_DATABASE_URL") or DEFAULT_DATABASE_URL
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        # State contains only JSON-compatible primitives; no arbitrary type imports.
        saver.serde = JsonPlusSerializer(allowed_msgpack_modules=[])
        if setup:
            await saver.setup()  # Idempotent migrations; requires DDL permission at startup.
        yield saver
