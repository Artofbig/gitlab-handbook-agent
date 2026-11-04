"""Local HTTP API for the Handbook graph; run with python run_api.py."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from api_store import ThreadStore
from checkpoint_store import DEFAULT_DATABASE_URL
from handbook_graph import ChatResponse, ConversationStateError, GraphSettings, aask, aresume, build_graph, thread_config
from handbook_retriever import HandbookRetriever
from hybrid_search import read_state, resolve_collection
from inspect_graph import checkpoint_view
from node_models import create_models
from reranking import DEFAULT_RERANK_MODEL, LocalReranker

logger = logging.getLogger(__name__)
ThreadId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200, pattern=r"^[^/\x00]+$")]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6000)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThreadCreate(APIModel):
    thread_id: ThreadId | None = None
    title: str = Field(default="Новый диалог", min_length=1, max_length=120)


class ThreadInfo(APIModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ThreadList(APIModel):
    items: list[ThreadInfo]
    total: int
    limit: int
    offset: int


class ChatRequest(APIModel):
    thread_id: ThreadId
    message: MessageText


class ChatResult(ChatResponse):
    thread_id: str
    checkpoint_id: str | None


class Message(APIModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    response: ChatResponse | None = None


class MessageList(APIModel):
    thread_id: str
    messages: list[Message]
    total_turns: int
    limit: int
    offset: int


@dataclass
class Runtime:
    graph: object
    saver: object
    store: object
    readiness: object
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@asynccontextmanager
async def production_runtime():
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    url = os.getenv("CHECKPOINT_DATABASE_URL") or DEFAULT_DATABASE_URL
    pool = AsyncConnectionPool(url, min_size=1, max_size=4, timeout=10, open=False,
                               kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0})
    # Pool warnings can contain connection details. Report only exception types.
    logging.getLogger("psycopg.pool").setLevel(logging.CRITICAL)
    client = None
    try:
        await pool.open(wait=True, timeout=15)
        saver = AsyncPostgresSaver(pool, serde=JsonPlusSerializer(allowed_msgpack_modules=[]))
        await saver.setup()
        store = ThreadStore(pool)
        await store.setup()
        await store.import_checkpoints()
        client = QdrantClient(url=os.getenv("QDRANT_URL") or "http://localhost:6333",
                              api_key=os.getenv("QDRANT_API_KEY") or None, timeout=10)
        alias = os.getenv("QDRANT_ALIAS") or "handbook"
        threads = int(os.getenv("API_MODEL_THREADS", "1"))
        reranker = None if os.getenv("API_RERANK_ENABLED", "true").lower() == "false" else LocalReranker(
            model=os.getenv("API_RERANK_MODEL") or DEFAULT_RERANK_MODEL, threads=threads,
            batch_size=int(os.getenv("API_RERANK_BATCH_SIZE", "1")))
        top_k = int(os.getenv("API_TOP_K", "8"))
        retrieve = HandbookRetriever(client, alias=alias, threads=threads, reranker=reranker, top_k=top_k,
                                     candidates=int(os.getenv("API_CANDIDATES", "100")),
                                     retrieval_k=int(os.getenv("API_RETRIEVAL_K", "30")),
                                     rerank_limit=int(os.getenv("API_RERANK_LIMIT", "40")))
        graph = build_graph(create_models(), retrieve, GraphSettings(max_sources=top_k,
                            max_context_chars=int(os.getenv("API_MAX_CONTEXT_CHARS", "24000"))), checkpointer=saver)

        async def readiness():
            checks = {"postgres": False, "qdrant": False, "index": False, "models_configured": True}
            try:
                await store.ping()
                checks["postgres"] = True
            except Exception:
                pass
            try:
                await asyncio.to_thread(client.get_collections)
                checks["qdrant"] = True
            except Exception:
                return checks
            try:
                def index_status():
                    collection = resolve_collection(client, alias)
                    return read_state(client, collection)["status"] == "ready"
                checks["index"] = await asyncio.to_thread(index_status)
            except Exception:
                pass
            return checks

        yield Runtime(graph, saver, store, readiness)
    except Exception as exc:
        logger.error("API runtime failed (%s)", type(exc).__name__)
        # Do not propagate errors containing DSNs/API keys into server logs.
        raise RuntimeError(f"API runtime failed ({type(exc).__name__}); check PostgreSQL and model settings") from None
    finally:
        if client is not None:
            client.close()
        await pool.close()


def create_app(runtime_factory=production_runtime):
    @asynccontextmanager
    async def lifespan(app):
        async with runtime_factory() as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="GitLab Handbook Agent", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def backend_error(request: Request, exc: Exception):
        logger.error("HTTP request failed (%s)", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": {"code": "backend_unavailable",
                            "message": "Сервис временно недоступен. Состояние запуска можно проверить через debug."}})

    def runtime(request):
        return request.app.state.runtime

    async def require_thread(rt, thread_id):
        row = await rt.store.get(thread_id)
        if row is None:
            # Existing CLI threads remain usable without recreating them.
            saved = await rt.saver.aget_tuple(thread_config(thread_id))
            if saved is not None:
                await rt.store.import_checkpoints()
                row = await rt.store.get(thread_id)
        if row is None:
            raise HTTPException(404, detail={"code": "thread_not_found"})
        return row

    @app.get("/healthz", tags=["health"])
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def ready(request: Request):
        checks = await runtime(request).readiness()
        good = all(checks.values())
        return JSONResponse(status_code=200 if good else 503,
                            content={"status": "ready" if good else "not_ready", "checks": checks})

    @app.post("/threads", response_model=ThreadInfo, status_code=201, tags=["threads"])
    async def create_thread(body: ThreadCreate, request: Request):
        rt = runtime(request)
        thread_id = body.thread_id or str(uuid4())
        # Do not accidentally reuse an existing CLI thread's ID.
        if await rt.saver.aget_tuple(thread_config(thread_id)) is not None:
            raise HTTPException(409, detail={"code": "thread_exists"})
        row = await rt.store.create(thread_id, body.title.strip() or "Новый диалог")
        if row is None:
            raise HTTPException(409, detail={"code": "thread_exists"})
        return row

    @app.get("/threads", response_model=ThreadList, tags=["threads"])
    async def list_threads(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
        store = runtime(request).store
        await store.import_checkpoints()
        items, total = await store.list(limit, offset)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get("/threads/{thread_id}/messages", response_model=MessageList, tags=["threads"])
    async def messages(thread_id: ThreadId, request: Request, limit: int = Query(50, ge=1, le=100),
                       offset: int = Query(0, ge=0)):
        rt = runtime(request)
        await require_thread(rt, thread_id)
        ids, total = await rt.store.completed_turns(thread_id, limit, offset)
        result = []
        for checkpoint_id in ids:
            config = thread_config(thread_id)
            config["configurable"]["checkpoint_id"] = checkpoint_id
            item = await rt.saver.aget_tuple(config)
            state = item.checkpoint["channel_values"]
            checked = ChatResponse.model_validate(state["response"])
            created = item.checkpoint["ts"]
            result.extend([
                {"id": checkpoint_id + ":user", "role": "user", "content": state["question"], "created_at": created},
                {"id": checkpoint_id + ":assistant", "role": "assistant", "content": checked.answer,
                 "response": checked, "created_at": created}])
        return {"thread_id": thread_id, "messages": result, "total_turns": total, "limit": limit, "offset": offset}

    @app.get("/threads/{thread_id}/debug", tags=["debug"])
    async def debug(thread_id: ThreadId, request: Request, history: bool = True,
                    limit: int = Query(100, ge=1, le=500), before: str | None = Query(None, max_length=200)):
        rt = runtime(request)
        await require_thread(rt, thread_id)
        config = thread_config(thread_id)
        if not history:
            item = await rt.saver.aget_tuple(config)
            return {"thread_id": thread_id, "latest": checkpoint_view(item) if item else None}
        before_config = {"configurable": {"thread_id": thread_id, "checkpoint_id": before}} if before else None
        items = [checkpoint_view(item) async for item in rt.saver.alist(config, limit=limit, before=before_config)]
        return {"thread_id": thread_id, "checkpoints": list(reversed(items)),
                "next_before": items[-1]["checkpoint_id"] if len(items) == limit else None}

    async def execute(rt, thread_id, question, response, resume=False):
        await require_thread(rt, thread_id)
        if rt.inference_lock.locked():
            raise HTTPException(429, detail={"code": "agent_busy"}, headers={"Retry-After": "1"})
        async with rt.inference_lock:
            async with rt.store.thread_lock(thread_id) as acquired:
                if not acquired:
                    raise HTTPException(409, detail={"code": "thread_busy"})
                try:
                    answer = await aresume(rt.graph, thread_id=thread_id) if resume else await aask(
                        rt.graph, question, thread_id=thread_id)
                except ConversationStateError as exc:
                    raise HTTPException(409, detail={"code": "thread_state_conflict", "message": str(exc)}) from None
                await rt.store.touch(thread_id, question)
                saved = await rt.saver.aget_tuple(thread_config(thread_id))
                response.status_code = 502 if answer.reason == "llm_error" else 503 if answer.reason == "retrieval_error" else 200
                return {**answer.model_dump(), "thread_id": thread_id,
                        "checkpoint_id": saved.checkpoint["id"]}

    @app.post("/chat", response_model=ChatResult, tags=["chat"],
              responses={404: {"description": "Unknown thread"}, 409: {"description": "Thread is busy or needs resume"},
                         429: {"description": "One request is already running"},
                         502: {"model": ChatResult}, 503: {"description": "Backend or retrieval unavailable"}})
    async def chat(body: ChatRequest, request: Request, response: Response):
        return await execute(runtime(request), body.thread_id, body.message, response)

    @app.post("/threads/{thread_id}/resume", response_model=ChatResult, tags=["chat"])
    async def resume(thread_id: ThreadId, request: Request, response: Response):
        return await execute(runtime(request), thread_id, None, response, resume=True)

    return app


app = create_app()
