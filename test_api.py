"""HTTP contracts with real LangGraph; optional real PostgreSQL, no LLM API calls."""
import asyncio
from contextlib import asynccontextmanager
import copy
from datetime import datetime, timezone
import os
import sys
import unittest
from uuid import uuid4

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from api import Runtime, create_app
from api_store import ThreadStore
from handbook_graph import build_graph, thread_config
from test_graph import DRAFT, HIT, ScriptedModel


class MemoryStore:
    def __init__(self, saver):
        self.saver, self.rows, self.locks = saver, {}, set()

    async def import_checkpoints(self):
        async for item in self.saver.alist(None):
            thread = item.config["configurable"]["thread_id"]
            if thread not in self.rows:
                await self.create(thread, thread)

    async def create(self, thread_id, title):
        if thread_id in self.rows:
            return None
        now = datetime.now(timezone.utc)
        self.rows[thread_id] = dict(thread_id=thread_id, title=title, created_at=now, updated_at=now)
        return self.rows[thread_id]

    async def get(self, thread_id):
        return self.rows.get(thread_id)

    async def list(self, limit, offset):
        rows = sorted(self.rows.values(), key=lambda row: row["updated_at"], reverse=True)
        return rows[offset:offset + limit], len(rows)

    async def touch(self, thread_id, question=None):
        row = self.rows[thread_id]
        row["updated_at"] = datetime.now(timezone.utc)
        if row["title"] == "Новый диалог" and question is not None:
            row["title"] = question[:120]

    async def completed_turns(self, thread_id, limit, offset):
        ids = sorted([item.checkpoint["id"] async for item in self.saver.alist(thread_config(thread_id))
                      if "history" in item.checkpoint.get("updated_channels", [])
                      and item.metadata.get("source") == "loop"
                      and not any(name.startswith("branch:to:") for name in item.checkpoint.get("updated_channels", []))])
        return ids[offset:offset + limit], len(ids)

    @asynccontextmanager
    async def thread_lock(self, thread_id):
        if thread_id in self.locks:
            yield False
        else:
            self.locks.add(thread_id)
            try:
                yield True
            finally:
                self.locks.remove(thread_id)


class APITests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)

    async def asyncSetUp(self):
        self.saver = InMemorySaver()
        self.store = MemoryStore(self.saver)
        self.ids = []
        self.checks = {"postgres": True, "qdrant": True, "index": True, "models_configured": True}
        self.model = ScriptedModel(
            ScopeDecision=[{"route": "research", "message": ""}] * 12,
            SearchPlan=[{"queries": [{"dense_query": "leave", "sparse_query": "leave"}]}] * 12,
            AnswerDraft=[copy.deepcopy(DRAFT) for _ in range(12)],
            GroundingReview=[{"accepted": True, "issues": []}] * 12)

    def thread(self):
        value = "api-test-" + str(uuid4())
        self.ids.append(value)
        return value

    @asynccontextmanager
    async def factory(self):
        graph = build_graph(self.model, lambda _: [copy.deepcopy(HIT)], checkpointer=self.saver)

        async def readiness():
            return self.checks

        self.runtime = Runtime(graph, self.saver, self.store, readiness)
        yield self.runtime

    @asynccontextmanager
    async def client(self, raise_app_exceptions=False):
        app = create_app(self.factory)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
                                         base_url="http://api.test") as client:
                yield client

    async def new_thread(self, client):
        thread = self.thread()
        response = await client.post("/threads", json={"thread_id": thread})
        self.assertEqual(response.status_code, 201, response.text)
        return thread

    async def test_routes_validation_and_new_thread(self):
        async with self.client() as client:
            routes = (await client.get("/openapi.json")).json()["paths"]
            self.assertEqual(set(routes), {"/healthz", "/readyz", "/threads", "/chat",
                "/threads/{thread_id}/messages", "/threads/{thread_id}/debug", "/threads/{thread_id}/resume"})
            self.assertEqual((await client.get("/docs")).status_code, 200)
            generated = await client.post("/threads", json={})
            self.assertEqual(generated.status_code, 201)
            self.ids.append(generated.json()["thread_id"])
            self.assertEqual((await client.post("/threads", json={"thread_id": ""})).status_code, 422)
            self.assertEqual((await client.post("/threads", json={"thread_id": "invalid/id"})).status_code, 422)
            thread = await self.new_thread(client)
            self.assertEqual((await client.post("/threads", json={"thread_id": thread})).status_code, 409)
            self.assertEqual((await client.post("/chat", json={"thread_id": thread, "message": "  "})).status_code, 422)
            self.assertEqual((await client.post("/chat", json={"thread_id": thread, "message": "x" * 6001})).status_code, 422)
            self.assertEqual((await client.post("/chat", json={"thread_id": thread, "message": "hi", "history": []})).status_code, 422)
            self.assertEqual((await client.get("/threads?limit=0")).status_code, 422)
            unknown = self.thread()
            self.assertEqual((await client.post("/chat", json={"thread_id": unknown, "message": "Отпуск?"})).status_code, 404)
            self.assertEqual((await client.get(f"/threads/{unknown}/messages")).status_code, 404)
            self.assertEqual((await client.get(f"/threads/{thread}/messages")).json()["messages"], [])
            self.assertEqual((await client.get(f"/threads/{thread}/debug?history=false")).json()["latest"], None)
            rows = (await client.get("/threads")).json()["items"]
            self.assertIn(thread, [row["thread_id"] for row in rows])

    async def test_full_messages_survive_history_trimming_and_restart(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            for index in range(5):
                result = await client.post("/chat", json={"thread_id": thread, "message": f"Отпуск {index}?"})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["status"], "ok")
                self.assertEqual(result.json()["sources"][0]["source_id"], "S1")
        async with self.client() as client:
            history = (await client.get(f"/threads/{thread}/messages")).json()
            self.assertEqual(history["total_turns"], 5)
            self.assertEqual(len(history["messages"]), 10)
            self.assertEqual(history["messages"][0]["content"], "Отпуск 0?")
            self.assertEqual(history["messages"][-1]["response"]["reason"], "answered")
            page = (await client.get(f"/threads/{thread}/messages?limit=1&offset=2")).json()
            self.assertEqual(len(page["messages"]), 2)
            self.assertEqual(page["messages"][0]["content"], "Отпуск 2?")
            debug = (await client.get(f"/threads/{thread}/debug?history=false")).json()
            self.assertEqual(len(debug["latest"]["state"]["history"]), 6)
            rows = (await client.get("/threads")).json()["items"]
            self.assertEqual(next(row for row in rows if row["thread_id"] == thread)["title"], "Отпуск 0?")

    async def test_debug_is_read_only_and_paged(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            await client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"})
            calls = len(self.model.calls)
            page = (await client.get(f"/threads/{thread}/debug?limit=2")).json()
            self.assertEqual(len(page["checkpoints"]), 2)
            before = page["next_before"]
            older = (await client.get(f"/threads/{thread}/debug?limit=2&before={before}")).json()
            first_ids = {item["checkpoint_id"] for item in page["checkpoints"]}
            self.assertFalse(first_ids & {item["checkpoint_id"] for item in older["checkpoints"]})
            latest = (await client.get(f"/threads/{thread}/debug?history=false")).json()
            self.assertIn("sources", latest["latest"]["state"])
            self.assertEqual(len(self.model.calls), calls)

    async def test_liveness_and_readiness_are_distinct(self):
        async with self.client() as client:
            self.assertEqual((await client.get("/readyz")).status_code, 200)
            self.checks["postgres"] = False
            self.assertEqual((await client.get("/readyz")).status_code, 503)
            self.assertEqual((await client.get("/healthz")).status_code, 200)
            self.assertEqual(self.model.calls, [])

    async def test_concurrent_chat_is_rejected_and_lock_is_released(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            invoke = self.runtime.graph.ainvoke
            started, release = asyncio.Event(), asyncio.Event()

            async def blocked(*args, **kwargs):
                started.set()
                await release.wait()
                return await invoke(*args, **kwargs)

            self.runtime.graph.ainvoke = blocked
            first = asyncio.create_task(client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"}))
            try:
                await asyncio.wait_for(started.wait(), 10)
                busy = await client.post("/chat", json={"thread_id": thread, "message": "Ещё?"})
                self.assertEqual(busy.status_code, 429)
                self.assertEqual(busy.headers["retry-after"], "1")
            finally:
                release.set()
                first_response = await first
            self.assertEqual(first_response.status_code, 200, first_response.text)
            self.assertFalse(self.runtime.inference_lock.locked())
            async with self.store.thread_lock(thread) as locked:
                self.assertTrue(locked)
                conflict = await client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"})
                self.assertEqual(conflict.status_code, 409)

    async def test_resume_pending_run_and_return_completed_answer(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            original = self.runtime.graph
            self.runtime.graph = original.builder.compile(checkpointer=self.saver, interrupt_after=["supervisor"])
            interrupted = await client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"})
            self.assertEqual(interrupted.status_code, 503)
            self.runtime.graph = original
            blocked = await client.post("/chat", json={"thread_id": thread, "message": "Новый вопрос?"})
            self.assertEqual(blocked.status_code, 409)
            resumed = await client.post(f"/threads/{thread}/resume")
            self.assertEqual(resumed.status_code, 200, resumed.text)
            calls = len(self.model.calls)
            repeated = await client.post(f"/threads/{thread}/resume")
            self.assertEqual(repeated.status_code, 200)
            self.assertEqual(len(self.model.calls), calls)
            self.assertEqual((await client.get(f"/threads/{thread}/messages")).json()["total_turns"], 1)

    async def test_model_and_retrieval_failures_are_not_successful_http(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            self.runtime.graph = build_graph(ScriptedModel(ScopeDecision=[RuntimeError("SECRET")]), lambda _: [], checkpointer=self.saver)
            failed = await client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"})
            self.assertEqual(failed.status_code, 502)
            self.assertNotIn("SECRET", failed.text)

            self.assertEqual(failed.json()["reason"], "llm_error")

            def broken(_):
                raise RuntimeError("SECRET_DATABASE_URL")

            self.runtime.graph = build_graph(ScriptedModel(), broken, checkpointer=self.saver)
            failed = await client.post("/chat", json={"thread_id": thread, "message": "Отпуск?"})
            self.assertEqual(failed.status_code, 503)
            self.assertEqual(failed.json()["reason"], "retrieval_error")
            self.assertNotIn("SECRET", failed.text)

    async def test_clarification_and_scope_refusal_are_normal_responses(self):
        async with self.client() as client:
            thread = await self.new_thread(client)
            self.runtime.graph = build_graph(ScriptedModel(ScopeDecision=[
                {"route": "clarify", "message": "Какую ситуацию нужно решить?"},
                {"route": "refuse", "message": ""}]), lambda _: [], checkpointer=self.saver)
            clarified = await client.post("/chat", json={"thread_id": thread, "message": "Помоги"})
            self.assertEqual(clarified.status_code, 200)
            self.assertEqual(clarified.json()["status"], "clarify")
            refused = await client.post("/chat", json={"thread_id": thread, "message": "Другой вопрос"})
            self.assertEqual(refused.status_code, 200)
            self.assertEqual(refused.json()["status"], "refuse")

    async def test_existing_cli_thread_can_be_opened(self):
        thread = self.thread()
        graph = build_graph(self.model, lambda _: [copy.deepcopy(HIT)], checkpointer=self.saver)
        await graph.ainvoke({"question": "CLI вопрос", "history": []}, config=thread_config(thread))
        async with self.client(raise_app_exceptions=True) as client:
            response = await client.get(f"/threads/{thread}/messages")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["messages"][0]["content"], "CLI вопрос")
            self.assertEqual((await client.post("/threads", json={"thread_id": thread})).status_code, 409)


@unittest.skipUnless(os.getenv("TEST_CHECKPOINT_DATABASE_URL"), "Set TEST_CHECKPOINT_DATABASE_URL for PostgreSQL HTTP tests")
class PostgresAPITests(APITests):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.pool = AsyncConnectionPool(os.environ["TEST_CHECKPOINT_DATABASE_URL"], min_size=1, max_size=4,
                                       open=False, kwargs={"autocommit": True, "row_factory": dict_row})
        await self.pool.open(wait=True)
        self.saver = AsyncPostgresSaver(self.pool, serde=JsonPlusSerializer(allowed_msgpack_modules=[]))
        await self.saver.setup()
        self.store = ThreadStore(self.pool)
        await self.store.setup()

    async def asyncTearDown(self):
        try:
            for thread in self.ids:
                await self.saver.adelete_thread(thread)
                async with self.pool.connection() as conn:
                    await conn.execute("DELETE FROM handbook_api_threads WHERE thread_id = %s", (thread,))
        finally:
            await self.pool.close()


if __name__ == "__main__":
    unittest.main()
