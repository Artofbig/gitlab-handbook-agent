"""Thread catalog and completed-turn paging alongside LangGraph's checkpoint tables."""
from contextlib import asynccontextmanager


class ThreadStore:
    def __init__(self, pool):
        self.pool = pool

    async def setup(self):
        async with self.pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS handbook_api_threads (
                    thread_id TEXT PRIMARY KEY CHECK (length(thread_id) BETWEEN 1 AND 200),
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

    async def import_checkpoints(self):
        # CLI conversations also appear in the API. No checkpoint content is modified.
        async with self.pool.connection() as conn:
            await conn.execute("""
                INSERT INTO handbook_api_threads (thread_id, title, created_at, updated_at)
                SELECT thread_id, left(thread_id, 120),
                       min((checkpoint->>'ts')::timestamptz), max((checkpoint->>'ts')::timestamptz)
                FROM checkpoints WHERE checkpoint_ns = '' AND length(thread_id) BETWEEN 1 AND 200
                GROUP BY thread_id
                ON CONFLICT (thread_id) DO UPDATE
                SET updated_at = greatest(handbook_api_threads.updated_at, EXCLUDED.updated_at)
            """)

    async def create(self, thread_id, title):
        async with self.pool.connection() as conn:
            return await (await conn.execute("""
                INSERT INTO handbook_api_threads(thread_id, title) VALUES (%s, %s)
                ON CONFLICT DO NOTHING RETURNING *
            """, (thread_id, title))).fetchone()

    async def get(self, thread_id):
        async with self.pool.connection() as conn:
            return await (await conn.execute(
                "SELECT * FROM handbook_api_threads WHERE thread_id = %s", (thread_id,))).fetchone()

    async def list(self, limit, offset):
        async with self.pool.connection() as conn:
            total = await (await conn.execute("SELECT count(*) AS total FROM handbook_api_threads")).fetchone()
            rows = await (await conn.execute("""
                SELECT * FROM handbook_api_threads
                ORDER BY updated_at DESC, thread_id ASC LIMIT %s OFFSET %s
            """, (limit, offset))).fetchall()
            return rows, total["total"]

    async def touch(self, thread_id, question=None):
        async with self.pool.connection() as conn:
            await conn.execute("""
                UPDATE handbook_api_threads SET updated_at = now(),
                    title = CASE WHEN title = 'Новый диалог' AND %s::text IS NOT NULL THEN left(%s::text, 120) ELSE title END
                WHERE thread_id = %s
            """, (question, question, thread_id))

    async def completed_turns(self, thread_id, limit, offset):
        # Only save_turn updates history. Read those checkpoints, not every large
        # intermediate state; this recovers full messages even after history trimming.
        predicate = """thread_id = %s AND checkpoint_ns = ''
            AND metadata->>'source' = 'loop'
            AND (checkpoint->'updated_channels') ? 'history'
            AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(checkpoint->'updated_channels') AS channel
                WHERE channel LIKE 'branch:to:%%'
            )"""
        async with self.pool.connection() as conn:
            total = await (await conn.execute(
                f"SELECT count(*) AS total FROM checkpoints WHERE {predicate}", (thread_id,))).fetchone()
            rows = await (await conn.execute(
                f"SELECT checkpoint_id FROM checkpoints WHERE {predicate} ORDER BY checkpoint_id ASC LIMIT %s OFFSET %s",
                (thread_id, limit, offset))).fetchall()
            return [row["checkpoint_id"] for row in rows], total["total"]

    @asynccontextmanager
    async def thread_lock(self, thread_id):
        # Transaction scope releases the lock even on cancellation/connection loss.
        # Graph checkpoints use other pool connections and commit independently.
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS acquired",
                    ("handbook-api:" + thread_id,))).fetchone()
                yield row["acquired"]

    async def ping(self):
        async with self.pool.connection() as conn:
            await conn.execute("SELECT 1")
