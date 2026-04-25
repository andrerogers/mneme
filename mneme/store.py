"""Async PostgreSQL store for Mneme — sessions, messages, facts, preferences.

Uses psycopg3 async with a connection pool (psycopg-pool). Schema lives in
the 'mneme' Postgres schema. The pool handles reconnection automatically.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

_MAX_HISTORY = 100
_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

_T = TypeVar("_T")


class Store:
    """Async Postgres store for the Mneme service."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                from psycopg_pool import AsyncConnectionPool

                log.info("mneme: opening PostgreSQL connection pool")
                pool = AsyncConnectionPool(
                    self._dsn,
                    min_size=2,
                    max_size=10,
                    open=False,
                    kwargs={"autocommit": True},
                )
                await pool.open()
                self._pool = pool
        return self._pool

    async def _run(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Call fn(conn) with a connection from the pool."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            return await fn(conn)

    async def close(self) -> None:
        """Shut down the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def init_db(self) -> None:
        """Run yoyo migrations in a thread executor."""
        dsn = self._dsn
        migrations_dir = str(_MIGRATIONS_DIR)

        def _migrate() -> None:
            from yoyo import get_backend, read_migrations

            yoyo_dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
            backend = get_backend(yoyo_dsn, migration_table="_mneme_yoyo_migrations")
            # Isolate log + version tables per-service so they don't collide
            # when Hive/Mneme/Engram share the same Postgres database.
            backend.log_table = "_mneme_yoyo_log"
            backend.version_table = "_mneme_yoyo_version"
            migrations = read_migrations(migrations_dir)
            with backend.lock():
                backend.apply_migrations(backend.to_apply(migrations))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _migrate)
        log.info("mneme: migrations applied")

    # ── Sessions ──────────────────────────────────────────────────────────

    async def create_session(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                # Upsert: if a caller-supplied session_id already exists with a
                # NULL workspace_id, fill it in; never overwrite an existing
                # workspace_id to avoid silent moves between workspaces.
                await conn.execute(
                    "INSERT INTO mneme.sessions (id, workspace_id) VALUES (%s, %s)"
                    " ON CONFLICT (id) DO UPDATE"
                    "   SET workspace_id = EXCLUDED.workspace_id"
                    "   WHERE mneme.sessions.workspace_id IS NULL"
                    "     AND EXCLUDED.workspace_id IS NOT NULL",
                    (sid, workspace_id),
                )
            return sid

        return await self._run(_do)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async def _do(conn: Any) -> dict[str, Any] | None:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        "SELECT id, workspace_id, created_at FROM mneme.sessions WHERE id = %s",
                        (session_id,),
                    )
                ).fetchone()
                if row is None:
                    return None
                messages = await (
                    await conn.execute(
                        "SELECT role, content, created_at FROM mneme.messages "
                        "WHERE session_id = %s ORDER BY created_at LIMIT %s",
                        (session_id, _MAX_HISTORY),
                    )
                ).fetchall()
            return {
                "session_id": row[0],
                "workspace_id": row[1],
                "created_at": row[2].isoformat(),
                "messages": [
                    {"role": m[0], "content": m[1], "created_at": m[2].isoformat()}
                    for m in messages
                ],
            }

        return await self._run(_do)

    async def append_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        embeddings: list[list[float] | None] | None = None,
    ) -> None:
        async def _do(conn: Any) -> None:
            async with conn.transaction():
                for i, msg in enumerate(messages):
                    emb = embeddings[i] if embeddings and i < len(embeddings) else None
                    if emb:
                        await conn.execute(
                            "INSERT INTO mneme.messages "
                            "(session_id, role, content, embedding) "
                            "VALUES (%s, %s, %s, %s::vector)",
                            (session_id, msg["role"], msg["content"], str(emb)),
                        )
                    else:
                        await conn.execute(
                            "INSERT INTO mneme.messages "
                            "(session_id, role, content) VALUES (%s, %s, %s)",
                            (session_id, msg["role"], msg["content"]),
                        )

        await self._run(_do)

    async def replace_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """Delete all messages for *session_id* and insert *messages* atomically."""

        async def _do(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM mneme.messages WHERE session_id = %s", (session_id,)
                )
                for msg in messages:
                    await conn.execute(
                        "INSERT INTO mneme.messages (session_id, role, content) VALUES (%s, %s, %s)",
                        (session_id, msg["role"], msg["content"]),
                    )

        await self._run(_do)

    async def delete_session(self, session_id: str) -> bool:
        async def _do(conn: Any) -> bool:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM mneme.sessions WHERE id = %s",
                    (session_id,),
                )
            return result.rowcount > 0

        return await self._run(_do)

    async def list_sessions(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            if workspace_id:
                rows = await (
                    await conn.execute(
                        "SELECT s.id, s.created_at, "
                        "(SELECT content FROM mneme.messages m WHERE m.session_id = s.id "
                        " ORDER BY m.created_at LIMIT 1), "
                        "(SELECT count(*) FROM mneme.messages m WHERE m.session_id = s.id) "
                        "FROM mneme.sessions s "
                        "WHERE s.workspace_id = %s AND s.archived = false "
                        "ORDER BY s.created_at DESC LIMIT 50",
                        (workspace_id,),
                    )
                ).fetchall()
            else:
                rows = await (
                    await conn.execute(
                        "SELECT s.id, s.created_at, "
                        "(SELECT content FROM mneme.messages m WHERE m.session_id = s.id "
                        " ORDER BY m.created_at LIMIT 1), "
                        "(SELECT count(*) FROM mneme.messages m WHERE m.session_id = s.id) "
                        "FROM mneme.sessions s "
                        "WHERE s.archived = false "
                        "ORDER BY s.created_at DESC LIMIT 50",
                    )
                ).fetchall()
            return [
                {
                    "session_id": r[0],
                    "created_at": r[1].isoformat(),
                    "preview": (r[2] or "")[:50],
                    "message_count": r[3],
                }
                for r in rows
            ]

        return await self._run(_do)

    # ── Facts ─────────────────────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        workspace_id: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        fid = str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                if embedding:
                    await conn.execute(
                        "INSERT INTO mneme.facts "
                        "(id, workspace_id, content, source, tags, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                        (fid, workspace_id, content, source, tags or [], str(embedding)),
                    )
                else:
                    await conn.execute(
                        "INSERT INTO mneme.facts (id, workspace_id, content, source, tags) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (fid, workspace_id, content, source, tags or []),
                    )
            return fid

        return await self._run(_do)

    async def recall(
        self,
        embedding: list[float],
        workspace_id: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            vec_str = str(embedding)
            if workspace_id:
                rows = await (
                    await conn.execute(
                        "SELECT id, content, source, tags, "
                        "1 - (embedding <=> %s::vector) AS score "
                        "FROM mneme.facts "
                        "WHERE workspace_id = %s AND embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s::vector LIMIT %s",
                        (vec_str, workspace_id, vec_str, k),
                    )
                ).fetchall()
            else:
                rows = await (
                    await conn.execute(
                        "SELECT id, content, source, tags, "
                        "1 - (embedding <=> %s::vector) AS score "
                        "FROM mneme.facts "
                        "WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s::vector LIMIT %s",
                        (vec_str, vec_str, k),
                    )
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "source": r[2],
                    "tags": r[3] or [],
                    "score": float(r[4]),
                }
                for r in rows
            ]

        return await self._run(_do)

    # ── Preferences ───────────────────────────────────────────────────────

    async def record_pref(
        self,
        workspace_id: str,
        action: str,
        suggestion_id: str | None = None,
        task_summary: str | None = None,
        feedback_text: str | None = None,
        task_type: str | None = None,
        approach_notes: str | None = None,
    ) -> str:
        pid = str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO mneme.preferences "
                    "(id, workspace_id, suggestion_id, action, task_summary, feedback_text, "
                    "task_type, approach_notes) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        pid,
                        workspace_id,
                        suggestion_id,
                        action,
                        task_summary,
                        feedback_text,
                        task_type,
                        approach_notes,
                    ),
                )
            return pid

        return await self._run(_do)

    async def list_prefs(self, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            rows = await (
                await conn.execute(
                    "SELECT id, suggestion_id, action, created_at, task_summary, feedback_text, "
                    "task_type, approach_notes "
                    "FROM mneme.preferences "
                    "WHERE workspace_id = %s ORDER BY created_at DESC LIMIT %s",
                    (workspace_id, limit),
                )
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "suggestion_id": r[1],
                    "action": r[2],
                    "created_at": r[3].isoformat(),
                    "task_summary": r[4],
                    "feedback_text": r[5],
                    "task_type": r[6],
                    "approach_notes": r[7],
                }
                for r in rows
            ]

        return await self._run(_do)

    # ── Message recall ───────────────────────────────────────────────────

    async def recall_messages(
        self,
        embedding: list[float],
        workspace_id: str | None = None,
        session_id: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            vec_str = str(embedding)
            if session_id:
                rows = await (
                    await conn.execute(
                        "SELECT m.id, m.session_id, m.role, m.content, m.created_at, "
                        "1 - (m.embedding <=> %s::vector) AS score "
                        "FROM mneme.messages m "
                        "WHERE m.session_id = %s AND m.embedding IS NOT NULL "
                        "ORDER BY m.embedding <=> %s::vector LIMIT %s",
                        (vec_str, session_id, vec_str, k),
                    )
                ).fetchall()
            elif workspace_id:
                rows = await (
                    await conn.execute(
                        "SELECT m.id, m.session_id, m.role, m.content, m.created_at, "
                        "1 - (m.embedding <=> %s::vector) AS score "
                        "FROM mneme.messages m "
                        "JOIN mneme.sessions s ON s.id = m.session_id "
                        "WHERE s.workspace_id = %s AND m.embedding IS NOT NULL "
                        "ORDER BY m.embedding <=> %s::vector LIMIT %s",
                        (vec_str, workspace_id, vec_str, k),
                    )
                ).fetchall()
            else:
                rows = await (
                    await conn.execute(
                        "SELECT m.id, m.session_id, m.role, m.content, m.created_at, "
                        "1 - (m.embedding <=> %s::vector) AS score "
                        "FROM mneme.messages m "
                        "WHERE m.embedding IS NOT NULL "
                        "ORDER BY m.embedding <=> %s::vector LIMIT %s",
                        (vec_str, vec_str, k),
                    )
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "session_id": r[1],
                    "role": r[2],
                    "content": r[3],
                    "created_at": r[4].isoformat(),
                    "score": float(r[5]),
                }
                for r in rows
            ]

        return await self._run(_do)
