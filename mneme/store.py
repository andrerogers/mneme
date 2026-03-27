"""Async PostgreSQL store for Mneme — sessions, messages, facts, preferences.

Uses psycopg3 async with a lazy singleton connection (same pattern as
hive/memory/store.py). Schema lives in the 'mneme' Postgres schema.

Reconnect: each public method calls _run(fn) which catches OperationalError /
InterfaceError, resets the connection singleton, and retries once — surviving
Postgres restarts without requiring a service restart.
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
        self._conn: object | None = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> Any:
        import psycopg

        async with self._lock:
            if self._conn is None or getattr(self._conn, "closed", True):
                log.info("mneme: opening PostgreSQL connection")
                self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def _run(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Call fn(conn) with one reconnect retry on transient connection errors."""
        import psycopg

        try:
            conn = await self._get_conn()
            return await fn(conn)
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            log.warning("mneme: connection lost (%s) — reconnecting", exc)
            async with self._lock:
                self._conn = None
            conn = await self._get_conn()
            return await fn(conn)

    async def init_db(self) -> None:
        """Run yoyo migrations in a thread executor."""
        dsn = self._dsn
        migrations_dir = str(_MIGRATIONS_DIR)

        def _migrate() -> None:
            from yoyo import get_backend, read_migrations

            yoyo_dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
            backend = get_backend(yoyo_dsn)
            migrations = read_migrations(migrations_dir)
            with backend.lock():
                backend.apply_migrations(backend.to_apply(migrations))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _migrate)
        log.info("mneme: migrations applied")

    # ── Sessions ──────────────────────────────────────────────────────────

    async def create_session(self, workspace_id: str | None = None) -> str:
        sid = str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO mneme.sessions (id, workspace_id) VALUES (%s, %s)",
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

    async def append_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        async def _do(conn: Any) -> None:
            async with conn.transaction():
                for msg in messages:
                    await conn.execute(
                        "INSERT INTO mneme.messages (session_id, role, content) VALUES (%s, %s, %s)",
                        (session_id, msg["role"], msg["content"]),
                    )

        await self._run(_do)

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
    ) -> str:
        pid = str(uuid.uuid4())

        async def _do(conn: Any) -> str:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO mneme.preferences (id, workspace_id, suggestion_id, action) "
                    "VALUES (%s, %s, %s, %s)",
                    (pid, workspace_id, suggestion_id, action),
                )
            return pid

        return await self._run(_do)

    async def list_prefs(self, workspace_id: str) -> list[dict[str, Any]]:
        async def _do(conn: Any) -> list[dict[str, Any]]:
            rows = await (
                await conn.execute(
                    "SELECT id, suggestion_id, action, created_at "
                    "FROM mneme.preferences "
                    "WHERE workspace_id = %s ORDER BY created_at DESC LIMIT 100",
                    (workspace_id,),
                )
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "suggestion_id": r[1],
                    "action": r[2],
                    "created_at": r[3].isoformat(),
                }
                for r in rows
            ]

        return await self._run(_do)
