"""Tests for Mneme API routes.

The store is mocked so tests run without Postgres or API keys.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mneme.app import app

client = TestClient(app)

_STORE = "mneme.app._store"


def _mock_store() -> AsyncMock:
    store = AsyncMock()
    store.init_db = AsyncMock()
    return store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_create_session() -> None:
    store = _mock_store()
    store.create_session = AsyncMock(return_value="sid-1")
    with patch(_STORE, store):
        r = client.post("/sessions", json={"workspace_id": "ws-1"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "sid-1"


def test_get_session() -> None:
    store = _mock_store()
    store.get_session = AsyncMock(
        return_value={
            "session_id": "sid-1",
            "workspace_id": "ws-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "messages": [
                {"role": "user", "content": "hello", "created_at": "2026-01-01T00:00:01+00:00"}
            ],
        }
    )
    with patch(_STORE, store):
        r = client.get("/sessions/sid-1")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sid-1"
    assert len(body["messages"]) == 1


def test_get_session_not_found() -> None:
    store = _mock_store()
    store.get_session = AsyncMock(return_value=None)
    with patch(_STORE, store):
        r = client.get("/sessions/nonexistent")
    assert r.status_code == 404


def test_append_messages() -> None:
    store = _mock_store()
    store.get_session = AsyncMock(
        return_value={"session_id": "sid-1", "workspace_id": None, "created_at": "", "messages": []}
    )
    store.append_messages = AsyncMock()
    with patch(_STORE, store):
        r = client.post(
            "/sessions/sid-1/messages",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 204


def test_list_sessions() -> None:
    store = _mock_store()
    store.list_sessions = AsyncMock(
        return_value=[
            {
                "session_id": "sid-1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "preview": "hello",
                "message_count": 2,
            }
        ]
    )
    with patch(_STORE, store):
        r = client.get("/sessions?workspace_id=ws-1")
    assert r.status_code == 200
    assert len(r.json()) == 1


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def test_remember() -> None:
    store = _mock_store()
    store.remember = AsyncMock(return_value="fact-1")
    with (
        patch(_STORE, store),
        patch("mneme.app.embeddings.embed", new=AsyncMock(return_value=[[0.1] * 1536])),
    ):
        r = client.post(
            "/remember",
            json={"content": "Python uses indentation", "workspace_id": "ws-1"},
        )
    assert r.status_code == 200
    assert r.json()["id"] == "fact-1"


def test_recall() -> None:
    store = _mock_store()
    store.recall = AsyncMock(
        return_value=[
            {
                "id": "fact-1",
                "content": "Python uses indentation",
                "source": None,
                "tags": [],
                "score": 0.95,
            }
        ]
    )
    with (
        patch(_STORE, store),
        patch("mneme.app.embeddings.embed", new=AsyncMock(return_value=[[0.1] * 1536])),
    ):
        r = client.post(
            "/recall",
            json={"query": "Python syntax", "workspace_id": "ws-1", "k": 3},
        )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_record_pref() -> None:
    store = _mock_store()
    store.record_pref = AsyncMock(return_value="pref-1")
    with patch(_STORE, store):
        r = client.post(
            "/prefs",
            json={"workspace_id": "ws-1", "action": "accepted", "suggestion_id": "s-1"},
        )
    assert r.status_code == 201
    assert r.json()["id"] == "pref-1"


def test_get_prefs() -> None:
    store = _mock_store()
    store.list_prefs = AsyncMock(
        return_value=[
            {
                "id": "pref-1",
                "suggestion_id": "s-1",
                "action": "accepted",
                "created_at": "2026-01-01T00:00:00+00:00",
                "task_summary": None,
                "feedback_text": None,
            }
        ]
    )
    with patch(_STORE, store):
        r = client.get("/prefs/ws-1")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_record_pref_with_task_summary() -> None:
    """POST /prefs with Phase 3 task_summary + feedback_text fields."""
    store = _mock_store()
    store.record_pref = AsyncMock(return_value="pref-2")
    with patch(_STORE, store):
        r = client.post(
            "/prefs",
            json={
                "workspace_id": "ws-1",
                "action": "rejected",
                "task_summary": "refactored login flow",
                "feedback_text": "broke existing tests",
            },
        )
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "pref-2"
    assert body["task_summary"] == "refactored login flow"
    assert body["feedback_text"] == "broke existing tests"
    # Verify store received the new fields
    call_kwargs = store.record_pref.call_args.kwargs
    assert call_kwargs["task_summary"] == "refactored login flow"
    assert call_kwargs["feedback_text"] == "broke existing tests"


def test_get_prefs_returns_new_fields() -> None:
    """GET /prefs/:workspace_id returns task_summary and feedback_text."""
    store = _mock_store()
    store.list_prefs = AsyncMock(
        return_value=[
            {
                "id": "pref-3",
                "suggestion_id": None,
                "action": "accepted",
                "created_at": "2026-01-01T00:00:00+00:00",
                "task_summary": "added auth middleware",
                "feedback_text": None,
            }
        ]
    )
    with patch(_STORE, store):
        r = client.get("/prefs/ws-1")
    assert r.status_code == 200
    item = r.json()[0]
    assert item["task_summary"] == "added auth middleware"
    assert item["feedback_text"] is None
