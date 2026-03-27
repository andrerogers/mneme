# ruff: noqa: E402
from optics import instrument_fastapi, setup_optics

setup_optics("mneme", service_version="0.0.1")

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

log = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from mneme import embeddings
from mneme.config import DATABASE_URL
from mneme.models import (
    AppendMessagesRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    PrefIn,
    PrefOut,
    RecallRequest,
    RecallResponse,
    RecallResult,
    RememberRequest,
    RememberResponse,
    SessionResponse,
    SessionSummary,
)
from mneme.store import Store

_store: Store | None = None


def _get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialised")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store  # noqa: PLW0603
    if DATABASE_URL:
        _store = Store(DATABASE_URL)
        await _store.init_db()
    yield


app = FastAPI(title="Mneme", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

instrument_fastapi(app)


# ── Health ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "mneme"}


# ── Sessions ──────────────────────────────────────────────────────────────


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest | None = None) -> CreateSessionResponse:
    store = _get_store()
    workspace_id = req.workspace_id if req else None
    sid = await store.create_session(workspace_id)
    return CreateSessionResponse(session_id=sid)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    store = _get_store()
    data = await store.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=data["session_id"],
        workspace_id=data["workspace_id"],
        created_at=data["created_at"],
        messages=data["messages"],
    )


@app.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    workspace_id: str | None = Query(default=None),
) -> list[SessionSummary]:
    store = _get_store()
    rows = await store.list_sessions(workspace_id)
    return [SessionSummary(**r) for r in rows]


@app.post("/sessions/{session_id}/messages", status_code=204)
async def append_messages(session_id: str, req: AppendMessagesRequest) -> None:
    store = _get_store()
    data = await store.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await store.append_messages(
        session_id, [{"role": m.role, "content": m.content} for m in req.messages]
    )


# ── Facts ─────────────────────────────────────────────────────────────────


@app.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest) -> RememberResponse:
    store = _get_store()
    embedding = None
    try:
        vecs = await embeddings.embed([req.content])
        embedding = vecs[0]
    except Exception as exc:
        log.warning("remember: embedding failed — storing fact without vector (%s)", exc)
    fid = await store.remember(
        content=req.content,
        workspace_id=req.workspace_id,
        source=req.source,
        tags=req.tags,
        embedding=embedding,
    )
    return RememberResponse(id=fid)


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest) -> RecallResponse:
    store = _get_store()
    vecs = await embeddings.embed([req.query])
    results = await store.recall(
        embedding=vecs[0],
        workspace_id=req.workspace_id,
        k=req.k,
    )
    return RecallResponse(results=[RecallResult(**r) for r in results])


# ── Preferences ───────────────────────────────────────────────────────────


@app.get("/prefs/{workspace_id}", response_model=list[PrefOut])
async def get_prefs(workspace_id: str) -> list[PrefOut]:
    store = _get_store()
    rows = await store.list_prefs(workspace_id)
    return [PrefOut(**r) for r in rows]


@app.post("/prefs", status_code=201, response_model=PrefOut)
async def record_pref(req: PrefIn) -> PrefOut:
    store = _get_store()
    pid = await store.record_pref(
        workspace_id=req.workspace_id,
        action=req.action,
        suggestion_id=req.suggestion_id,
    )
    return PrefOut(
        id=pid,
        suggestion_id=req.suggestion_id,
        action=req.action,
        created_at="",  # client can ignore; real value is in DB
    )
