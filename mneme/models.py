"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    workspace_id: str | None = None
    session_id: str | None = None  # caller-specified ID; auto-UUID when absent


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageIn(BaseModel):
    role: str
    content: str


class AppendMessagesRequest(BaseModel):
    messages: list[MessageIn]


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str


class SessionResponse(BaseModel):
    session_id: str
    workspace_id: str | None
    created_at: str
    messages: list[MessageOut]


class SessionSummary(BaseModel):
    session_id: str
    preview: str
    created_at: str
    message_count: int


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


class RememberRequest(BaseModel):
    workspace_id: str | None = None
    content: str
    source: str | None = None
    tags: list[str] | None = None


class RememberResponse(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str
    workspace_id: str | None = None
    k: int = 5


class RecallResult(BaseModel):
    id: str
    content: str
    source: str | None
    tags: list[str]
    score: float


class RecallResponse(BaseModel):
    results: list[RecallResult]


# ---------------------------------------------------------------------------
# Message recall
# ---------------------------------------------------------------------------


class RecallMessagesRequest(BaseModel):
    query: str
    workspace_id: str | None = None
    session_id: str | None = None
    k: int = 5


class MessageRecallResult(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: str
    score: float


class RecallMessagesResponse(BaseModel):
    results: list[MessageRecallResult]


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


class PrefIn(BaseModel):
    workspace_id: str
    suggestion_id: str | None = None
    action: str  # 'accepted', 'ignored', 'rejected'
    # Phase 3 additions — feedback loop signals
    task_summary: str | None = None
    feedback_text: str | None = None
    # Phase 4 additions — richer embeddings for semantic retrieval
    task_type: str | None = None  # e.g. "bugfix", "feature", "refactor"
    approach_notes: str | None = None  # how the task was approached


class PrefOut(BaseModel):
    id: str
    suggestion_id: str | None
    action: str
    created_at: str
    # Phase 3 additions
    task_summary: str | None = None
    feedback_text: str | None = None
    # Phase 4 additions
    task_type: str | None = None
    approach_notes: str | None = None
