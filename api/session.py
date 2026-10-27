"""In-memory session store for authenticated API clients."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from meta_analyst.providers import ProviderStore, UsageMeter, seed_store_from_env


@dataclass
class Session:
    token: str
    created_at: float = field(default_factory=time.time)
    df: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    meta_accounts: list[dict[str, Any]] = field(default_factory=list)
    provider_store: ProviderStore = field(default_factory=seed_store_from_env)
    usage: UsageMeter = field(default_factory=UsageMeter)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    chart_insights: dict[str, str | None] = field(default_factory=dict)
    detailed_analysis: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        token = secrets.token_urlsafe(32)
        session = Session(token=token)
        with self._lock:
            self._sessions[token] = session
        return session

    def get(self, token: str | None) -> Session | None:
        if not token:
            return None
        with self._lock:
            return self._sessions.get(token)

    def delete(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


store = SessionStore()
