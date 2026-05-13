"""
PII-safe structured logger for LenaDena BankBot.
All query text is SHA-256 hashed before writing to logs.
"""

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.safety import scrub_pii

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_interaction(
    session_id: str,
    query: str,
    intent: str,
    safety_verdict: str,
    tools_used: list[str],
    response_preview: str,
    latency_ms: int,
    status: str = "ok",
    langsmith_trace_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Write one interaction record to the JSONL log. Returns the record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query_hash": _hash_text(query),
        "intent": intent,
        "safety_verdict": safety_verdict,
        "tools_used": tools_used,
        "response_preview": scrub_pii(response_preview[:120]),
        "latency_ms": latency_ms,
        "status": status,
        "langsmith_trace_url": langsmith_trace_url,
    }
    if extra:
        record.update(extra)

    with open(INTERACTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def read_logs(n: int = 50) -> list[dict]:
    """Return last n log records."""
    if not INTERACTION_LOG.exists():
        return []
    lines = INTERACTION_LOG.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]


class Timer:
    """Context manager that measures elapsed ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
