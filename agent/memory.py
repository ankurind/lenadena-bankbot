"""
Memory management for LenaDena BankBot.
Short-term: last k turns in conversation (per session).
Long-term: user preferences persisted to JSON across sessions.
"""

import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
PREFS_FILE = LOG_DIR / "user_preferences.json"

SHORT_TERM_WINDOW = 6  # number of turns to keep in memory


class ShortTermMemory:
    """Sliding window conversation memory."""

    def __init__(self, k: int = SHORT_TERM_WINDOW):
        self.k = k
        self.history: list[dict] = []  # [{"role": "user"|"assistant", "content": str}]

    def add_user(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        # Keep at most k*2 messages (k user + k assistant)
        if len(self.history) > self.k * 2:
            self.history = self.history[-(self.k * 2):]

    def get_history(self) -> list[dict]:
        return list(self.history)

    def format_for_prompt(self) -> str:
        """Return history as a formatted string for injection into prompts."""
        if not self.history:
            return ""
        lines = ["Conversation so far:"]
        for msg in self.history:
            role = "Customer" if msg["role"] == "user" else "BankBot"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def reset(self):
        self.history = []


class LongTermMemory:
    """
    Persists user preferences across sessions.
    Keyed by session_id (anonymous — no PII stored).
    """

    def __init__(self):
        self._store: dict = self._load()

    def _load(self) -> dict:
        if PREFS_FILE.exists():
            try:
                return json.loads(PREFS_FILE.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self):
        PREFS_FILE.write_text(json.dumps(self._store, indent=2))

    def set_preference(self, session_id: str, key: str, value: str):
        if session_id not in self._store:
            self._store[session_id] = {"preferences": {}, "updated_at": None}
        self._store[session_id]["preferences"][key] = value
        self._store[session_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_preferences(self, session_id: str) -> dict:
        return self._store.get(session_id, {}).get("preferences", {})

    def get_preference(self, session_id: str, key: str) -> Optional[str]:
        return self.get_preferences(session_id).get(key)

    def format_for_prompt(self, session_id: str) -> str:
        prefs = self.get_preferences(session_id)
        if not prefs:
            return ""
        lines = ["Known customer preferences from prior sessions:"]
        for k, v in prefs.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


# Module-level singletons
_long_term = LongTermMemory()


def get_long_term_memory() -> LongTermMemory:
    return _long_term
