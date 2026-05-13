"""
Feedback store and adaptive behaviour for LenaDena BankBot.
Stores user thumbs-up/down per query pattern.
If a pattern gets >= 2 negative ratings, the agent adds a disclaimer + escalation offer.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = LOG_DIR / "feedback_store.json"

NEGATIVE_THRESHOLD = 2  # thumbs-down to trigger behaviour change


def _load() -> dict:
    if FEEDBACK_FILE.exists():
        try:
            return json.loads(FEEDBACK_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(store: dict):
    FEEDBACK_FILE.write_text(json.dumps(store, indent=2))


def _topic_key(query: str) -> str:
    """Extract a normalised topic key from a query for grouping feedback."""
    q = query.lower().strip()
    topic_patterns = [
        (r"\b(fd|fixed deposit)\b", "fd"),
        (r"\b(credit card)\b", "credit_card"),
        (r"\b(personal loan)\b", "personal_loan"),
        (r"\b(home loan)\b", "home_loan"),
        (r"\b(savings account|savings)\b", "savings"),
        (r"\b(dispute|wrong charge|incorrect transaction)\b", "dispute"),
        (r"\b(eligibility|eligible)\b", "eligibility"),
        (r"\b(kyc|documents)\b", "kyc"),
        (r"\b(minimum balance)\b", "min_balance"),
    ]
    for pattern, key in topic_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return key
    return "general"


def record_feedback(
    query: str,
    response_preview: str,
    rating: Literal["thumbs_up", "thumbs_down"],
    session_id: str,
):
    """Record a feedback event."""
    store = _load()
    topic = _topic_key(query)

    if topic not in store:
        store[topic] = {
            "thumbs_up": 0,
            "thumbs_down": 0,
            "events": [],
        }

    store[topic][rating] += 1
    store[topic]["events"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "rating": rating,
        "response_preview": response_preview[:100],
    })

    _save(store)
    return topic


def get_low_confidence_topics() -> set[str]:
    """Return topics with >= NEGATIVE_THRESHOLD thumbs-down."""
    store = _load()
    return {
        topic for topic, data in store.items()
        if data.get("thumbs_down", 0) >= NEGATIVE_THRESHOLD
    }


def is_low_confidence_query(query: str) -> bool:
    """True if this query's topic has accumulated enough negative feedback."""
    topic = _topic_key(query)
    return topic in get_low_confidence_topics()


DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Note:** Some customers have found answers on this topic unclear. "
    "For the most accurate and personalised information, I recommend speaking "
    "with a LenaDena Bank advisor: 📞 1800-123-5362 or visit your nearest branch."
)


def apply_adaptive_behaviour(response: str, query: str) -> str:
    """Append disclaimer if this topic has accumulated negative feedback."""
    if is_low_confidence_query(query):
        return response + DISCLAIMER
    return response


def get_feedback_summary() -> dict:
    """Return current feedback stats."""
    store = _load()
    summary = {}
    for topic, data in store.items():
        total = data["thumbs_up"] + data["thumbs_down"]
        summary[topic] = {
            "total_ratings": total,
            "thumbs_up": data["thumbs_up"],
            "thumbs_down": data["thumbs_down"],
            "satisfaction_rate": round(data["thumbs_up"] / total * 100, 1) if total > 0 else None,
            "adaptive_mode": data.get("thumbs_down", 0) >= NEGATIVE_THRESHOLD,
        }
    return summary
