"""
Phase 7: Adaptive Behaviour
LenaDena BankBot

Goal: Introduce feedback signals (thumbs-up/down), store per topic, modify behaviour
      when a topic accumulates negative feedback, show before/after comparison.
Run: python phases/phase7_adaptive.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agent.graph import run_agent
from agent.memory import ShortTermMemory
from agent.feedback import (
    record_feedback, is_low_confidence_query, apply_adaptive_behaviour,
    get_low_confidence_topics, get_feedback_summary,
)
from agent.logging_utils import new_session_id

DIVIDER = "=" * 70

# Reset feedback store for clean demo
FEEDBACK_FILE = Path(__file__).parent.parent / "logs" / "feedback_store.json"


def main():
    print(DIVIDER)
    print("PHASE 7: ADAPTIVE BEHAVIOUR — LenaDena BankBot")
    print(DIVIDER)

    # Clear for clean demo
    if FEEDBACK_FILE.exists():
        FEEDBACK_FILE.unlink()
        print("Cleared previous feedback store for clean demo.\n")

    print("""
Adaptive Behaviour Logic:
  - Each response can receive thumbs-up (good) or thumbs-down (bad) feedback
  - Feedback stored in logs/feedback_store.json keyed by topic
  - Topics: fd, credit_card, personal_loan, savings, dispute, eligibility, etc.
  - When a topic reaches 2+ thumbs-down → enters "low confidence" mode
  - Low confidence mode: appends a disclaimer + human advisor referral to responses
""")

    session_id = new_session_id()
    memory = ShortTermMemory(k=6)
    query = "Am I eligible for a personal loan at LenaDena Bank?"

    # --- BEFORE: no negative feedback ---
    print(f"{DIVIDER}")
    print("[1/4] BEFORE — No negative feedback yet")
    print(DIVIDER)
    result = run_agent(query, session_id, memory)
    base_response = result["final_response"]
    adapted = apply_adaptive_behaviour(base_response, query)
    print(f"Query: {query}")
    print(f"\nResponse:\n{adapted}")
    print(f"\nDisclaimer added: {'Yes' if adapted != base_response else 'No'}")

    # --- Simulate 2 negative feedback events ---
    print(f"\n{DIVIDER}")
    print("[2/4] Simulating 2 thumbs-down on 'personal_loan' topic")
    print(DIVIDER)
    record_feedback(
        query="Am I eligible for a personal loan?",
        response_preview="General eligibility: CIBIL 700, min income ₹25,000/month...",
        rating="thumbs_down",
        session_id="user_A",
    )
    record_feedback(
        query="What is the personal loan interest rate?",
        response_preview="Personal Loan: 10.5%-18% p.a., tenure 12-60 months...",
        rating="thumbs_down",
        session_id="user_B",
    )
    print(f"Feedback recorded.")
    print(f"Low confidence topics now: {get_low_confidence_topics()}")
    print(f"Is personal loan query low confidence? {is_low_confidence_query(query)}")

    # --- AFTER: disclaimer appended ---
    print(f"\n{DIVIDER}")
    print("[3/4] AFTER — 2 thumbs-down → disclaimer appended")
    print(DIVIDER)
    result_after = run_agent(query, session_id, memory)
    base_after = result_after["final_response"]
    adapted_after = apply_adaptive_behaviour(base_after, query)
    print(f"Query: {query}")
    print(f"\nResponse:\n{adapted_after}")
    print(f"\nDisclaimer added: {'Yes' if adapted_after != base_after else 'No'}")

    print(f"\n{DIVIDER}")
    print("[4/4] Feedback Summary")
    print(DIVIDER)
    # Add some positive feedback for contrast
    record_feedback("What are the FD interest rates?", "FD rates: 1 year 6.80%...", "thumbs_up", "user_C")
    record_feedback("How do I dispute a transaction?", "Call 1800-123-5362...", "thumbs_up", "user_D")

    summary = get_feedback_summary()
    print(f"{'Topic':<20} {'👍':<6} {'👎':<6} {'Mode'}")
    print("-" * 50)
    for topic, stats in summary.items():
        mode = "⚠️  ADAPTIVE (disclaimer)" if stats["adaptive_mode"] else "✅ Normal"
        print(f"{topic:<20} {stats['thumbs_up']:<6} {stats['thumbs_down']:<6} {mode}")

    print(f"\n{DIVIDER}")
    print("BEFORE vs AFTER COMPARISON TABLE")
    print(DIVIDER)
    print("""
| Aspect                 | Before (0 thumbs-down)   | After (2 thumbs-down)             |
|------------------------|--------------------------|-----------------------------------|
| Core response          | Eligibility information  | Same eligibility information      |
| Disclaimer appended    | No                       | Yes                               |
| Escalation suggestion  | No                       | Yes — "call 1800-123-5362"        |
| Behaviour trigger      | None                     | Topic has ≥2 negative ratings     |
| Storage                | N/A                      | logs/feedback_store.json          |
| Persistence            | N/A                      | Persists across sessions          |

What changed: the agent appends a disclaimer for any query under the 'personal_loan' topic.
Why this approach: transparent to the user — they see this topic has been unclear for others,
which increases trust rather than silently changing tone.
""")


if __name__ == "__main__":
    main()
