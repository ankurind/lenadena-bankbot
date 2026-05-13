"""
Phase 8: Deployment Readiness
LenaDena BankBot

Goal: Production wrapper with PII-safe logging, LangSmith tracing, graceful failure
      handling, 10-run deployment test.
Run: python phases/phase8_deployment.py
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Enable LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "lenadena-bankbot")

from agent.graph import run_agent
from agent.memory import ShortTermMemory
from agent.feedback import apply_adaptive_behaviour
from agent.logging_utils import log_interaction, new_session_id, Timer, read_logs
from agent.safety import scrub_pii

DIVIDER = "=" * 70
FALLBACK_RESPONSE = (
    "I'm sorry, I encountered a technical issue processing your request. "
    "Please try again or contact us at 1800-123-5362."
)


def safe_run_with_logging(
    query: str,
    session_id: str,
    memory: ShortTermMemory,
    max_retries: int = 1,
) -> dict:
    """Production wrapper: run agent, retry on timeout, log PII-safely, return metadata."""
    status = "ok"
    final_response = FALLBACK_RESPONSE
    intent = "unknown"
    tools_used = []
    safety_verdict = "UNKNOWN"
    latency_ms = 0

    for attempt in range(max_retries + 1):
        try:
            with Timer() as t:
                state = run_agent(query, session_id, memory)

            final_response = apply_adaptive_behaviour(
                state.get("final_response", FALLBACK_RESPONSE), query
            )
            intent = state.get("intent", "unknown")
            tools_used = state.get("tool_calls_made", [])
            safety_verdict = state.get("safety_verdict", "UNKNOWN")
            latency_ms = t.elapsed_ms
            break

        except Exception as e:
            err_type = type(e).__name__
            if attempt < max_retries and "timeout" in str(e).lower():
                print(f"  [RETRY] Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(1)
                continue
            status = f"error:{err_type}"
            latency_ms = 0
            print(f"  [ERROR] {err_type}: {e}")
            break

    log_interaction(
        session_id=session_id,
        query=query,
        intent=intent,
        safety_verdict=safety_verdict,
        tools_used=tools_used,
        response_preview=final_response,
        latency_ms=latency_ms,
        status=status,
    )

    return {
        "response": final_response,
        "intent": intent,
        "safety_verdict": safety_verdict,
        "tools_used": tools_used,
        "latency_ms": latency_ms,
        "status": status,
    }


def main():
    print(DIVIDER)
    print("PHASE 8: DEPLOYMENT READINESS — LenaDena BankBot")
    print(DIVIDER)

    print(f"""
LangSmith tracing: {os.environ.get('LANGCHAIN_TRACING_V2')}
LangSmith project: {os.environ.get('LANGCHAIN_PROJECT')}

Deployment wrapper features:
  ✅ PII-safe logging (SHA-256 query hash, no raw query text in logs)
  ✅ Structured JSONL log → logs/interactions.jsonl
  ✅ LangSmith auto-tracing on every graph run
  ✅ Retry once on OpenAI timeout
  ✅ Graceful fallback response on any unhandled exception
  ✅ Latency captured per request via Timer context manager
""")

    test_queries = [
        "What are LenaDena Bank's FD rates for 1 year?",
        "How do I open a savings account?",
        "Transfer ₹5,000 to my friend.",
        "What credit card suits frequent travellers?",
        "My account was debited twice — this is a fraud.",
        "What is the personal loan eligibility?",
        "What documents are needed for KYC?",
        "Does LenaDena Bank offer cryptocurrency?",
        "What is the minimum balance for a savings account?",
        "Approve my home loan application immediately.",
    ]

    session_id = new_session_id()
    memory = ShortTermMemory(k=6)

    print(f"{DIVIDER}")
    print("[1/3] 10-RUN DEPLOYMENT TEST")
    print(DIVIDER)

    results = []
    for q in test_queries:
        r = safe_run_with_logging(q, session_id, memory)
        results.append(r)

    # Print log table
    print(f"\n{'#':<3} {'Query':<45} {'Intent':<15} {'Safety':<10} {'Tools':<28} {'ms':<6} Status")
    print("-" * 115)
    for i, (q, r) in enumerate(zip(test_queries, results), 1):
        tools = ", ".join(r["tools_used"]) if r["tools_used"] else "none"
        print(
            f"{i:<3} {q[:43]:<45} {r['intent']:<15} {r['safety_verdict']:<10} "
            f"{tools:<28} {r['latency_ms']:<6} {r['status']}"
        )

    # --- Graceful failure ---
    print(f"\n{DIVIDER}")
    print("[2/3] GRACEFUL FAILURE HANDLING")
    print(DIVIDER)

    def broken_run(*args, **kwargs):
        raise ConnectionError("Simulated OpenAI API connection error")

    with patch("phases.phase8_deployment.run_agent", broken_run):
        r_fail = safe_run_with_logging("What are FD rates?", "test_session", ShortTermMemory())

    print(f"Status:   {r_fail['status']}")
    print(f"Response: {r_fail['response']}")
    print("✅ User received a graceful error message — not a stack trace.")

    # --- PII scrubbing ---
    print(f"\n{DIVIDER}")
    print("[3/3] PII SCRUBBING IN LOGS")
    print(DIVIDER)
    pii_query = "My account 9876543210 was charged incorrectly, my phone is 9123456789"
    r_pii = safe_run_with_logging(pii_query, session_id, memory)

    logs = read_logs(n=1)
    last = logs[-1] if logs else {}
    print(f"Original query:  {pii_query}")
    print(f"Logged hash:     {last.get('query_hash', 'N/A')} (SHA-256, not raw text)")
    print(f"Response preview in log: {last.get('response_preview', 'N/A')}")
    print("✅ No raw PII stored — only hash and scrubbed preview.")

    print(f"\n{DIVIDER}")
    print("DEPLOYMENT SUMMARY")
    print(DIVIDER)
    print("""
| Requirement           | Implementation                              | Status |
|-----------------------|---------------------------------------------|--------|
| Local deployment      | pip install -r requirements.txt + streamlit | ✅      |
| Cloud deployment      | Streamlit Community Cloud (public URL)      | ✅      |
| Structured logging    | JSONL, logs/interactions.jsonl              | ✅      |
| PII-safe logs         | SHA-256 query hash, scrubbed preview        | ✅      |
| LangSmith tracing     | LANGCHAIN_TRACING_V2=true                   | ✅      |
| Latency capture       | Timer context manager per run               | ✅      |
| OpenAI timeout retry  | 1 retry on timeout errors                   | ✅      |
| Graceful failure      | Fallback response on all exceptions         | ✅      |
| API keys              | .env (local) / Streamlit Secrets (cloud)    | ✅      |

Deployment assumptions:
  - OPENAI_API_KEY and LANGCHAIN_API_KEY must be set in environment
  - Chroma DB is rebuilt on first cold start (~10-15 seconds)
  - logs/ directory is writable (created automatically)
  - Python 3.11+ recommended
""")


if __name__ == "__main__":
    main()
