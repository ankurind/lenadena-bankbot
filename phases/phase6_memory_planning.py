"""
Phase 6: Planning, Memory & Multi-Agent Context (LangGraph)
LenaDena BankBot

Goal: Demonstrate the LangGraph StateGraph (Triage → Advisory → Review),
      short-term + long-term memory, multi-step reasoning, multi-turn context carryover.
Run: python phases/phase6_memory_planning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agent.graph import run_agent
from agent.memory import ShortTermMemory, get_long_term_memory
from agent.logging_utils import new_session_id

DIVIDER = "=" * 70


def main():
    print(DIVIDER)
    print("PHASE 6: LANGGRAPH MULTI-AGENT + MEMORY — LenaDena BankBot")
    print(DIVIDER)

    print("""
Architecture:
  User Query
       │
  [Triage Agent]
    ├── Programmatic safety check (regex rules)
    ├── LLM intent classification
    └── Route: REFUSE | ESCALATE | PROCEED
            │              │              │
         [Refuse]     [Escalate]   [Advisory Agent]
         response      response     ├── RAG retrieval (Chroma)
                                    ├── Tool selection & calling
                                    └── Draft response
                                             │
                                     [Review Node]
                                     Safety review → Final response

State fields:
  query, session_id, intent, safety_verdict, retrieved_docs,
  tool_calls_made, draft_response, final_response,
  short_term_history (last 6 turns), long_term_prefs
""")

    # --- Single query: trace the graph ---
    print(f"\n{DIVIDER}")
    print("[1/5] Single query — full graph trace")
    print(DIVIDER)
    session_id = new_session_id()
    memory = ShortTermMemory(k=6)
    query = "What are the LenaDena Bank FD rates for 1 year and 2 years?"
    result = run_agent(query, session_id, memory)
    print(f"Query:          {result['query']}")
    print(f"Intent:         {result['intent']}")
    print(f"Safety verdict: {result['safety_verdict']}")
    print(f"Tools used:     {result['tool_calls_made']}")
    print(f"\nFinal response:\n{result['final_response']}")

    # --- Multi-step reasoning ---
    print(f"\n{DIVIDER}")
    print("[2/5] Multi-step reasoning — Card vs Loan comparison")
    print(DIVIDER)
    complex_query = "Should I get a credit card or a personal loan for a ₹50,000 expense? What are the costs?"
    result2 = run_agent(complex_query, session_id, memory)
    print(f"Query: {complex_query}")
    print(f"Intent: {result2['intent']} | Tools: {result2['tool_calls_made']}")
    print(f"\nFinal response:\n{result2['final_response']}")

    # --- Multi-turn conversation ---
    print(f"\n{DIVIDER}")
    print("[3/5] Multi-turn conversation — 5 turns with context carryover")
    print(DIVIDER)
    session_mt = new_session_id()
    memory_mt = ShortTermMemory(k=6)
    turns = [
        "I'm looking at LenaDena Bank FD options.",
        "What's the rate for 2 years?",
        "And what about the premature withdrawal penalty?",
        "Is the penalty different for senior citizens?",
        "Good to know. Now, what credit cards does LenaDena offer?",
    ]
    for i, turn in enumerate(turns, 1):
        r = run_agent(turn, session_mt, memory_mt)
        print(f"\nTurn {i}")
        print(f"  User:    {turn}")
        print(f"  BankBot: {r['final_response'][:200]}")

    print(f"\n  Short-term memory now holds {len(memory_mt.history)} messages.")

    # --- Memory comparison ---
    print(f"\n{DIVIDER}")
    print("[4/5] Memory comparison — WITH vs WITHOUT context")
    print(DIVIDER)

    # Without memory
    r_without = run_agent("And what about the penalty?", new_session_id(), ShortTermMemory(k=0))
    print("WITHOUT MEMORY:")
    print(f"  Q: 'And what about the penalty?'")
    print(f"  A: {r_without['final_response'][:200]}")

    # With memory (seeded with FD conversation)
    session_with = new_session_id()
    mem_with = ShortTermMemory(k=6)
    run_agent("I'm looking at FD options at LenaDena Bank.", session_with, mem_with)
    run_agent("What's the 2-year FD rate?", session_with, mem_with)
    r_with = run_agent("And what about the penalty?", session_with, mem_with)
    print("\nWITH MEMORY (after 2 FD-related turns):")
    print(f"  Q: 'And what about the penalty?'")
    print(f"  A: {r_with['final_response'][:200]}")

    # --- Long-term memory ---
    print(f"\n{DIVIDER}")
    print("[5/5] Long-term memory — preference persistence across sessions")
    print(DIVIDER)
    lt = get_long_term_memory()
    lt.set_preference(session_id, "product_interest", "Fixed Deposits")
    lt.set_preference(session_id, "preferred_tenure", "1-2 years")
    print(f"Saved preferences for session {session_id}: {lt.get_preferences(session_id)}")

    # New short-term session inherits long-term prefs
    new_st = ShortTermMemory(k=6)
    r_lt = run_agent("What would you recommend for me?", session_id, new_st)
    print(f"\nQ: 'What would you recommend for me?' (with long-term prefs loaded)")
    print(f"A: {r_lt['final_response'][:300]}")

    print(f"\n{DIVIDER}")
    print("MEMORY SUMMARY")
    print(DIVIDER)
    print("""
| Capability              | Implementation                          | Evidence              |
|-------------------------|-----------------------------------------|-----------------------|
| Multi-agent flow        | LangGraph StateGraph (5 nodes)          | Graph trace above     |
| Intent classification   | Triage Agent → LLM                      | intent field in state |
| Safety gate             | Triage + check_safety() regex           | REFUSE/ESCALATE route |
| RAG in advisory         | Chroma retrieval → context injection    | retrieved_docs field  |
| Tool use in advisory    | create_openai_tools_agent               | tool_calls_made field |
| Post-response review    | Review node (Triage 2nd pass)           | draft vs final resp.  |
| Short-term memory       | ShortTermMemory(k=6) sliding window     | Turn 3 refs Turn 1    |
| Long-term memory        | JSON file logs/user_preferences.json    | Prefs across sessions |
| Memory reset rule       | Short-term: new instance; Long-term: persistent file |          |
| Multi-step reasoning    | Advisory decomposes complex query       | Card vs loan compare  |
""")


if __name__ == "__main__":
    main()
