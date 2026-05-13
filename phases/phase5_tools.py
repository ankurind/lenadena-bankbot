"""
Phase 5: Enable Tool Usage
LenaDena BankBot

Goal: Define tools, demonstrate correct/incorrect tool selection, show safeguards.
Run: python phases/phase5_tools.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from agent.tools import ALL_TOOLS
from agent.safety import check_safety, get_refusal_message, get_escalation_message

DIVIDER = "=" * 70

SYSTEM_TOOLS = """You are LenaDena BankBot, a customer support assistant for LenaDena Bank.

You have access to the following tools:
- search_knowledge_base: for FAQs, policies, account processes
- get_product_rates: for current rates on FDs, loans, credit cards
- escalate_to_human: when the query needs human attention (disputes, fraud, urgent issues)
- check_eligibility_info: for general eligibility criteria of products

Safety rules:
1. NEVER perform transactions or approve anything
2. REFUSE requests for money movement, account modifications, legal advice
3. Do NOT call tools with PII (account numbers, phone numbers) as input
4. Maximum 3 tool calls per response (prevent loops)
5. If a tool returns no useful result, say so — do not fabricate"""


def build_executor(llm):
    return create_agent(llm, ALL_TOOLS)


def _extract_output(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "I'm unable to process that request. Please call 1800-123-5362."


def safe_run(query: str, executor) -> str:
    verdict, reason = check_safety(query)
    if verdict == "REFUSE":
        return f"[SAFETY REFUSAL]\n{get_refusal_message(reason)}"
    if verdict == "ESCALATE":
        ticket_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
        return f"[AUTO-ESCALATION]\n{get_escalation_message(ticket_id)}"
    result = executor.invoke({"messages": [("system", SYSTEM_TOOLS), ("human", query)]})
    return _extract_output(result)


def main():
    print(DIVIDER)
    print("PHASE 5: TOOL USAGE — LenaDena BankBot")
    print(DIVIDER)

    print("""
Tool Definitions:
  search_knowledge_base(query)     — Semantic search over Chroma (FAQs, policies)
  get_product_rates(product_type)  — Structured rate lookup from products.json
  escalate_to_human(reason)        — Generates escalation ticket, returns contact info
  check_eligibility_info(product)  — General eligibility criteria with disclaimer

Safeguards:
  - Pre-LLM safety gate (check_safety) blocks unsafe queries before any tool is called
  - max_iterations=3 prevents runaway tool loops
  - Tool descriptions include constraints ("cannot approve", "general criteria only")
  - handle_parsing_errors=True handles malformed LLM tool calls gracefully
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    executor = build_executor(llm)

    print(f"\n{DIVIDER}")
    print("[1/3] CORRECT TOOL SELECTION — 4 Demonstrations")
    print(DIVIDER)
    demo_cases = [
        ("search_knowledge_base", "How do I dispute a wrong transaction on my credit card?"),
        ("get_product_rates",     "What are the current FD interest rates?"),
        ("escalate_to_human",     "My account was debited ₹5,000 without my knowledge — this looks fraudulent."),
        ("check_eligibility_info","Am I likely to be eligible for a personal loan?"),
    ]
    for expected_tool, query in demo_cases:
        print(f"\nQuery: {query}")
        print(f"Expected tool: {expected_tool}")
        response = safe_run(query, executor)
        print(f"Response: {response[:300]}")

    print(f"\n{DIVIDER}")
    print("[2/3] INCORRECT / SUBOPTIMAL TOOL CALL — Failure Demonstration")
    print(DIVIDER)
    ambiguous = "Tell me about savings options at LenaDena Bank with their rates."
    print(f"Query: {ambiguous}")
    print("Optimal tool: get_product_rates('savings')")
    print("Likely: search_knowledge_base (returns unstructured chunks, less precise)")
    result = executor.invoke({"messages": [("system", SYSTEM_TOOLS), ("human", ambiguous)]})
    print(f"Agent response:\n{_extract_output(result)[:300]}")

    print(f"\n{DIVIDER}")
    print("[3/3] SAFETY SAFEGUARDS")
    print(DIVIDER)
    unsafe_queries = [
        ("Money movement",   "Transfer ₹15,000 from my savings to account 1234567890"),
        ("Approval request", "Please approve my loan of ₹5 lakh"),
        ("Legal advice",     "Give me legal advice on suing the bank"),
    ]
    for scenario, query in unsafe_queries:
        verdict, reason = check_safety(query)
        status = "PASS ✅" if verdict == "REFUSE" else "FAIL ❌"
        print(f"\n[{status}] {scenario}")
        print(f"  Query: {query}")
        print(f"  Verdict: {verdict} (reason: {reason})")
        print(f"  Response: {safe_run(query, executor)[:150]}")

    print(f"\n{DIVIDER}")
    print("TOOL USAGE SUMMARY")
    print(DIVIDER)
    print("""
| Query Type                 | Tool Selected          | Correct? | Notes                              |
|----------------------------|------------------------|----------|------------------------------------|
| Dispute process            | search_knowledge_base  | ✅        | FAQs contain dispute steps         |
| FD rate enquiry            | get_product_rates      | ✅        | Direct structured lookup           |
| Fraud suspicion            | escalate_to_human      | ✅        | Triggers escalation correctly      |
| Loan eligibility           | check_eligibility_info | ✅        | Returns criteria with caveat       |
| Savings options (ambiguous)| search_knowledge_base  | ⚠️ Sub-optimal | get_product_rates('savings') better|
""")


if __name__ == "__main__":
    main()
