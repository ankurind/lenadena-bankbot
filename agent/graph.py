"""
LangGraph multi-agent StateGraph for LenaDena BankBot.

Two agents:
  Triage Agent  — intent classification, safety gate, routing, post-response review
  Advisory Agent — RAG retrieval, tool use, response drafting

Flow:
  User Query → triage_node → [REFUSE | ESCALATE | advisory_node → review_node] → final response
"""

import uuid
from typing import TypedDict, Annotated, Literal, Optional
import operator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END

from agent.safety import check_safety, get_refusal_message, get_escalation_message, scrub_pii
from agent.tools import ALL_TOOLS
from agent.retrieval import retrieve, format_context
from agent.memory import ShortTermMemory, get_long_term_memory

llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_smart = ChatOpenAI(model="gpt-4o", temperature=0.2)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    session_id: str
    intent: Optional[str]
    safety_verdict: Optional[str]        # PROCEED | REFUSE | ESCALATE
    safety_reason: Optional[str]
    retrieved_docs: Optional[str]
    tool_calls_made: list[str]
    draft_response: Optional[str]
    final_response: Optional[str]
    short_term_history: list[dict]       # [{role, content}]
    long_term_prefs: dict
    needs_review: bool


# ── Triage Node ───────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = """You are the Triage Agent for LenaDena BankBot.

Your tasks:
1. Classify the customer's intent into one of:
   account_info | loan_info | card_info | fd_info | dispute | fraud | policy | eligibility | out_of_scope | unclear

2. Determine routing:
   - REFUSE: ONLY for direct requests to perform money movement, execute transactions, approve loans/accounts, provide legal advice, or share credentials. NOT for information queries about services the bank may not offer.
   - ESCALATE: fraud suspicion, dispute, blocked account, urgent unresolved complaint
   - PROCEED: all informational and advisory queries, including questions about whether the bank offers a service (even if the answer is "no"), product comparisons, rates, eligibility information

Important: asking "does the bank offer X?" is always PROCEED, even if X is out of scope.

Output your response in this exact format:
INTENT: <intent_category>
ROUTING: <REFUSE|ESCALATE|PROCEED>
REASON: <one sentence explaining your decision>"""


def triage_node(state: AgentState) -> AgentState:
    """Classifies intent and determines routing."""
    # First: programmatic safety check (fast, rule-based)
    verdict, reason = check_safety(state["query"])

    if verdict in ("REFUSE", "ESCALATE"):
        return {
            **state,
            "safety_verdict": verdict,
            "safety_reason": reason,
            "intent": "flagged",
        }

    # LLM-based intent classification
    messages = [
        SystemMessage(content=TRIAGE_SYSTEM),
        HumanMessage(content=f"Customer query: {state['query']}"),
    ]
    response = llm_fast.invoke(messages)
    raw = response.content

    # Parse structured output
    intent = "unclear"
    routing = "PROCEED"
    for line in raw.splitlines():
        if line.startswith("INTENT:"):
            intent = line.split(":", 1)[1].strip()
        elif line.startswith("ROUTING:"):
            routing = line.split(":", 1)[1].strip()

    return {
        **state,
        "intent": intent,
        "safety_verdict": routing,
        "safety_reason": raw,
    }


# ── Advisory Node ─────────────────────────────────────────────────────────────

ADVISORY_SYSTEM = """You are the Advisory Agent for LenaDena BankBot.

Your job:
- Use the provided tools and context to answer the customer's question accurately
- Base your answer ONLY on the provided context and tool results
- Never invent rates, policies, or product details
- Include your confidence level (High/Medium/Low) at the end

Safety constraints:
- NEVER perform transactions or approvals
- If you cannot find the answer in context, say so and direct to 1800-123-5362

{history_section}
{prefs_section}"""

ADVISORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ADVISORY_SYSTEM),
    ("human", "Context:\n{context}\n\nCustomer question: {query}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def advisory_node(state: AgentState) -> AgentState:
    """Retrieves context, uses tools, drafts a response."""
    # RAG retrieval
    docs = retrieve(state["query"], k=3)
    context = format_context(docs)

    # Build history and prefs sections
    history_section = ""
    if state["short_term_history"]:
        lines = ["Prior conversation turns:"]
        for msg in state["short_term_history"][-6:]:
            role = "Customer" if msg["role"] == "user" else "BankBot"
            lines.append(f"  {role}: {msg['content']}")
        history_section = "\n".join(lines)

    prefs_section = ""
    if state["long_term_prefs"]:
        lines = ["Known preferences from prior sessions:"]
        for k, v in state["long_term_prefs"].items():
            lines.append(f"  - {k}: {v}")
        prefs_section = "\n".join(lines)

    # Build system prompt with context injected
    system_content = (
        ADVISORY_SYSTEM.format(
            history_section=history_section,
            prefs_section=prefs_section,
        )
        + f"\n\nContext from knowledge base:\n{context}"
    )

    agent = create_agent(llm_fast, ALL_TOOLS)
    result = agent.invoke({
        "messages": [
            ("system", system_content),
            ("human", state["query"]),
        ]
    })

    # Extract final text from last AI message
    messages = result.get("messages", [])
    draft = "I'm unable to process that request. Please call 1800-123-5362."
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            draft = msg.content
            break

    # Track which tools were called (ToolMessage role indicates tool use)
    tools_used = []
    for msg in messages:
        if hasattr(msg, "name") and msg.name:
            tools_used.append(msg.name)
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", ""))
    tools_used = list(dict.fromkeys(t for t in tools_used if t))

    return {
        **state,
        "retrieved_docs": context,
        "tool_calls_made": tools_used,
        "draft_response": draft,
        "needs_review": True,
    }


# ── Review Node (Triage Agent second pass) ────────────────────────────────────

REVIEW_SYSTEM = """You are the safety reviewer for LenaDena BankBot.

Review the draft response ONLY for genuine safety issues:
1. Does it attempt to perform a transaction, approve a loan, or give legal advice? → Remove and add disclaimer.
2. Does it contain raw PII (account numbers, phone numbers)? → Scrub it.
3. Does it make up facts NOT present in the context provided? → Correct or remove.

Do NOT remove or rewrite factual rate/product information that came from the knowledge base context.
Do NOT add disclaimers just because rates "might change" — the agent already sources from our own data.
If the response is informational and accurate: output it EXACTLY as-is, without any rewrites.

The goal is to block genuinely unsafe content, NOT to second-guess factual answers."""


def review_node(state: AgentState) -> AgentState:
    """Triage Agent reviews draft for safety before returning to user."""
    messages = [
        SystemMessage(content=REVIEW_SYSTEM),
        HumanMessage(content=(
            f"Draft response:\n{state['draft_response']}\n\n"
            f"Original query: {state['query']}\n"
            f"Context used:\n{state.get('retrieved_docs', '')[:500]}"
        )),
    ]
    reviewed = llm_fast.invoke(messages)
    return {
        **state,
        "final_response": reviewed.content,
        "needs_review": False,
    }


# ── Refuse / Escalate Nodes ───────────────────────────────────────────────────

def refuse_node(state: AgentState) -> AgentState:
    reason = state.get("safety_reason", "money_movement")
    msg = get_refusal_message(reason)
    return {**state, "final_response": msg, "draft_response": msg}


def escalate_node(state: AgentState) -> AgentState:
    ticket_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
    msg = get_escalation_message(ticket_id)
    return {**state, "final_response": msg, "draft_response": msg}


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_after_triage(state: AgentState) -> Literal["advisory", "refuse", "escalate"]:
    verdict = state.get("safety_verdict", "PROCEED")
    if verdict == "REFUSE":
        return "refuse"
    if verdict == "ESCALATE":
        return "escalate"
    return "advisory"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("advisory", advisory_node)
    graph.add_node("review", review_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "advisory": "advisory",
            "refuse": "refuse",
            "escalate": "escalate",
        }
    )

    graph.add_edge("advisory", "review")
    graph.add_edge("review", END)
    graph.add_edge("refuse", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# Module-level compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    query: str,
    session_id: str,
    short_term_memory: Optional[ShortTermMemory] = None,
) -> dict:
    """
    High-level entry point for running the full multi-agent graph.
    Returns state dict with final_response, intent, tools_used, etc.
    """
    lt_memory = get_long_term_memory()
    history = short_term_memory.get_history() if short_term_memory else []
    prefs = lt_memory.get_preferences(session_id)

    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "intent": None,
        "safety_verdict": None,
        "safety_reason": None,
        "retrieved_docs": None,
        "tool_calls_made": [],
        "draft_response": None,
        "final_response": None,
        "short_term_history": history,
        "long_term_prefs": prefs,
        "needs_review": False,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    # Update short-term memory
    if short_term_memory is not None:
        short_term_memory.add_user(query)
        short_term_memory.add_assistant(final_state.get("final_response", ""))

    return final_state
