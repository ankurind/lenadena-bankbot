"""
LenaDena BankBot — Streamlit Chat Interface
Main entry point for Streamlit Community Cloud deployment.
"""

import sys
import os
import uuid

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# ── Load secrets / env ────────────────────────────────────────────────────────
def _load_secrets():
    """Load API keys from .env first, then override with Streamlit secrets if available."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if "LANGCHAIN_API_KEY" in st.secrets:
            os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        if "LANGCHAIN_TRACING_V2" in st.secrets:
            os.environ["LANGCHAIN_TRACING_V2"] = str(st.secrets["LANGCHAIN_TRACING_V2"])
        if "LANGCHAIN_PROJECT" in st.secrets:
            os.environ["LANGCHAIN_PROJECT"] = str(st.secrets["LANGCHAIN_PROJECT"])
        if "LANGSMITH_ENDPOINT" in st.secrets:
            os.environ["LANGSMITH_ENDPOINT"] = str(st.secrets["LANGSMITH_ENDPOINT"])
    except Exception:
        pass

_load_secrets()

# ── Lazy imports (after env is set) ──────────────────────────────────────────
from agent.graph import run_agent
from agent.memory import ShortTermMemory, get_long_term_memory
from agent.feedback import record_feedback, apply_adaptive_behaviour
from agent.logging_utils import new_session_id, Timer, log_interaction
from agent.retrieval import build_vectorstore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LenaDena BankBot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise vector store on first load ─────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base...")
def init_vectorstore():
    return build_vectorstore()

init_vectorstore()

# ── Session state setup ───────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

if "memory" not in st.session_state:
    st.session_state.memory = ShortTermMemory(k=6)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_key" not in st.session_state:
    st.session_state.last_response_key = None

if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None  # {"query": ..., "response": ...}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1a3c5e/ffffff?text=LenaDena+Bank", use_container_width=True)
    st.markdown("### 🏦 LenaDena BankBot")
    st.markdown(
        "AI-powered support assistant for LenaDena Bank. "
        "Ask about accounts, FD rates, loans, credit cards, and policies."
    )
    st.divider()
    st.markdown("**Session ID:** `" + st.session_state.session_id + "`")
    st.markdown("**Turns:** " + str(len(st.session_state.messages) // 2))

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = new_session_id()
        st.session_state.memory = ShortTermMemory(k=6)
        st.session_state.messages = []
        st.session_state.pending_feedback = None
        st.rerun()

    st.divider()
    st.markdown("**Try asking:**")
    sample_queries = [
        "What are FD rates for 2 years?",
        "Which credit card suits a frequent traveller?",
        "Transfer ₹5,000 to my friend.",
        "Am I eligible for a home loan?",
        "My account was debited incorrectly.",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state["_prefill"] = q
            st.rerun()

    st.divider()
    st.caption("⚠️ LenaDena Bank is a fictional bank for demonstration purposes.")

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🏦 LenaDena BankBot — AI Banking Support")
st.caption("Non-transactional advisory assistant | Powered by GPT-4o-mini + LangChain + LangGraph")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Agent details", expanded=False):
                meta = msg["meta"]
                cols = st.columns(3)
                cols[0].metric("Intent", meta.get("intent", "—"))
                cols[1].metric("Safety", meta.get("safety_verdict", "—"))
                cols[2].metric("Latency", f"{meta.get('latency_ms', 0)}ms")
                if meta.get("tools_used"):
                    st.markdown("**Tools called:** " + ", ".join(f"`{t}`" for t in meta["tools_used"]))
                if meta.get("langsmith_url"):
                    st.markdown(f"[🔗 View LangSmith trace]({meta['langsmith_url']})")

# Feedback buttons for last response
if st.session_state.pending_feedback:
    pf = st.session_state.pending_feedback
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍 Helpful", key="thumbs_up"):
            record_feedback(pf["query"], pf["response"], "thumbs_up", st.session_state.session_id)
            st.session_state.pending_feedback = None
            st.success("Thanks for your feedback!")
            st.rerun()
    with col2:
        if st.button("👎 Not helpful", key="thumbs_down"):
            record_feedback(pf["query"], pf["response"], "thumbs_down", st.session_state.session_id)
            st.session_state.pending_feedback = None
            st.warning("Thanks — we'll use this to improve.")
            st.rerun()

# Chat input
prefill = st.session_state.pop("_prefill", "")
user_input = st.chat_input("Ask LenaDena BankBot anything about accounts, rates, loans, or policies...")

if user_input or prefill:
    query = user_input or prefill

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                with Timer() as t:
                    state = run_agent(query, st.session_state.session_id, st.session_state.memory)

                response = apply_adaptive_behaviour(
                    state.get("final_response", "I'm unable to process that. Please call 1800-123-5362."),
                    query
                )

                st.markdown(response)

                meta = {
                    "intent": state.get("intent", "—"),
                    "safety_verdict": state.get("safety_verdict", "—"),
                    "tools_used": state.get("tool_calls_made", []),
                    "latency_ms": t.elapsed_ms,
                    "langsmith_url": None,
                }

                with st.expander("🔍 Agent details", expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Intent", meta["intent"])
                    cols[1].metric("Safety", meta["safety_verdict"])
                    cols[2].metric("Latency", f"{meta['latency_ms']}ms")
                    if meta["tools_used"]:
                        st.markdown("**Tools called:** " + ", ".join(f"`{t}`" for t in meta["tools_used"]))

            except Exception as e:
                import traceback
                response = "I'm sorry, I encountered a technical issue. Please try again or call 1800-123-5362."
                meta = {"intent": "error", "safety_verdict": "ERROR", "tools_used": [], "latency_ms": 0}
                st.error(f"Error: {type(e).__name__}: {e}")
                st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": response, "meta": meta})
    st.session_state.pending_feedback = {"query": query, "response": response[:150]}
    st.rerun()
