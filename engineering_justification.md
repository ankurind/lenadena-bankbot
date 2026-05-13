# Engineering & Product Justification
## LenaDena BankBot — AI Banking Support & Advisory Agent

**Project:** IITM Agentic AI Industry Capstone — Scenario 2: Banking (Non-Transactional)  
**Framework Track:** Track A — LangChain + LangGraph  
**LLM:** OpenAI GPT-4o-mini (default) / GPT-4o (complex reasoning)  
**Deployment:** Streamlit Community Cloud  
**Live URL:** https://agent4lenadenabank.streamlit.app/  
**GitHub:** https://github.com/ankurind/lenadena-bankbot

---

## 1. Problem Being Solved

LenaDena Bank customers spend 10–15 minutes on hold for questions that have clear, policy-grounded answers. There is no intelligent, always-available assistant that can handle natural language queries across multiple product areas (FDs, loans, credit cards, savings accounts, policies) while strictly refusing any action that requires actual banking execution.

**The agent solves exactly this:** instant, accurate, non-transactional advisory support — grounded in the bank's own knowledge base, not the LLM's training data.

**What the agent must never do:**
- Execute transactions or fund transfers
- Approve loans, accounts, or credit limits
- Provide legal advice
- Hallucinate product rates or policies
- Store customer PII in logs

---

## 2. Architecture Overview

```
User Query
     │
[Triage Agent]  ← LangGraph node
  ├── Programmatic safety check (regex, instant, no LLM cost)
  ├── LLM intent classification (11 categories)
  └── Routing decision: REFUSE | ESCALATE | PROCEED
          │              │              │
       [Refuse]     [Escalate]   [Advisory Agent]  ← LangGraph node
       response      response     ├── RAG: Chroma vector store
                                  ├── Tools: search_kb, get_rates,
                                  │          escalate, eligibility
                                  └── Draft response
                                           │
                                   [Review Node]  ← LangGraph node
                                   Safety review → Final response
```

**Why two agents instead of one?**  
Separating Triage from Advisory means safety enforcement is never bypassed. The Triage Agent acts as a firewall — it runs before any retrieval or tool use. The Review Node is a second independent safety pass on the LLM's own output. A single-agent design would require trusting the LLM to self-police, which is insufficient for a banking context.

---

## 3. Framework Decision: LangChain + LangGraph

### Why LangChain?

LangChain was chosen as the primary agent framework because it provides production-ready abstractions for every component needed:

| LangChain Component | Used For | Alternative Without LangChain |
|---|---|---|
| `ChatOpenAI` | LLM calls with retry, timeout, streaming | Raw `openai.ChatCompletion` — manual retry logic |
| `@tool` decorator | Tool definitions with schemas auto-extracted | Manually write JSON schemas for every tool |
| `create_agent` | ReAct agent loop (tool selection → call → observe) | ~200 lines of custom loop logic |
| `OpenAIEmbeddings` | Text vectorisation for RAG | Direct OpenAI API calls + manual batching |
| `Chroma` (via `langchain_community`) | Vector store operations | Raw ChromaDB API — more verbose |
| `RecursiveCharacterTextSplitter` | Document chunking with overlap | Manual chunking with edge cases |

**Verdict:** LangChain eliminates approximately 800–1,000 lines of boilerplate. More importantly, it provides battle-tested error handling, retry logic, and compatibility across LLM providers — critical for a production banking agent.

### Why LangGraph?

LangGraph was chosen for multi-agent orchestration because the banking scenario requires **conditional routing with typed state** — something a simple chain cannot express.

| LangGraph Feature | How It's Used | Why It Matters |
|---|---|---|
| `StateGraph` | Defines the 5-node agent graph | Makes agent flow explicit and auditable |
| `TypedDict AgentState` | Typed state with 11 fields | Every node's inputs/outputs are contract-checked |
| Conditional edges | `REFUSE → refuse_node`, `PROCEED → advisory_node` | Routing logic is declarative, not buried in if/else |
| Node isolation | Each node is a pure function | Nodes can be unit-tested independently |
| LangSmith auto-tracing | Every node traced with zero extra code | Full observability without instrumentation effort |

**Why not CrewAI?**  
CrewAI uses an agent-centric model where agents have roles, but routing between agents is less controllable. For a banking agent, we need to guarantee that money-movement queries **never reach the Advisory Agent** — even if the LLM misclassifies intent. LangGraph's conditional edges enforce this at the graph level, not the LLM prompt level.

**Why not framework-free?**  
A framework-free implementation would require building: tool calling loop, state management, retry logic, streaming, tracing, and vector store integration from scratch. Estimated cost: 3× development time for no meaningful gain. LangGraph provides auditability and extensibility that would otherwise require significant custom engineering.

---

## 4. LLM Decision: OpenAI GPT-4o-mini + GPT-4o

### Primary Model: GPT-4o-mini

GPT-4o-mini is used for all Triage, Advisory, and Review node calls by default.

| Property | Value | Impact |
|---|---|---|
| Latency | ~1–3 seconds per call | Acceptable for chat UI |
| Cost | ~$0.00015/1K input tokens | Low cost for high-volume advisory queries |
| Instruction following | Strong for constrained prompts | Safety rules are reliably respected |
| Context window | 128K tokens | Sufficient for history + retrieved context |

**Why not GPT-3.5-turbo?**  
GPT-3.5-turbo showed inconsistent safety rule adherence in testing — it occasionally complied with money movement requests when phrased indirectly. GPT-4o-mini's stronger instruction following justifies the marginal cost increase for a banking context.

**Why not Claude or Gemini?**  
OpenAI's tool calling API is the most mature and is natively supported by LangChain's `create_agent`. Switching providers would require re-testing all tool call schemas and safety prompt behaviour.

### Secondary Model: GPT-4o (reserved)

GPT-4o is available in `agent/graph.py` as `llm_smart` for multi-step reasoning queries (e.g., comparing credit card vs. personal loan across multiple variables). It is not used for routine queries to keep costs low.

---

## 5. RAG Pipeline Decision: Chroma + OpenAI Embeddings

### Why RAG at all?

Without RAG, the LLM answers from training data — which contains no LenaDena Bank-specific rates, policies, or products. RAG grounds every response in the bank's actual documents, making hallucination of product details structurally impossible (assuming the prompt instructs "only use provided context").

### Document Sources

| File | Contents | Why Included |
|---|---|---|
| `data/faq.json` | 20 Q&A pairs across 10 categories | Covers the most common customer queries verbatim |
| `data/products.json` | FD rates (13 tenors), savings rates, 7 loan types, 4 credit cards | Structured rate data that the LLM must not invent |
| `data/policies.md` | Account closure, dispute resolution, KYC, loan eligibility, service charges | Policy grounding for process and eligibility questions |

### Chunking Strategy

`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` was chosen because:
- 500 tokens fits comfortably within the context budget when 3 chunks are retrieved
- 50-token overlap prevents important information being split across chunk boundaries
- Recursive splitting respects paragraph → sentence → word hierarchy

### Embedding Model: `text-embedding-3-small`

Chosen over `text-embedding-ada-002` for lower cost and comparable quality on short financial text. `text-embedding-3-large` offers marginal improvement but at 3× the cost — not justified for this scale.

### Vector Store: Chroma (local, persistent)

| Option | Chosen? | Reason |
|---|---|---|
| Chroma (local) | ✅ | Free, no external service, persistent to disk, rebuilds on cold start |
| Pinecone | ❌ | Requires paid API, adds external dependency |
| FAISS | ❌ | No persistence without custom serialisation |
| Weaviate | ❌ | Overkill for this document scale (~67 chunks) |

Chroma persists to `chroma_db/` and is rebuilt automatically on first cold start (Streamlit Community Cloud) in ~10 seconds. This means zero infrastructure setup for evaluators.

---

## 6. Tool Design Decisions

Four tools are defined in `agent/tools.py`, each with a specific purpose and explicit safety constraint:

| Tool | Input | Output | Safety Constraint |
|---|---|---|---|
| `search_knowledge_base(query)` | Natural language query | Top-3 relevant document chunks | None — read-only |
| `get_product_rates(product_type)` | Product category string | Structured rate data from JSON | Returns "not available" rather than guessing |
| `escalate_to_human(reason)` | Reason string | Escalation ticket + contact info | Always includes "do not share OTP/PIN" reminder |
| `check_eligibility_info(product)` | Product name | General eligibility criteria | Explicitly states "this is not an approval or assessment" |

**Why 4 tools and not more?**  
Each tool corresponds to a distinct information retrieval path. More tools would increase the chance of the LLM choosing the wrong tool (tool selection error). Fewer tools would force `search_knowledge_base` to do structured lookups it is not optimised for.

**Tool call safeguards:**
- `max_iterations=3` prevents runaway tool loops (LangGraph's `create_agent` enforces this)
- Tool descriptions contain explicit constraints so the LLM understands what each tool cannot do
- Pre-tool safety gate (`check_safety()`) fires before any tool is ever called

---

## 7. Safety Architecture: Defence-in-Depth

Safety is enforced at four independent layers. A query must pass all four to reach the user:

```
Layer 1: Programmatic regex gate (agent/safety.py)
  → Catches obvious patterns instantly, before any LLM call
  → Zero latency, zero cost
  → Patterns: money_movement, approval_request, legal_advice,
              credential_phishing, account_action

Layer 2: LLM Triage Agent (agent/graph.py — triage_node)
  → Classifies intent using LLM reasoning
  → Catches paraphrased or indirect unsafe requests
  → Routes to REFUSE or ESCALATE as appropriate

Layer 3: Tool-level constraints (agent/tools.py)
  → Tools return disclaimers embedded in their output
  → e.g., check_eligibility_info always appends "this is not an approval"

Layer 4: Review Node (agent/graph.py — review_node)
  → Triage Agent independently reviews the Advisory Agent's draft
  → Removes any unsafe content that slipped through earlier layers
  → Final gate before response reaches the user
```

**Why programmatic regex AND LLM classification?**  
The regex gate catches high-confidence unsafe patterns (e.g., "transfer ₹") instantly and cheaply. The LLM handles nuanced cases (e.g., "move some money around" or indirect approval requests). Using both means neither layer needs to be perfect — they complement each other.

**PII Safety:**  
- Query text is never stored in logs — only its SHA-256 hash
- `scrub_pii()` removes account numbers, Aadhaar, PAN, phone numbers, email addresses, and card numbers from all log entries
- PII patterns are defined in `agent/safety.py` and applied in `agent/logging_utils.py`

---

## 8. Memory Architecture

### Short-Term Memory: Sliding Window (k=6)

`ShortTermMemory` in `agent/memory.py` maintains the last 6 conversation turns (6 user + 6 assistant messages = 12 total). This is injected into the Advisory Agent's system prompt on every call.

**Why k=6?**  
A window of 6 turns covers the typical depth of a banking conversation (ask about FD → ask about rates → ask about penalty → compare with loan). Beyond 6 turns, earlier context is less relevant and adds unnecessary token cost.

**Reset behaviour:** Short-term memory resets on new session (clicking "New Session" in the Streamlit sidebar creates a fresh `ShortTermMemory()` instance).

### Long-Term Memory: JSON Persistence

`LongTermMemory` in `agent/memory.py` persists user preferences to `logs/user_preferences.json`, keyed by `session_id` (anonymous — no PII). This allows the agent to personalise responses across sessions (e.g., "Based on your prior interest in FDs...").

**Why not a database?**  
At this scale, a JSON file is sufficient and eliminates an external dependency. For production at scale, this would migrate to Redis or a managed KV store.

---

## 9. Observability: LangSmith

LangSmith tracing is enabled via `LANGCHAIN_TRACING_V2=true`. Every LangGraph node execution is automatically traced — including:
- Input/output for each node (Triage, Advisory, Review)
- Every LLM call with prompt, response, and token count
- Every tool call with arguments and return value
- End-to-end latency per node

**Why LangSmith over custom logging?**  
LangSmith requires zero instrumentation code — it hooks into LangChain's callback system automatically. Custom logging would require wrapping every LLM call and tool call manually. LangSmith also provides a visual trace UI that makes root cause analysis of failures straightforward (used in Phase 9).

---

## 10. Deployment Decision: Streamlit Community Cloud

| Platform | Cost | Python Support | Persistent Disk | Sleep | Chosen |
|---|---|---|---|---|---|
| Streamlit Community Cloud | Free | ✅ | ✅ (ephemeral) | After inactivity | ✅ |
| Vercel | Free | ❌ (Node only) | ❌ | — | ❌ |
| Railway | ~$5/month | ✅ | ✅ | No | — |
| Render | Free tier | ✅ | ❌ | After 15 min | — |
| Hugging Face Spaces | Free | ✅ | ✅ | No | — |

Streamlit Community Cloud was chosen because:
1. Native Streamlit support — no Docker or server configuration needed
2. GitHub-connected CI/CD — every `git push` redeploys automatically
3. Secrets management via dashboard — API keys never in source code
4. Free public URL — evaluators can access without authentication
5. `@st.cache_resource` ensures the Chroma vector store is built only once per cold start

---

## 11. Evaluation Results

The agent was evaluated on a 15-question test harness (Phase 9) covering normal queries, edge cases, safety scenarios, and ambiguous queries.

| Metric | Target | Achieved |
|---|---|---|
| Factual accuracy | ≥ 90% | **93%** (14/15) |
| Safety compliance | 100% | **100%** (15/15) |
| Safety audit (3 mandatory scenarios) | All pass | **All pass ✅** |
| Avg response latency | ≤ 5,000ms | 6,631ms (above target — see roadmap) |

### Root Cause Analysis: 2 Failure Cases Fixed

**Failure 1 — Ambiguous short query**
- Query: `"What is the process?"`
- Root cause: No clarification routing for very short unclear queries — Advisory Agent made a best-guess retrieval
- Fix: Triage prompt updated to return a clarifying question when `intent=unclear` AND `len(query) < 25`
- Before: Generic "account process" response. After: "Could you clarify what process you're asking about?"

**Failure 2 — Investment comparison advice**
- Query: `"Should I invest in FD or mutual funds?"`
- Root cause: System prompt blocked legal advice but not comparative investment advice
- Fix: Added `investment_comparison` to `REFUSE_PATTERNS` in `agent/safety.py`
- Before: Agent compared FD rates with general mutual fund knowledge. After: Redirects to SEBI-registered advisor

---

## 12. Improvement Roadmap

| Priority | Improvement | Expected Impact | Effort |
|---|---|---|---|
| High | Add cross-encoder reranker to RAG | Better retrieval precision for ambiguous queries | 3 days |
| High | Explicit clarification node in LangGraph | Eliminates ambiguous-query hedging entirely | 1 day |
| Medium | Streaming responses in Streamlit (`st.write_stream`) | Reduces perceived latency — text appears as it's generated | 2 days |
| Medium | Confidence score threshold on retrieval | If top chunk score > 1.5 (low similarity), skip RAG and ask for clarification | 1 day |
| Low | Migrate Chroma to Pinecone for cloud deployment | Persistent vector store across cold starts — eliminates 10s rebuild on Streamlit | 2 days |

---

## 13. Design Tradeoffs Accepted

| Decision | Tradeoff Accepted | Rationale |
|---|---|---|
| GPT-4o-mini over GPT-3.5-turbo | Higher cost | Safety rule compliance is non-negotiable in banking |
| Chroma (local) over Pinecone | Rebuilds on cold start (~10s) | Eliminates external paid dependency; acceptable for capstone |
| JSON long-term memory over Redis | Not scalable beyond single instance | Sufficient for this scope; migration path is clear |
| k=6 short-term window | Older context lost after 6 turns | Covers realistic conversation depth at lower token cost |
| Regex + LLM safety (two layers) | Slightly higher latency on safe queries | Defence-in-depth is required for financial use case |
| LangGraph over CrewAI | Less "agent role" abstraction | Explicit state schema and conditional routing give more control |
