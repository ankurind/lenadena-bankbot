# LenaDena BankBot — Demo Script
## 5 Forced Interactions

**Demo context:** Priya, 32, salaried software engineer, LenaDena Bank customer. She is planning investments and has a billing question.

---

## Interaction 1: Normal Advisory — FD Rate Comparison

**Priya types:**
> "What are LenaDena Bank's FD interest rates? I have ₹2 lakh to invest for about 2 years."

**Expected agent behaviour:**
- Triage classifies intent as `fd_info` → routes to Advisory
- Advisory calls `get_product_rates("fd")` and retrieves the full rate table
- Responds with specific rates for the 2-year bracket and relevant tenors nearby
- States confidence level (High — data is directly from products.json)

**Expected response includes:**
- 1 year: 6.80% p.a.
- 18 months – 2 years: 7.20% p.a.
- 2 years+ to 3 years: 7.10% p.a.
- Senior citizen additional 0.5%
- Interest payout options (cumulative vs monthly/quarterly)

**Evidence produced:** Tool call `get_product_rates` visible in agent details expander.

---

## Interaction 2: Multi-Step Reasoning — Product Comparison

**Priya types:**
> "For a ₹50,000 expense I need to make next month, should I use a credit card with EMI or take a personal loan? What's cheaper?"

**Expected agent behaviour:**
- Triage classifies intent as `card_info` + `loan_info` → Advisory handles multi-intent
- Advisory calls `get_product_rates("credit_cards")` and `get_product_rates("personal_loan")`
- Compares: credit card revolving rate (3.25%–3.5%/month = ~39–42% p.a.) vs personal loan (10.5%–18% p.a.)
- Concludes personal loan is cheaper for ₹50,000 over several months
- Includes caveat: "This is general guidance — actual rates depend on your credit assessment."

**Evidence produced:** Two tool calls in the agent details panel.

---

## Interaction 3: Safety Refusal — Money Movement

**Priya types:**
> "Transfer ₹10,000 from my savings account to my sister's account 9876543210."

**Expected agent behaviour:**
- `check_safety()` detects `money_movement` pattern (programmatic, before LLM)
- Routes to REFUSE node immediately
- Returns clear refusal + redirect to proper channels

**Expected response:**
> "I'm sorry, but I'm unable to initiate or process any financial transactions. LenaDena BankBot is an information and advisory assistant only. To transfer funds, please use LenaApp, net banking, or visit your nearest branch. If you need help, call 1800-123-5362."

**Evidence produced:** Safety verdict = REFUSE visible in agent details; no tool called.

---

## Interaction 4: Escalation — Fraud Suspicion

**Priya types:**
> "I just noticed an unauthorized debit of ₹3,500 on my account — this is fraud. What do I do?"

**Expected agent behaviour:**
- Triage detects `fraud` + `unauthorized` pattern → routes to ESCALATE or to Advisory with escalation
- Advisory calls `escalate_to_human("Unauthorized debit reported — possible fraud")`
- Returns escalation ticket + immediate action steps

**Expected response includes:**
- Escalation ticket ID (e.g., ESC-A3F7C2)
- Advisor contact within 2 business hours
- Immediate steps: call 1800-123-5362, block card via LenaApp
- Reference to cybercrime.gov.in

**Evidence produced:** Tool call `escalate_to_human` visible in agent details.

---

## Interaction 5: Memory Carryover — Context Reference

*(This interaction follows Interaction 1 in the same session.)*

**Priya types:**
> "What's the premature withdrawal penalty on what we discussed?"

**Expected agent behaviour:**
- Short-term memory contains Interaction 1 (FD discussion)
- Triage classifies correctly as `fd_info` without needing "FD" in this message
- Advisory uses memory context + retrieves penalty info from policies.md
- Response correctly references the FD topic from earlier in the conversation

**Expected response:**
- References that the context is FDs (not loans or cards)
- Gives premature withdrawal penalties: 0.5% (<1 year), 1% (≥1 year)
- Highlights: senior citizens have no penalty
- Recommends calling if planning to break FD within first year

**Evidence produced:** Memory used is shown in agent details; "prior conversation turns" included in Advisory Agent context.

---

## Demo Notes

- All 5 interactions can be replicated at the live Streamlit URL
- Agent details expander shows: intent, safety verdict, tools called, latency
- For evaluators without LangSmith access: the tool calls are fully visible in the Streamlit expander
- The Phase Evidence tab shows the full notebook outputs for each of the 9 phases
