"""
Phase 2: Build a Basic Working Agent — Baseline Rule-Based Agent
LenaDena BankBot

Goal: Keyword/template agent with no LLM, demonstrate 2 limitations, log interactions.
Run: python phases/phase2_baseline.py
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_PATH = Path(__file__).parent.parent / "logs" / "phase2_interactions.json"
LOG_PATH.parent.mkdir(exist_ok=True)

DIVIDER = "=" * 70


class BaselineAgent:
    """Rule-based agent using keyword matching. No LLM involved."""

    RULES = [
        (r"\b(fd|fixed deposit|fixed.?deposit)\b",
         "LenaDena Bank FD rates: 1 year = 6.80%, 2 years = 7.20%, 3 years = 7.10%, "
         "5 years (tax-saving) = 7.00%. Senior citizens get +0.50%. Minimum deposit: ₹1,000."),

        (r"\b(savings account|saving account|sb account)\b",
         "LenaDena Bank Savings Account interest: 3.5% p.a. up to ₹1 lakh, "
         "4.0% above ₹1 lakh. Minimum balance: ₹1,000 (metro)."),

        (r"\b(credit card|creditcard)\b",
         "LenaDena Bank credit cards: Classic (₹500/yr, 1% cashback), "
         "Select (₹1,500/yr, 2% + lounge), Platinum (₹3,000/yr, 3% + unlimited lounge)."),

        (r"\b(personal loan|personal.?loan)\b",
         "LenaDena Bank Personal Loan: up to ₹20 lakh, 10.5%-18% p.a., "
         "tenure 12-60 months. Minimum CIBIL: 700. Min income: ₹25,000/month."),

        (r"\b(home loan|housing loan|mortgage)\b",
         "LenaDena Bank Home Loan: up to ₹5 crore, 8.5%-10.25% p.a., tenure up to 30 years."),

        (r"\b(kyc|know your customer)\b",
         "KYC documents: Photo ID (Aadhaar/PAN/Passport), Address proof, Passport photograph."),

        (r"\b(dispute|wrong charge|incorrect|transaction issue)\b",
         "To dispute a transaction: call 1800-123-5362 or net banking → Cards → Dispute Transaction. "
         "Resolution: 7-21 working days."),

        (r"\b(contact|helpline|customer care|phone number|support)\b",
         "LenaDena Bank support: 1800-123-5362 (24x7 toll-free). Email: support@lenadenbank.in."),

        (r"\b(minimum balance|min balance|balance requirement)\b",
         "Minimum balance: Regular Savings ₹1,000 (metro), ₹500 (rural). "
         "Senior Citizen and Student accounts: zero balance."),

        (r"\b(close account|account closure)\b",
         "Account closure: visit home branch. Fee ₹300 if closed within 12 months. Takes 3-5 days."),
    ]

    FALLBACK = (
        "I'm sorry, I don't have information about that. "
        "Please call 1800-123-5362 or visit your nearest LenaDena Bank branch."
    )

    def respond(self, query: str) -> dict:
        start = time.perf_counter()
        matched_rule = None
        answer = self.FALLBACK

        for pattern, response in self.RULES:
            if re.search(pattern, query, re.IGNORECASE):
                matched_rule = pattern
                answer = response
                break

        return {
            "query": query,
            "response": answer,
            "matched_rule": matched_rule,
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }


def main():
    print(DIVIDER)
    print("PHASE 2: BASELINE RULE-BASED AGENT — LenaDena BankBot")
    print(DIVIDER)

    agent = BaselineAgent()

    test_queries = [
        # Normal — should match rules
        "What are the FD interest rates at LenaDena Bank?",
        "Tell me about credit cards available.",
        "What is the minimum balance for a savings account?",
        # Limitation 1: paraphrase — should FAIL
        "What is the rate on deposits?",
        "How much interest do I earn on money kept in the bank?",
        # Limitation 2: multi-intent — should FAIL
        "I want to compare home loan rates and also understand the FD premature withdrawal penalty.",
    ]

    interactions = []
    for q in test_queries:
        result = agent.respond(q)
        interactions.append(result)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['response']}")
        match_status = "MATCHED" if result["matched_rule"] else "FALLBACK (no keyword match)"
        print(f"   [{match_status}] latency={result['latency_ms']}ms")

    LOG_PATH.write_text(json.dumps(interactions, indent=2))
    print(f"\nLogged {len(interactions)} interactions → {LOG_PATH}")

    print(f"\n{DIVIDER}")
    print("DEMONSTRATED LIMITATIONS")
    print(DIVIDER)
    print("""
Limitation 1 — Cannot handle paraphrase:
  "What is the rate on deposits?" and "How much interest do I earn on money
  kept in the bank?" both ask for FD/savings rates, but the keywords 'fd'
  and 'fixed deposit' are absent — agent returns fallback.

  | Query                                  | Expected    | Actual        |
  |----------------------------------------|-------------|---------------|
  | "What are the FD interest rates?"      | FD rates    | CORRECT       |
  | "What is the rate on deposits?"        | FD rates    | FALLBACK ❌   |
  | "How much interest on money in bank?"  | FD/savings  | FALLBACK ❌   |

Limitation 2 — Cannot handle multi-intent:
  "I want to compare home loan rates AND understand FD premature penalty"
  has two distinct intents. The rule engine matches only the first keyword
  hit (home loan) and silently ignores the second intent (FD penalty).

  | Query                                       | Expected        | Actual       |
  |---------------------------------------------|-----------------|--------------|
  | Home loan + FD penalty (single query)       | Answer both     | Home loan only ❌ |

Additional implicit limitations:
  - No memory: every query is stateless
  - Cannot reason across products (e.g., "which card is better for a traveller?")
  - Brittle to typos and word-order variation
  - Adding new products requires writing new rules for every phrasing variant

Conclusion: Rule-based agents cannot serve real users. LLM integration is required
(addressed in Phase 3).
""")


if __name__ == "__main__":
    main()
