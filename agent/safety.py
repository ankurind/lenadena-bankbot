"""
Safety guardrails for LenaDena BankBot.
Enforces Scenario 2 safety requirements:
  - Refuse money movement, approvals, legal advice
  - No PII in logs
  - Escalation triggers
"""

import re
from typing import Tuple

# ── Unsafe intent patterns ────────────────────────────────────────────────────

REFUSE_PATTERNS = [
    # Money movement — covers ₹ symbol, rupees, amounts, account-to-account transfers
    (r"\b(transfer|send|move|wire|remit)\b.{0,40}(₹|\b(money|fund|amount|rupee|rs\.?|inr)\b)",
     "money_movement"),
    (r"\b(transfer|send|move)\b.{0,40}\b(account|upi|neft|rtgs|imps)\b",
     "money_movement"),
    (r"\b(deposit|withdraw)\b.{0,20}\b(now|immediately|for me|my account)\b",
     "money_movement"),
    # Loan/account approvals
    (r"\b(approve|sanction|grant|give)\b.{0,20}\b(loan|credit|overdraft|limit)\b",
     "approval_request"),
    (r"\bopen\b.{0,15}\b(account|fd|rd)\b.{0,15}\b(for me|now|immediately)\b",
     "account_action"),
    # Legal advice
    (r"\b(legal|law|lawsuit|sue|court|rbi complaint|consumer forum)\b",
     "legal_advice"),
    (r"\b(illegal|fraud by bank|bank cheated)\b",
     "legal_advice"),
    # Credential/PIN requests
    (r"\b(otp|pin|password|cvv|atm pin|mpin)\b.{0,20}\b(share|give|tell|send)\b",
     "credential_phishing"),
]

ESCALATE_PATTERNS = [
    r"\b(complaint|grievance|unhappy|dissatisfied|not working|issue not resolved)\b",
    r"\b(my account is blocked|can't access|locked out)\b",
    r"\b(money debited|wrong transaction|fraudulent|unauthorized)\b",
    r"\b(emergency|urgent|immediately|asap)\b",
]

# ── PII scrubbing patterns ────────────────────────────────────────────────────

PII_PATTERNS = [
    (r"\b\d{10}\b", "<PHONE>"),                              # 10-digit phone numbers
    (r"\b\d{12}\b", "<AADHAAR>"),                            # 12-digit Aadhaar
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "<PAN>"),                # PAN card
    (r"\b\d{9,18}\b", "<ACCOUNT_NO>"),                       # Bank account numbers
    (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "<EMAIL>"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "<CARD_NO>"),
]

# ── Refusal messages ──────────────────────────────────────────────────────────

REFUSAL_MESSAGES = {
    "money_movement": (
        "I'm sorry, but I'm unable to initiate or process any financial transactions. "
        "LenaDena BankBot is an information and advisory assistant only. "
        "To transfer funds, please use LenaApp, net banking, or visit your nearest branch. "
        "If you need help, call 1800-123-5362."
    ),
    "approval_request": (
        "I'm not authorised to approve loans, credit limits, or open accounts. "
        "These decisions require formal credit assessment by our banking team. "
        "Please apply through our website, LenaApp, or visit a branch. "
        "I can provide information about eligibility criteria and products if that helps."
    ),
    "account_action": (
        "Account opening, closure, and modifications require verification at a branch or "
        "through our official digital channels. I can guide you through the process, "
        "but I cannot perform these actions directly."
    ),
    "legal_advice": (
        "I'm unable to provide legal advice or opinions on legal matters. "
        "For disputes, please contact our Grievance Officer at grievance.officer@lenadenbank.in "
        "or approach the Banking Ombudsman. I can share our general dispute resolution process "
        "if that would help."
    ),
    "credential_phishing": (
        "LenaDena Bank will NEVER ask for your OTP, PIN, password, or CVV — not even "
        "through this chat. Please do not share these with anyone. "
        "If someone is requesting these, call our fraud helpline immediately: 1800-123-5362."
    ),
}

ESCALATION_MESSAGE = (
    "This query requires attention from one of our human advisors. "
    "I'm escalating this for you now.\n\n"
    "**Escalation Reference:** {ticket_id}\n"
    "A relationship manager will contact you within 2 business hours on your registered number. "
    "Alternatively, call us at 1800-123-5362 or email support@lenadenbank.in."
)


def check_safety(query: str) -> Tuple[str, str | None]:
    """
    Returns (verdict, reason_code).
    verdict: 'REFUSE' | 'ESCALATE' | 'PROCEED'
    """
    q = query.lower()

    for pattern, reason in REFUSE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "REFUSE", reason

    for pattern in ESCALATE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "ESCALATE", "sensitive_case"

    return "PROCEED", None


def get_refusal_message(reason_code: str) -> str:
    return REFUSAL_MESSAGES.get(reason_code, REFUSAL_MESSAGES["money_movement"])


def get_escalation_message(ticket_id: str) -> str:
    return ESCALATION_MESSAGE.format(ticket_id=ticket_id)


def scrub_pii(text: str) -> str:
    """Remove PII from text before logging."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def is_pii_query(query: str) -> bool:
    """True if the query contains PII that should not be passed to the LLM."""
    for pattern, _ in PII_PATTERNS:
        if re.search(pattern, query):
            return True
    return False
