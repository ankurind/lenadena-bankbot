"""
Tool definitions for LenaDena BankBot.
All tools are read-only — no data modification allowed.
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent / "data"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search LenaDena Bank's knowledge base (FAQs, policies, product documents)
    for information relevant to the customer's question.
    Use this for general questions about bank services, processes, and policies.
    """
    from agent.retrieval import retrieve, format_context
    docs = retrieve(query, k=3)
    if not docs:
        return "No relevant information found in the knowledge base for this query."
    return format_context(docs)


@tool
def get_product_rates(product_type: str) -> str:
    """
    Return the current rates and details for a specific LenaDena Bank product.
    product_type must be one of: 'fd', 'savings', 'personal_loan', 'home_loan',
    'car_loan', 'education_loan', 'gold_loan', 'credit_cards'.
    """
    products_path = DATA_DIR / "products.json"
    if not products_path.exists():
        return "Product data not available."

    products = json.loads(products_path.read_text())
    pt = product_type.lower().replace(" ", "_")

    if pt in ("fd", "fixed_deposit"):
        rates = products["fixed_deposits"]["rates"]
        lines = ["LenaDena Bank Fixed Deposit Rates:\n"]
        for r in rates:
            line = f"  {r['tenure']}: General {r['general_rate']}, Senior Citizen {r['senior_citizen_rate']}"
            if r.get("notes"):
                line += f" ({r['notes']})"
            lines.append(line)
        penalty = products["fixed_deposits"]["premature_withdrawal_penalty"]
        lines.append(f"\nPremature withdrawal penalty:")
        lines.append(f"  < 1 year: {penalty['less_than_1_year']}")
        lines.append(f"  ≥ 1 year: {penalty['1_year_or_more']}")
        lines.append(f"  Senior citizen: {penalty['senior_citizen']}")
        return "\n".join(lines)

    elif pt in ("savings", "savings_account"):
        lines = ["LenaDena Bank Savings Account Options:\n"]
        for acc in products["savings_accounts"]:
            lines.append(f"{acc['name']}:")
            lines.append(f"  Interest: {acc.get('interest_rate', acc.get('interest_rate_upto_1lakh', 'N/A'))}")
            lines.append(f"  Min balance: ₹{acc.get('min_balance', acc.get('min_balance_metro', 'N/A'))}")
        return "\n".join(lines)

    elif pt in ("personal_loan",):
        for loan in products["loans"]:
            if loan["type"] == "Personal Loan":
                return (
                    f"LenaDena Bank Personal Loan:\n"
                    f"  Max amount: {loan['max_amount']}\n"
                    f"  Interest rate: {loan['interest_rate_range']}\n"
                    f"  Tenure: {loan['tenure']}\n"
                    f"  Processing fee: {loan['processing_fee']}\n"
                    f"  Min CIBIL: {loan['min_cibil']}\n"
                    f"  Min monthly income (salaried): ₹{loan['min_monthly_income_salaried']:,}"
                )

    elif pt in ("home_loan",):
        for loan in products["loans"]:
            if loan["type"] == "Home Loan":
                return (
                    f"LenaDena Bank Home Loan:\n"
                    f"  Max amount: {loan['max_amount']}\n"
                    f"  Interest rate: {loan['interest_rate_range']}\n"
                    f"  Tenure: {loan['tenure']}\n"
                    f"  Processing fee: {loan['processing_fee']}\n"
                    f"  Min CIBIL: {loan['min_cibil']}"
                )

    elif pt in ("car_loan",):
        lines = ["LenaDena Bank Car Loans:"]
        for loan in products["loans"]:
            if "Car Loan" in loan["type"]:
                lines.append(f"\n{loan['type']}: {loan['max_amount']}, {loan['interest_rate_range']}, tenure {loan.get('tenure','N/A')}")
        return "\n".join(lines)

    elif pt in ("education_loan",):
        for loan in products["loans"]:
            if loan["type"] == "Education Loan":
                return (
                    f"LenaDena Bank Education Loan:\n"
                    f"  Max (India): {loan['max_amount_india']}\n"
                    f"  Max (Abroad): {loan['max_amount_abroad']}\n"
                    f"  Interest rate: {loan['interest_rate_range']}\n"
                    f"  Moratorium: {loan['moratorium']}"
                )

    elif pt in ("gold_loan",):
        for loan in products["loans"]:
            if loan["type"] == "Gold Loan":
                return (
                    f"LenaDena Bank Gold Loan:\n"
                    f"  LTV: {loan['ltv_ratio']}\n"
                    f"  Interest rate: {loan['interest_rate_range']}\n"
                    f"  Tenure: {loan['tenure']}\n"
                    f"  Disbursement: {loan['disbursement']}"
                )

    elif pt in ("credit_cards", "credit_card"):
        lines = ["LenaDena Bank Credit Cards:\n"]
        for card in products["credit_cards"]:
            lines.append(
                f"{card['name']}: Annual fee ₹{card['annual_fee']}, "
                f"Cashback {card['cashback']}, Limit {card['credit_limit_range']}, "
                f"Lounge: {card.get('lounge_access', 'None')}"
            )
        return "\n".join(lines)

    return (
        f"Product type '{product_type}' not recognised. "
        "Valid types: fd, savings, personal_loan, home_loan, car_loan, education_loan, gold_loan, credit_cards."
    )


@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalate the current customer query to a human banking advisor.
    Use this when: the query involves a dispute, fraud, account blockage,
    unresolved complaint, or any situation requiring personalised human judgment.
    Provide a brief reason for escalation.
    """
    ticket_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
    return (
        f"I've escalated your query to a LenaDena Bank human advisor.\n\n"
        f"**Escalation Ticket:** {ticket_id}\n"
        f"**Reason:** {reason}\n\n"
        f"A relationship manager will contact you within 2 business hours "
        f"on your registered mobile number.\n\n"
        f"You can also reach us directly:\n"
        f"  📞 1800-123-5362 (24x7 toll-free)\n"
        f"  ✉️  support@lenadenbank.in\n"
        f"  💬 Live chat on LenaApp (8 AM – 10 PM)"
    )


@tool
def check_eligibility_info(product: str) -> str:
    """
    Return general eligibility criteria for a LenaDena Bank product.
    product must be one of: 'personal_loan', 'home_loan', 'car_loan',
    'education_loan', 'credit_card', 'savings_account'.
    NOTE: This returns GENERAL criteria only — actual eligibility requires
    a formal credit assessment. This tool cannot approve or reject any application.
    """
    eligibility_map = {
        "personal_loan": (
            "LenaDena Bank Personal Loan — General Eligibility:\n"
            "  Age: 21–60 years (salaried), 21–65 years (self-employed)\n"
            "  Min monthly income: ₹25,000 (salaried)\n"
            "  Min annual income: ₹3 lakh (self-employed)\n"
            "  Min CIBIL score: 700\n"
            "  Min employment: 1 year with current employer (salaried), 2 years in business (self-employed)\n"
            "  ⚠️  Meeting these criteria does not guarantee approval. "
            "Final decision is subject to LenaDena Bank's credit assessment."
        ),
        "home_loan": (
            "LenaDena Bank Home Loan — General Eligibility:\n"
            "  Age: 21 years minimum; loan must mature before age 70\n"
            "  Min CIBIL score: 700\n"
            "  DTI ratio: Total EMIs must not exceed 50% of gross monthly income\n"
            "  Indian resident\n"
            "  ⚠️  Final approval subject to LenaDena Bank's credit assessment."
        ),
        "car_loan": (
            "LenaDena Bank Car Loan — General Eligibility:\n"
            "  Age: 21–65 years\n"
            "  Min CIBIL score: 680 (new), 700 (used)\n"
            "  Stable income source (salaried or self-employed)\n"
            "  ⚠️  Final approval subject to credit assessment."
        ),
        "education_loan": (
            "LenaDena Bank Education Loan — General Eligibility:\n"
            "  Student: Indian national, admission confirmed at recognised institution\n"
            "  Co-applicant (parent/guardian) required\n"
            "  Min CIBIL of co-applicant: 650\n"
            "  Collateral required for loans above ₹7.5 lakh"
        ),
        "credit_card": (
            "LenaDena Bank Credit Card — General Eligibility:\n"
            "  Classic: Min income ₹20,000/month\n"
            "  Select: Min income ₹40,000/month\n"
            "  Platinum: Min income ₹75,000/month (salaried) or ₹10L p.a. (self-employed)\n"
            "  Min CIBIL score: 700 for all cards\n"
            "  Age: 18+ years\n"
            "  ⚠️  Final limit and approval subject to credit assessment."
        ),
        "savings_account": (
            "LenaDena Bank Savings Account — Eligibility:\n"
            "  Indian resident (resident individual or NRI with NRE/NRO account type)\n"
            "  Valid KYC documents: photo ID + address proof + photograph\n"
            "  Minimum age: 18 years (minor accounts available with guardian)\n"
            "  No minimum income requirement"
        ),
    }
    key = product.lower().replace(" ", "_")
    return eligibility_map.get(
        key,
        f"Eligibility info for '{product}' not available. "
        "Please call 1800-123-5362 for details."
    )


# List of all tools for agent registration
ALL_TOOLS = [search_knowledge_base, get_product_rates, escalate_to_human, check_eligibility_info]
