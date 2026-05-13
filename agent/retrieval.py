"""
RAG pipeline for LenaDena BankBot.
Loads data/, chunks, embeds via OpenAI, stores in Chroma.
"""

import json
import os
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_vectorstore: Optional[Chroma] = None


def _load_documents() -> list[Document]:
    docs = []

    # Load faq.json
    faq_path = DATA_DIR / "faq.json"
    if faq_path.exists():
        faqs = json.loads(faq_path.read_text())
        for item in faqs:
            content = f"Category: {item['category']}\nQ: {item['question']}\nA: {item['answer']}"
            docs.append(Document(page_content=content, metadata={"source": "faq", "id": item["id"], "category": item["category"]}))

    # Load products.json — flatten to text
    products_path = DATA_DIR / "products.json"
    if products_path.exists():
        products = json.loads(products_path.read_text())
        # FD rates
        fd_text = "LenaDena Bank Fixed Deposit Rates:\n"
        for rate in products.get("fixed_deposits", {}).get("rates", []):
            fd_text += f"Tenure: {rate['tenure']} — General: {rate['general_rate']}, Senior Citizen: {rate['senior_citizen_rate']}\n"
        docs.append(Document(page_content=fd_text, metadata={"source": "products", "category": "fd_rates"}))

        # Credit cards
        for card in products.get("credit_cards", []):
            card_text = (
                f"LenaDena Bank Credit Card: {card['name']}\n"
                f"Annual fee: ₹{card['annual_fee']}\n"
                f"Cashback: {card['cashback']}\n"
                f"Credit limit range: {card['credit_limit_range']}\n"
                f"Lounge access: {card.get('lounge_access', 'None')}\n"
                f"Eligibility: Min income {card.get('eligibility_min_income', card.get('eligibility', 'N/A'))}"
            )
            docs.append(Document(page_content=card_text, metadata={"source": "products", "category": "credit_cards", "name": card["name"]}))

        # Loans
        for loan in products.get("loans", []):
            loan_text = (
                f"LenaDena Bank {loan['type']}:\n"
                f"Maximum amount: {loan.get('max_amount', loan.get('max_amount_abroad', 'N/A'))}\n"
                f"Interest rate: {loan['interest_rate_range']}\n"
                f"Tenure: {loan.get('tenure', 'N/A')}\n"
                f"Minimum CIBIL: {loan.get('min_cibil', 'N/A')}\n"
                f"Processing fee: {loan.get('processing_fee', 'N/A')}"
            )
            docs.append(Document(page_content=loan_text, metadata={"source": "products", "category": "loans", "type": loan["type"]}))

        # Savings accounts
        for acc in products.get("savings_accounts", []):
            acc_text = (
                f"LenaDena Bank Savings Account: {acc['name']}\n"
                f"Interest rate: {acc.get('interest_rate', acc.get('interest_rate_upto_1lakh', 'N/A'))}\n"
                f"Minimum balance: ₹{acc.get('min_balance', acc.get('min_balance_metro', 'N/A'))}\n"
                f"Features: {', '.join(acc.get('features', []))}"
            )
            docs.append(Document(page_content=acc_text, metadata={"source": "products", "category": "savings_accounts", "name": acc["name"]}))

    # Load policies.md
    policies_path = DATA_DIR / "policies.md"
    if policies_path.exists():
        policy_text = policies_path.read_text()
        docs.append(Document(page_content=policy_text, metadata={"source": "policies"}))

    return docs


def _chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Build or load the Chroma vector store."""
    global _vectorstore

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not force_rebuild and CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        return _vectorstore

    print("Building vector store from documents...")
    docs = _load_documents()
    chunks = _chunk_documents(docs)
    print(f"  Loaded {len(docs)} documents → {len(chunks)} chunks")

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  Vector store built and persisted to {CHROMA_DIR}")
    return _vectorstore


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 3) -> list[Document]:
    """Semantic search — returns top-k relevant chunks."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def retrieve_with_scores(query: str, k: int = 3) -> list[tuple[Document, float]]:
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k)


def format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a context string for the prompt."""
    if not docs:
        return "No relevant information found in the knowledge base."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
