"""
Phase 4: Add Knowledge & Retrieval (RAG)
LenaDena BankBot

Goal: Build RAG pipeline over LenaDena Bank documents. Compare with/without RAG.
      Handle missing-information cases gracefully.
Run: python phases/phase4_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.retrieval import build_vectorstore, retrieve, retrieve_with_scores, format_context

DIVIDER = "=" * 70

SYSTEM_NO_RAG = """You are LenaDena BankBot, a customer support assistant for LenaDena Bank.
Safety rules: REFUSE money movement, legal advice, credential requests.
Reasoning: Identify intent, check safety, then answer. State confidence level."""

SYSTEM_RAG = """You are LenaDena BankBot, a customer support assistant for LenaDena Bank.

Safety rules:
1. REFUSE any request to transfer money, approve loans, or perform transactions
2. Do NOT provide legal advice
3. ONLY use information from the provided context — do not invent rates or policies
4. If the context does not contain the answer, say:
   "I don't have that information currently. Please call 1800-123-5362 or visit lenadenbank.in."

Reasoning approach:
- Identify what the customer is asking and whether it is safe to answer
- Use ONLY the provided context to answer
- State your confidence level (High / Medium / Low)"""


def make_rag_responder(llm):
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG),
        ("human", "Context from LenaDena Bank knowledge base:\n{context}\n\nCustomer question: {query}"),
    ])
    chain = rag_prompt | llm | StrOutputParser()

    def respond(query: str) -> dict:
        docs = retrieve(query, k=3)
        context = format_context(docs)
        response = chain.invoke({"context": context, "query": query})
        return {"query": query, "context": context, "response": response, "num_docs": len(docs)}

    return respond


def make_no_rag_responder(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_NO_RAG),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return lambda q: chain.invoke({"query": q})


def main():
    print(DIVIDER)
    print("PHASE 4: RAG — KNOWLEDGE & RETRIEVAL — LenaDena BankBot")
    print(DIVIDER)

    print("\n[1/4] Building vector store from LenaDena Bank documents...")
    build_vectorstore(force_rebuild=True)
    print("Vector store ready.")

    print(f"\n[2/4] Testing retrieval quality")
    print(DIVIDER)
    retrieval_test_queries = [
        "What are the FD interest rates for 2 years?",
        "Is there a penalty for premature FD withdrawal?",
        "What documents do I need to open an account?",
    ]
    for q in retrieval_test_queries:
        results = retrieve_with_scores(q, k=3)
        print(f"\nQuery: {q}")
        for doc, score in results:
            print(f"  Score: {score:.4f} | Source: {doc.metadata.get('source')} | {doc.page_content[:80]}...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    rag_respond = make_rag_responder(llm)
    no_rag_respond = make_no_rag_responder(llm)

    print(f"\n[3/4] Comparison: WITHOUT RAG vs WITH RAG")
    print(DIVIDER)
    comparison_queries = [
        "What is the FD interest rate for 18 months at LenaDena Bank?",
        "What are the eligibility criteria for a home loan at LenaDena Bank?",
        "What is the penalty for premature FD withdrawal if I'm a senior citizen?",
    ]
    for q in comparison_queries:
        no_rag_resp = no_rag_respond(q)
        rag_result = rag_respond(q)
        print(f"\nQUERY: {q}")
        print(f"WITHOUT RAG:\n  {no_rag_resp[:200]}")
        print(f"\nWITH RAG:\n  {rag_result['response'][:200]}")
        print("-" * 60)

    print(f"\n[4/4] Missing information — graceful handling")
    print(DIVIDER)
    out_of_scope = [
        "What is LenaDena Bank's live USD to INR forex rate right now?",
        "Does LenaDena Bank offer cryptocurrency trading?",
    ]
    for q in out_of_scope:
        result = rag_respond(q)
        print(f"\nQ: {q}")
        print(f"  Retrieved docs: {result['num_docs']} (low relevance expected)")
        print(f"  A: {result['response'][:200]}")

    print(f"\n{DIVIDER}")
    print("RAG IMPROVEMENT SUMMARY")
    print(DIVIDER)
    print("""
| Aspect                    | Without RAG                        | With RAG                              |
|---------------------------|------------------------------------|---------------------------------------|
| FD rate accuracy          | LLM may hallucinate generic rates  | Returns exact rates from products.json|
| Eligibility details       | Generic banking knowledge          | LenaDena-specific CIBIL, income rules |
| Senior citizen FD penalty | May not know bank-specific policy  | "No penalty" from policies.md         |
| Missing info handling     | May confidently hallucinate        | Explicit "I don't have that info"     |
| Source grounding          | None — relies on training data     | Every response traceable to a doc     |

Conclusion: RAG eliminates hallucination of LenaDena Bank-specific details.
The "only use provided context" instruction is the key safety net.
""")


if __name__ == "__main__":
    main()
