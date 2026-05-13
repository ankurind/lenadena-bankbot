"""
Phase Evidence page — shows source code + captured output for each phase script.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Phase Evidence — LenaDena BankBot",
    page_icon="📓",
    layout="wide",
)
st.title("📓 Phase Evidence")
st.caption(
    "Source code and captured output for each development phase. "
    "Run `python run_all.py` locally to refresh the output files."
)

ROOT = Path(__file__).parent.parent.parent
PHASES_DIR = ROOT / "phases"
LOGS_DIR = ROOT / "logs"
NOTEBOOKS_DIR = ROOT / "notebooks"

PHASES = [
    (1,  "Phase 1: Problem Framing",               None,                          NOTEBOOKS_DIR / "phase1_problem_framing.md"),
    (2,  "Phase 2: Baseline Agent",                PHASES_DIR / "phase2_baseline.py",       LOGS_DIR / "phase2_output.txt"),
    (3,  "Phase 3: LLM + Prompt Engineering",      PHASES_DIR / "phase3_llm_prompting.py",  LOGS_DIR / "phase3_output.txt"),
    (4,  "Phase 4: RAG",                           PHASES_DIR / "phase4_rag.py",             LOGS_DIR / "phase4_output.txt"),
    (5,  "Phase 5: Tool Usage",                    PHASES_DIR / "phase5_tools.py",           LOGS_DIR / "phase5_output.txt"),
    (6,  "Phase 6: LangGraph + Memory",            PHASES_DIR / "phase6_memory_planning.py", LOGS_DIR / "phase6_output.txt"),
    (7,  "Phase 7: Adaptive Behaviour",            PHASES_DIR / "phase7_adaptive.py",        LOGS_DIR / "phase7_output.txt"),
    (8,  "Phase 8: Deployment Readiness",          PHASES_DIR / "phase8_deployment.py",      LOGS_DIR / "phase8_output.txt"),
    (9,  "Phase 9: Evaluation & Review",           PHASES_DIR / "phase9_evaluation.py",      LOGS_DIR / "phase9_output.txt"),
]

tabs = st.tabs([p[1] for p in PHASES])

for tab, (num, title, src_path, output_path) in zip(tabs, PHASES):
    with tab:
        if num == 1:
            # Phase 1 is a markdown document
            if output_path and output_path.exists():
                st.markdown(output_path.read_text())
            else:
                st.warning(f"`{output_path}` not found.")
            continue

        col_src, col_out = st.columns([1, 1])

        with col_src:
            st.subheader("Source Code")
            if src_path and src_path.exists():
                st.code(src_path.read_text(), language="python")
            else:
                st.warning(f"Source not found: `{src_path}`")

        with col_out:
            st.subheader("Captured Output")
            if output_path and output_path.exists():
                output_text = output_path.read_text()
                st.code(output_text, language="text")
            else:
                st.info(
                    f"Output not yet generated for this phase.\n\n"
                    f"Run: `python run_all.py --phase {num}`\n\n"
                    f"This will execute the phase script and save output to "
                    f"`logs/phase{num}_output.txt`."
                )
